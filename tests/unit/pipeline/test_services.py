from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from gs_video.camera.classify import CameraKind
from gs_video.camera.opencv_solver import CameraSolution
from gs_video.camera.serialization import (
    MappedTrajectory,
    read_camera_solution,
    read_mapped_trajectory,
    write_camera_solution,
    write_mapped_trajectory,
)
from gs_video.domain.contracts import (
    MaskSequence,
    Prompt,
    RenderSequence,
    SegmentationBackend,
)
from gs_video.domain.errors import CancelledError, RepairableError, UnsupportedMaterialError
from gs_video.domain.models import (
    ArtifactCategory,
    ArtifactRef,
    ArtifactRole,
    LocalGroundAnchor,
    Project,
    SceneSummary,
    SourcePerspectiveCalibration,
    StageName,
    StageState,
    StageStatus,
    SubjectPromptState,
    SynthesisConstraintMode,
    SynthesisPlacement,
    VideoSummary,
)
from gs_video.media.export import ExportResult
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter, discard_progress
from gs_video.pipeline.services import (
    CameraSolveWorkflowService,
    CompositeWorkflowService,
    ExportWorkflowService,
    MediaIngestService,
    RendererWorkflowService,
    SegmentWorkflowService,
    TrajectoryMapWorkflowService,
    WorkflowPaths,
    _frame_inventory,
    _preview_size,
    _sha256,
)
from gs_video.scene.worker_client import RendererWorkerIdentity
from gs_video.scene.worker_protocol import RenderSequenceRequest
import gs_video.pipeline.services as workflow_services


CacheKey = str
ProjectMutation = Callable[[Project], None]


def key(character: str) -> CacheKey:
    return character * 64


def artifact_ref(
    project: Project,
    category: ArtifactCategory,
    cache_key: str,
    member: str | None = None,
) -> ArtifactRef:
    return ArtifactRef(
        project_id=project.project_id,
        category=category,
        cache_key=cache_key,
        member=member,
    )


def artifact_path(root: Path, reference: ArtifactRef) -> Path:
    return root / reference.relative_path()


def write_rgb(path: Path, size: tuple[int, int], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value, value, value)).save(path)


def write_mask(path: Path, size: tuple[int, int], value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, value).save(path)


def replace_rgb_with_same_size(path: Path, value: int) -> None:
    with Image.open(path) as image:
        size = image.size
    original = path.read_bytes()
    replacement = path.with_name(f".{path.stem}-replacement{path.suffix}")
    for candidate in (value, 0, 32, 64, 96, 128, 192, 255):
        write_rgb(replacement, size, candidate)
        remaining = len(original) - replacement.stat().st_size
        if remaining >= 0 and replacement.read_bytes() != original:
            with replacement.open("ab") as stream:
                stream.write(b"\0" * remaining)
            break
    assert replacement.stat().st_size == len(original)
    assert replacement.read_bytes() != original
    replacement.replace(path)


def replace_bytes_with_same_size(path: Path, payload: bytes) -> None:
    assert len(payload) == path.stat().st_size
    replacement = path.with_name(f".{path.name}-replacement")
    replacement.write_bytes(payload)
    replacement.replace(path)


def source_project(root: Path, *, frame_count: int | None = 3) -> Project:
    source = root / "source" / "source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source-video")
    project = Project(name="production")
    project.source_video = "source/source.mp4"
    project.workflow.source_summary = VideoSummary(
        filename="source.mp4",
        size=source.stat().st_size,
        sha256="04c767fd3d68b42476a64bb0c0edec569938a96cf2af218e0af8973e52792233",
        width=8,
        height=6,
        duration_seconds=10,
        fps="24/1",
        has_audio=True,
        frame_count=frame_count,
    )
    return project


@dataclass
class FakeRepository:
    project: Project
    updates: int = 0

    def update(self, mutation: ProjectMutation) -> Project:
        self.updates += 1
        mutation(self.project)
        return self.project.model_copy(deep=True)


class FakeMediaBackend:
    def __init__(self, identity: str = "fake-ffmpeg-1") -> None:
        self.identity = identity
        self.source_calls = 0
        self.proxy_calls = 0

    def extract_source_frames(self, source: Path, output_dir: Path) -> list[Path]:
        del source
        self.source_calls += 1
        return [
            self._source_frame(output_dir, index, value)
            for index, value in enumerate((32, 64, 96), start=1)
        ]

    @staticmethod
    def _source_frame(output_dir: Path, index: int, value: int) -> Path:
        path = output_dir / f"{index:06d}.png"
        write_rgb(path, (8, 6), value)
        return path

    def extract_proxy_frames(
        self, source: Path, output_dir: Path, max_height: int
    ) -> list[Path]:
        del source, max_height
        self.proxy_calls += 1
        paths: list[Path] = []
        for index, value in enumerate((32, 64, 96), start=1):
            path = output_dir / f"{index:06d}.jpg"
            write_rgb(path, (4, 3), value)
            paths.append(path)
        return paths


def test_ingest_registers_proxy_and_full_resolution_frames(tmp_path: Path) -> None:
    project = source_project(tmp_path)
    backend = FakeMediaBackend()
    service = MediaIngestService(WorkflowPaths(tmp_path), backend)

    result = service.run(project, CancellationToken(), discard_progress)

    assert result.artifacts[ArtifactRole.PROXY_FRAMES] == artifact_ref(
        project, ArtifactCategory.PROXIES, result.cache_key
    )
    assert result.artifacts[ArtifactRole.SOURCE_FRAMES] == artifact_ref(
        project, ArtifactCategory.FRAMES, result.cache_key
    )
    with Image.open(
        artifact_path(tmp_path, result.artifacts[ArtifactRole.SOURCE_FRAMES])
        / "000001.png"
    ) as image:
        assert image.mode == "RGB"
        assert image.size == (8, 6)
    assert backend.source_calls == 1
    assert backend.proxy_calls == 1


def test_ingest_persists_discovered_frame_count_once(tmp_path: Path) -> None:
    project = source_project(tmp_path, frame_count=None)
    repository = FakeRepository(project.model_copy(deep=True))
    service = MediaIngestService(
        WorkflowPaths(tmp_path, update_project=repository.update), FakeMediaBackend()
    )

    service.run(project, CancellationToken(), discard_progress)

    assert repository.updates == 1
    assert repository.project.workflow.source_summary is not None
    assert repository.project.workflow.source_summary.frame_count == 3
    assert project.workflow.source_summary is not None
    assert project.workflow.source_summary.frame_count == 3


def test_discovered_frame_count_does_not_change_ingest_cache_key(tmp_path: Path) -> None:
    project = source_project(tmp_path, frame_count=None)
    backend = FakeMediaBackend()
    service = MediaIngestService(WorkflowPaths(tmp_path), backend)

    first = service.run(project, CancellationToken(), discard_progress)
    second = service.run(project, CancellationToken(), discard_progress)

    assert second.cache_key == first.cache_key
    assert backend.source_calls == 1
    assert backend.proxy_calls == 1


def test_ingest_cache_key_includes_backend_identity(tmp_path: Path) -> None:
    project = source_project(tmp_path)

    first = MediaIngestService(
        WorkflowPaths(tmp_path), FakeMediaBackend("fake-ffmpeg-1")
    ).run(project, CancellationToken(), discard_progress)
    second = MediaIngestService(
        WorkflowPaths(tmp_path), FakeMediaBackend("fake-ffmpeg-2")
    ).run(project, CancellationToken(), discard_progress)

    assert first.cache_key != second.cache_key


def test_ingest_cache_hit_revalidates_proxy_dimensions(tmp_path: Path) -> None:
    project = source_project(tmp_path)
    backend = FakeMediaBackend()
    service = MediaIngestService(WorkflowPaths(tmp_path), backend)
    first = service.run(project, CancellationToken(), discard_progress)
    proxy_dir = artifact_path(tmp_path, first.artifacts[ArtifactRole.PROXY_FRAMES])
    for index in range(1, 4):
        write_rgb(proxy_dir / f"{index:06d}.jpg", (16, 12), index)

    with pytest.raises(RepairableError, match="代理帧尺寸"):
        service.run(project, CancellationToken(), discard_progress)

    assert backend.proxy_calls == 1


def test_ingest_rejects_source_replacement_by_backend_before_publication(
    tmp_path: Path,
) -> None:
    project = source_project(tmp_path)

    class ReplacingBackend(FakeMediaBackend):
        def extract_source_frames(
            self, source: Path, output_dir: Path
        ) -> list[Path]:
            frames = super().extract_source_frames(source, output_dir)
            replace_bytes_with_same_size(source, b"source-V1deo")
            return frames

    with pytest.raises(RepairableError, match="源视频|authority|变化|摘要"):
        MediaIngestService(WorkflowPaths(tmp_path), ReplacingBackend()).run(
            project, CancellationToken(), discard_progress
        )

    assert not any((tmp_path / "frames").glob("?" * 64))


@pytest.mark.parametrize(("width", "height"), [(1, 6), (8, 1), (7, 6), (8, 5)])
def test_ingest_rejects_dimensions_that_cannot_be_exported_without_resizing(
    tmp_path: Path, width: int, height: int
) -> None:
    project = source_project(tmp_path)
    assert project.workflow.source_summary is not None
    project.workflow.source_summary.width = width
    project.workflow.source_summary.height = height
    backend = FakeMediaBackend()

    with pytest.raises(RepairableError, match="偶数|yuv420p|编码"):
        MediaIngestService(WorkflowPaths(tmp_path), backend).run(
            project, CancellationToken(), discard_progress
        )

    assert backend.source_calls == 0
    assert backend.proxy_calls == 0


class FakeSegmenter:
    backend = SegmentationBackend.EDGETAM
    worker_prefix = ("fake-python",)

    def __init__(self, root: Path) -> None:
        self.model_config = root / "models" / "config.yaml"
        self.checkpoint = root / "models" / "model.pt"
        self.model_config.parent.mkdir(parents=True, exist_ok=True)
        self.model_config.write_bytes(b"config")
        self.checkpoint.write_bytes(b"checkpoint")
        self.output_dir: Path | None = None

    def segment(
        self,
        frames: list[Path],
        prompt: Prompt,
        output_dir: Path,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> MaskSequence:
        assert prompt == Prompt(frame_index=0, x=1, y=1)
        assert not output_dir.exists()
        output_dir.mkdir()
        self.output_dir = output_dir
        for index, frame in enumerate(frames, start=1):
            token.raise_if_cancelled()
            with Image.open(frame) as image:
                write_mask(output_dir / f"{index:06d}.png", image.size)
            emit(index, len(frames), f"mask {index}")
        return MaskSequence(output_dir, len(frames))


class ReplacingSegmenter(FakeSegmenter):
    def segment(
        self,
        frames: list[Path],
        prompt: Prompt,
        output_dir: Path,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> MaskSequence:
        replace_rgb_with_same_size(frames[0], 200)
        return super().segment(frames, prompt, output_dir, emit, token)


def ingest_succeeded(project: Project, root: Path, cache_key: str = key("a")) -> Project:
    frames = root / "frames" / cache_key
    proxies = root / "proxies" / cache_key
    for index, value in enumerate((32, 64, 96), start=1):
        write_rgb(frames / f"{index:06d}.png", (8, 6), value)
        write_rgb(proxies / f"{index:06d}.jpg", (4, 3), value)
    project.stages[StageName.INGEST] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=cache_key,
        artifacts={
            ArtifactRole.SOURCE_FRAMES: artifact_ref(
                project, ArtifactCategory.FRAMES, cache_key
            ),
            ArtifactRole.PROXY_FRAMES: artifact_ref(
                project, ArtifactCategory.PROXIES, cache_key
            ),
        },
    )
    return project


def test_segment_uses_proxy_frames_and_publishes_flat_mask_inventory(tmp_path: Path) -> None:
    project = ingest_succeeded(source_project(tmp_path), tmp_path)
    project.workflow.subject_prompt = SubjectPromptState(frame_index=0, x=1, y=1)
    segmenter = FakeSegmenter(tmp_path)
    service = SegmentWorkflowService(WorkflowPaths(tmp_path), segmenter)

    result = service.run(project, CancellationToken(), discard_progress)

    relative = result.artifacts[ArtifactRole.SUBJECT_MASKS]
    assert relative == artifact_ref(
        project, ArtifactCategory.MASKS, result.cache_key
    )
    assert sorted(path.name for path in artifact_path(tmp_path, relative).iterdir()) == [
        "000001.png",
        "000002.png",
        "000003.png",
    ]
    assert segmenter.output_dir is not None
    assert segmenter.output_dir.name == "worker-output"
    assert not segmenter.output_dir.exists()


def test_segment_rejects_stale_ingest_artifact_authority(tmp_path: Path) -> None:
    project = ingest_succeeded(source_project(tmp_path), tmp_path)
    project.workflow.subject_prompt = SubjectPromptState(frame_index=0, x=1, y=1)
    project.stages[StageName.INGEST].artifacts[ArtifactRole.PROXY_FRAMES] = artifact_ref(
        project, ArtifactCategory.PROXIES, key("b")
    )

    with pytest.raises(RepairableError, match="cache|缓存|artifact|产物"):
        SegmentWorkflowService(WorkflowPaths(tmp_path), FakeSegmenter(tmp_path)).run(
            project, CancellationToken(), discard_progress
        )


def test_segment_cache_hit_revalidates_mask_count_and_proxy_sizes(
    tmp_path: Path,
) -> None:
    project = ingest_succeeded(source_project(tmp_path), tmp_path)
    project.workflow.subject_prompt = SubjectPromptState(frame_index=0, x=1, y=1)
    service = SegmentWorkflowService(WorkflowPaths(tmp_path), FakeSegmenter(tmp_path))
    first = service.run(project, CancellationToken(), discard_progress)
    mask_dir = artifact_path(tmp_path, first.artifacts[ArtifactRole.SUBJECT_MASKS])
    write_mask(mask_dir / "000002.png", (8, 6))

    with pytest.raises(RepairableError, match="尺寸|代理"):
        service.run(project, CancellationToken(), discard_progress)


def test_segment_rejects_proxy_replacement_by_worker_before_publication(
    tmp_path: Path,
) -> None:
    project = ingest_succeeded(source_project(tmp_path), tmp_path)
    project.workflow.subject_prompt = SubjectPromptState(frame_index=0, x=1, y=1)

    with pytest.raises(RepairableError, match="代理|authority|变化|fingerprint"):
        SegmentWorkflowService(
            WorkflowPaths(tmp_path), ReplacingSegmenter(tmp_path)
        ).run(project, CancellationToken(), discard_progress)

    assert not any((tmp_path / "masks").glob("?" * 64))


class FakeSolver:
    identity = "fake-opencv-solver-1"

    def solve(
        self,
        frame_paths: list[Path] | tuple[Path, ...],
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> CameraSolution:
        poses = []
        for index, _path in enumerate(frame_paths, start=1):
            token.raise_if_cancelled()
            pose = np.eye(4)
            pose[0, 3] = index - 1
            poses.append(pose)
            emit(index, len(frame_paths), f"camera {index}")
        intrinsics = np.array([[3.0, 0.0, 2.0], [0.0, 3.0, 1.5], [0.0, 0.0, 1.0]])
        return CameraSolution(
            intrinsics,
            poses,
            CameraKind.SIX_DOF,
            0.9,
            {"backend": "fake"},
        )


class ReplacingSolver(FakeSolver):
    def solve(
        self,
        frame_paths: list[Path] | tuple[Path, ...],
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> CameraSolution:
        replace_rgb_with_same_size(frame_paths[0], 200)
        return super().solve(frame_paths, emit, token)


def test_custom_solver_requires_explicit_backend_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identity"):
        CameraSolveWorkflowService(WorkflowPaths(tmp_path), FakeSolver())


def test_camera_solver_publishes_serialized_solution(tmp_path: Path) -> None:
    project = ingest_succeeded(source_project(tmp_path), tmp_path)
    service = CameraSolveWorkflowService(
        WorkflowPaths(tmp_path), FakeSolver(), backend_identity="fake-opencv-1"
    )

    result = service.run(project, CancellationToken(), discard_progress)

    relative = result.artifacts[ArtifactRole.CAMERA_SOLUTION]
    assert relative == artifact_ref(
        project, ArtifactCategory.CAMERA, result.cache_key, "solution.json"
    )
    restored = read_camera_solution(artifact_path(tmp_path, relative))
    assert len(restored.camera_to_world) == 3
    np.testing.assert_allclose(restored.camera_to_world[2][:3, 3], [2, 0, 0])


def test_camera_cache_hit_revalidates_pose_count(tmp_path: Path) -> None:
    project = ingest_succeeded(source_project(tmp_path), tmp_path)
    service = CameraSolveWorkflowService(
        WorkflowPaths(tmp_path), FakeSolver(), backend_identity="fake-opencv-1"
    )
    first = service.run(project, CancellationToken(), discard_progress)
    path = artifact_path(tmp_path, first.artifacts[ArtifactRole.CAMERA_SOLUTION])
    restored = read_camera_solution(path)
    write_camera_solution(
        path,
        CameraSolution(
            restored.intrinsics,
            [restored.camera_to_world[0]],
            restored.kind,
            restored.confidence,
            restored.diagnostics,
        ),
    )

    with pytest.raises(RepairableError, match="帧数|轨迹"):
        service.run(project, CancellationToken(), discard_progress)


def test_camera_solver_rejects_proxy_replacement_before_publication(
    tmp_path: Path,
) -> None:
    project = ingest_succeeded(source_project(tmp_path), tmp_path)

    with pytest.raises(RepairableError, match="代理|authority|变化|fingerprint"):
        CameraSolveWorkflowService(
            WorkflowPaths(tmp_path),
            ReplacingSolver(),
            backend_identity="replacing-solver-v1",
        ).run(project, CancellationToken(), discard_progress)

    assert not any((tmp_path / "camera").glob("?" * 64))


def camera_succeeded(project: Project, root: Path, cache_key: str = key("c")) -> Project:
    project = ingest_succeeded(project, root)
    camera_dir = root / "camera" / cache_key
    camera_dir.mkdir(parents=True, exist_ok=True)
    write_camera_solution(camera_dir / "solution.json", FakeSolver().solve(
        [Path("1"), Path("2"), Path("3")], discard_progress, CancellationToken()
    ))
    project.stages[StageName.SOLVE_CAMERA] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=cache_key,
        artifacts={
            ArtifactRole.CAMERA_SOLUTION: artifact_ref(
                project, ArtifactCategory.CAMERA, cache_key, "solution.json"
            )
        },
    )
    return project


def authorize_mapping(project: Project) -> None:
    project.scene_ply = "source/scene.ply"
    project.stages[StageName.SEGMENT] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=key("b"),
    )
    project.workflow.source_perspective_calibration = SourcePerspectiveCalibration(
        source_asset_id="source/source.mp4",
        ingest_cache_key=key("a"),
        segment_cache_key=key("b"),
        anchor_frame_index=0,
        image_width=8,
        image_height=6,
        vertical_fov=55,
        horizon_line=(0.0, 1.0, -3.0),
        gravity_direction_camera=(0.0, -1.0, 0.0),
        revision=2,
    )
    identity = tuple(tuple(float(value) for value in row) for row in np.eye(4))
    project.workflow.local_ground_anchor = LocalGroundAnchor(
        scene_asset_id="source/scene.ply",
        p0_world=(0.0, 0.0, 0.0),
        p1_world=(1.0, 0.0, 0.0),
        p2_world=(0.0, 0.0, 1.0),
        plane_normal=(0.0, -1.0, 0.0),
        plane_offset=0.0,
        frozen_camera_to_world=identity,
        frozen_camera_fingerprint="camera-fingerprint",
        preview_artifact_id="preview-2",
        camera_revision=2,
        pick_buffer_revision=4,
        revision=3,
    )
    project.workflow.synthesis_placement = SynthesisPlacement(
        source_calibration_revision=2,
        ground_anchor_revision=3,
        mode=SynthesisConstraintMode.PERSPECTIVE,
        scene_azimuth=0.0,
        subject_to_scene_scale=1.0,
        composition_offset_local=(0.0, 0.0),
        anchor_camera_to_world=identity,
        intrinsics=((5.0, 0.0, 4.0), (0.0, 5.0, 3.0), (0.0, 0.0, 1.0)),
        solver_cache_key=key("f"),
        revision=4,
    )
    project.workflow.confirmed_synthesis_placement_revision = 4
    project.workflow.motion_scale = 0.5


def test_trajectory_mapper_requires_and_serializes_single_preview_authority(
    tmp_path: Path,
) -> None:
    project = camera_succeeded(source_project(tmp_path), tmp_path)
    authorize_mapping(project)

    result = TrajectoryMapWorkflowService(WorkflowPaths(tmp_path)).run(
        project, CancellationToken(), discard_progress
    )

    relative = result.artifacts[ArtifactRole.MAPPED_TRAJECTORY]
    assert relative == artifact_ref(
        project,
        ArtifactCategory.TRAJECTORIES,
        result.cache_key,
        "trajectory.json",
    )
    mapped = read_mapped_trajectory(artifact_path(tmp_path, relative))
    assert mapped.fov_y_degrees == 55
    assert len(mapped.camera_to_world) == 3
    assert np.linalg.norm(
        mapped.camera_to_world[1][:3, 3] - mapped.camera_to_world[0][:3, 3]
    ) == pytest.approx(0.5)


def test_trajectory_mapper_rejects_mismatched_placement_authority(tmp_path: Path) -> None:
    project = camera_succeeded(source_project(tmp_path), tmp_path)
    authorize_mapping(project)
    assert project.workflow.synthesis_placement is not None
    project.workflow.synthesis_placement.ground_anchor_revision = 2

    with pytest.raises(RepairableError, match="authority|授权|预览"):
        TrajectoryMapWorkflowService(WorkflowPaths(tmp_path)).run(
            project, CancellationToken(), discard_progress
        )


def test_trajectory_cache_hit_revalidates_fov_and_pose_count(tmp_path: Path) -> None:
    project = camera_succeeded(source_project(tmp_path), tmp_path)
    authorize_mapping(project)
    service = TrajectoryMapWorkflowService(WorkflowPaths(tmp_path))
    first = service.run(project, CancellationToken(), discard_progress)
    path = artifact_path(
        tmp_path, first.artifacts[ArtifactRole.MAPPED_TRAJECTORY]
    )
    cached = read_mapped_trajectory(path)
    write_mapped_trajectory(
        path,
        MappedTrajectory(77, cached.camera_to_world),
    )

    with pytest.raises(RepairableError, match="轨迹|FOV|fov"):
        service.run(project, CancellationToken(), discard_progress)


def test_trajectory_rejects_camera_replacement_during_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = camera_succeeded(source_project(tmp_path), tmp_path)
    authorize_mapping(project)
    solution_path = artifact_path(
        tmp_path,
        project.stages[StageName.SOLVE_CAMERA].artifacts[
            ArtifactRole.CAMERA_SOLUTION
        ],
    )
    original_map = workflow_services.map_trajectory

    def replacing_map(
        solution: CameraSolution,
        target_camera_to_world: np.ndarray,
        translation_scale: float,
        anchor_frame_index: int = 0,
    ) -> tuple[np.ndarray, ...]:
        replacement = solution_path.with_name("replacement.json")
        restored = read_camera_solution(solution_path)
        write_camera_solution(
            replacement,
            CameraSolution(
                restored.intrinsics,
                restored.camera_to_world,
                restored.kind,
                0.8,
                restored.diagnostics,
            ),
        )
        assert replacement.stat().st_size == solution_path.stat().st_size
        replacement.replace(solution_path)
        return original_map(
            solution,
            target_camera_to_world,
            translation_scale,
            anchor_frame_index,
        )

    monkeypatch.setattr(workflow_services, "map_trajectory", replacing_map)

    with pytest.raises(RepairableError, match="相机|authority|变化|摘要"):
        TrajectoryMapWorkflowService(WorkflowPaths(tmp_path)).run(
            project, CancellationToken(), discard_progress
        )

    assert not any((tmp_path / "trajectories").glob("?" * 64))


class MismatchedRendererWorker:
    def probe(
        self, *, token: CancellationToken | None = None
    ) -> RendererWorkerIdentity:
        del token
        return RendererWorkerIdentity(torch="2.9.0", gsplat="1.5.3", device="cuda")

    def render_sequence(
        self,
        request: RenderSequenceRequest,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> RenderSequence:
        token.raise_if_cancelled()
        request.output_dir.mkdir()
        frame = request.output_dir / "000001.png"
        write_rgb(frame, (request.width, request.height), 64)
        emit(1, 1, "rendered")
        return RenderSequence(
            frame_dir=request.output_dir,
            frame_paths=(frame,),
            source_frame_indices=(0,),
            width=request.width,
            height=request.height,
            implementation_version="gsplat-1.6.0",
        )


class CpuFakeRendererWorker:
    def __init__(self) -> None:
        self.requests: list[RenderSequenceRequest] = []

    def probe(
        self, *, token: CancellationToken | None = None
    ) -> RendererWorkerIdentity:
        if token is not None:
            token.raise_if_cancelled()
        return RendererWorkerIdentity(
            torch="cpu-fake",
            gsplat="cpu-fake-1",
            device="cpu",
        )

    def render_sequence(
        self,
        request: RenderSequenceRequest,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> RenderSequence:
        self.requests.append(request)
        request.output_dir.mkdir()
        trajectory = read_mapped_trajectory(request.camera_manifest)
        source_indices = tuple(
            range(0, len(trajectory.camera_to_world), request.preview_stride)
        )
        frames: list[Path] = []
        for current, _source_index in enumerate(source_indices, start=1):
            token.raise_if_cancelled()
            frame = request.output_dir / f"{current:06d}.png"
            write_rgb(frame, (request.width, request.height), 64)
            frames.append(frame)
            emit(
                current,
                len(source_indices),
                f"cpu fake render {current}/{len(source_indices)}",
            )
        return RenderSequence(
            frame_dir=request.output_dir,
            frame_paths=tuple(frames),
            source_frame_indices=source_indices,
            width=request.width,
            height=request.height,
            implementation_version="gsplat-cpu-fake-1",
        )


def test_renderer_service_orchestrates_cpu_adapter_and_registers_artifacts(
    tmp_path: Path,
) -> None:
    project = renderer_project(tmp_path)
    worker = CpuFakeRendererWorker()
    events: list[tuple[int, int, str]] = []
    service = RendererWorkflowService(WorkflowPaths(tmp_path), worker)

    result = service.run(
        project,
        "final",
        CancellationToken(),
        lambda current, total, message: events.append((current, total, message)),
    )

    assert len(worker.requests) == 1
    assert worker.requests[0].scene_path == tmp_path / "source" / "scene.ply"
    assert result.artifacts[ArtifactRole.RENDER_FRAMES] == artifact_ref(
        project, ArtifactCategory.RENDERS, result.cache_key
    )
    output = artifact_path(tmp_path, result.artifacts[ArtifactRole.RENDER_FRAMES])
    assert (output / "000001.png").is_file()
    assert events == [(1, 1, "cpu fake render 1/1")]


def renderer_project(root: Path) -> Project:
    project = source_project(root, frame_count=1)
    scene = root / "source" / "scene.ply"
    scene.write_bytes(b"gaussian-scene")
    project.scene_ply = "source/scene.ply"
    project.workflow.scene_summary = SceneSummary(
        filename="scene.ply",
        size=scene.stat().st_size,
        sha256="cf5fa7399fb8ae55ff379089bae93067f89c6da2437508f5e5311e5c9efc94af",
        gaussian_count=1,
        estimated_vram_mb=1,
    )
    mapped_key = key("b")
    trajectory = root / "trajectories" / mapped_key / "trajectory.json"
    trajectory.parent.mkdir(parents=True)
    write_mapped_trajectory(
        trajectory,
        MappedTrajectory(55.0, (np.eye(4, dtype=np.float64),)),
    )
    project.stages[StageName.MAP_TRAJECTORY] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=mapped_key,
        artifacts={
            ArtifactRole.MAPPED_TRAJECTORY: artifact_ref(
                project,
                ArtifactCategory.TRAJECTORIES,
                mapped_key,
                "trajectory.json",
            )
        },
    )
    return project


def test_renderer_rejects_terminal_identity_that_differs_from_probe(
    tmp_path: Path,
) -> None:
    project = renderer_project(tmp_path)
    service = RendererWorkflowService(
        WorkflowPaths(tmp_path), MismatchedRendererWorker()  # type: ignore[arg-type]
    )

    with pytest.raises(RepairableError, match="实现|identity|版本"):
        service.run(project, "final", CancellationToken(), discard_progress)

    assert not any((tmp_path / "renders").glob("?" * 64))


class RecordingExporter:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, Fraction, int, Path, tuple[int, int]]] = []

    def __call__(
        self,
        frames_dir: Path,
        source_video: Path,
        fps: Fraction,
        frame_count: int,
        output: Path,
        *,
        cancellation_check: Callable[[], None] | None = None,
    ) -> ExportResult:
        if cancellation_check is not None:
            cancellation_check()
        with Image.open(frames_dir / "000001.png") as first:
            size = first.size
        self.calls.append((frames_dir, source_video, fps, frame_count, output, size))
        output.write_bytes(b"verified-mp4")
        return ExportResult(
            output=output,
            fps=fps,
            frame_count=frame_count,
            duration=Fraction(frame_count, 1) / fps,
            has_audio=True,
        )


class RecordingProber:
    def __init__(self, frame_count: int, *, has_audio: bool = True) -> None:
        self.frame_count = frame_count
        self.has_audio = has_audio
        self.calls: list[Path] = []

    def __call__(
        self,
        path: Path,
        *,
        cancellation_check: Callable[[], None] | None = None,
    ) -> ExportResult:
        if cancellation_check is not None:
            cancellation_check()
        self.calls.append(path)
        fps = Fraction(24, 1)
        return ExportResult(
            output=path,
            fps=fps,
            frame_count=self.frame_count,
            duration=Fraction(self.frame_count, 1) / fps,
            has_audio=self.has_audio,
        )


def test_custom_exporters_require_explicit_identity(tmp_path: Path) -> None:
    exporter = RecordingExporter()

    with pytest.raises(ValueError, match="identity"):
        CompositeWorkflowService(WorkflowPaths(tmp_path), exporter=exporter)
    with pytest.raises(ValueError, match="identity"):
        ExportWorkflowService(WorkflowPaths(tmp_path), exporter=exporter)


def completed_render_project(root: Path) -> Project:
    project = ingest_succeeded(source_project(root), root)
    segment_key = key("b")
    render_key = key("e")
    masks = root / "masks" / segment_key
    renders = root / "renders" / render_key
    for index, value in enumerate((255, 128, 0), start=1):
        write_mask(masks / f"{index:06d}.png", (4, 3), value)
        write_rgb(renders / f"{index:06d}.png", (8, 6), 200)
    project.stages[StageName.SEGMENT] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=segment_key,
        artifacts={
            ArtifactRole.SUBJECT_MASKS: artifact_ref(
                project, ArtifactCategory.MASKS, segment_key
            )
        },
    )
    project.stages[StageName.RENDER] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=render_key,
        artifacts={
            ArtifactRole.RENDER_FRAMES: artifact_ref(
                project, ArtifactCategory.RENDERS, render_key
            )
        },
    )
    return project


def test_compositor_uses_source_dimensions_and_registers_preview(
    tmp_path: Path,
) -> None:
    project = completed_render_project(tmp_path)
    exporter = RecordingExporter()
    prober = RecordingProber(2)
    service = CompositeWorkflowService(
        WorkflowPaths(tmp_path),
        preview_frame_limit=2,
        exporter=exporter,
        exporter_identity="recording-exporter-v1",
        prober=prober,
    )

    result = service.run(project, CancellationToken(), discard_progress)

    assert result.artifacts[ArtifactRole.COMPOSITE_FRAMES] == artifact_ref(
        project, ArtifactCategory.COMPOSITES, result.cache_key
    )
    assert result.artifacts[ArtifactRole.COMPOSITE_PREVIEW] == artifact_ref(
        project,
        ArtifactCategory.PREVIEWS,
        result.cache_key,
        "composite-preview.mp4",
    )
    with Image.open(
        artifact_path(tmp_path, result.artifacts[ArtifactRole.COMPOSITE_FRAMES])
        / "000001.png"
    ) as image:
        assert image.mode == "RGB"
        assert image.size == (8, 6)
    assert len(exporter.calls) == 1
    assert exporter.calls[0][3] == 2
    assert exporter.calls[0][5] == (8, 6)
    assert len(prober.calls) == 1


def test_composite_preview_cache_hit_rejects_content_tampering(tmp_path: Path) -> None:
    project = completed_render_project(tmp_path)
    exporter = RecordingExporter()
    prober = RecordingProber(3)
    service = CompositeWorkflowService(
        WorkflowPaths(tmp_path),
        exporter=exporter,
        exporter_identity="recording-exporter-v1",
        prober=prober,
    )
    first = service.run(project, CancellationToken(), discard_progress)
    preview = artifact_path(
        tmp_path, first.artifacts[ArtifactRole.COMPOSITE_PREVIEW]
    )
    preview.write_bytes(b"tampered preview")

    with pytest.raises(RepairableError, match="hash|摘要|清单|manifest"):
        service.run(project, CancellationToken(), discard_progress)

    assert len(exporter.calls) == 1


def test_composite_preview_cache_hit_rejects_manifest_type_tampering(
    tmp_path: Path,
) -> None:
    project = completed_render_project(tmp_path)
    service = CompositeWorkflowService(
        WorkflowPaths(tmp_path),
        exporter=RecordingExporter(),
        exporter_identity="recording-exporter-v1",
        prober=RecordingProber(3),
    )
    first = service.run(project, CancellationToken(), discard_progress)
    preview = artifact_path(
        tmp_path, first.artifacts[ArtifactRole.COMPOSITE_PREVIEW]
    )
    manifest = preview.parent / "manifest.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["fps"] = 24
    manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RepairableError, match="manifest|字段|类型"):
        service.run(project, CancellationToken(), discard_progress)


def test_preview_exporter_metadata_is_validated_before_publication(
    tmp_path: Path,
) -> None:
    project = completed_render_project(tmp_path)

    class WrongMetadataExporter(RecordingExporter):
        def __call__(
            self,
            frames_dir: Path,
            source_video: Path,
            fps: Fraction,
            frame_count: int,
            output: Path,
            *,
            cancellation_check: Callable[[], None] | None = None,
        ) -> ExportResult:
            result = super().__call__(
                frames_dir,
                source_video,
                fps,
                frame_count,
                output,
                cancellation_check=cancellation_check,
            )
            return ExportResult(
                output=result.output,
                fps=result.fps,
                frame_count=result.frame_count + 1,
                duration=result.duration,
                has_audio=result.has_audio,
            )

    with pytest.raises(RepairableError, match="metadata|元数据|帧数"):
        CompositeWorkflowService(
            WorkflowPaths(tmp_path),
            exporter=WrongMetadataExporter(),
            exporter_identity="wrong-metadata-exporter-v1",
            prober=RecordingProber(3),
        ).run(project, CancellationToken(), discard_progress)

    assert not any((tmp_path / "previews").glob("?" * 64))


def test_compositor_rejects_nonconsecutive_render_inventory(tmp_path: Path) -> None:
    project = completed_render_project(tmp_path)
    render_path = artifact_path(
        tmp_path,
        project.stages[StageName.RENDER].artifacts[ArtifactRole.RENDER_FRAMES],
    )
    (render_path / "000002.png").rename(render_path / "000004.png")

    with pytest.raises(RepairableError, match="连续|inventory|帧"):
        CompositeWorkflowService(
            WorkflowPaths(tmp_path),
            exporter=RecordingExporter(),
            exporter_identity="recording-exporter-v1",
            prober=RecordingProber(3),
        ).run(project, CancellationToken(), discard_progress)


def test_compositor_rejects_masks_that_are_not_proxy_resolution(tmp_path: Path) -> None:
    project = completed_render_project(tmp_path)
    mask_path = artifact_path(
        tmp_path,
        project.stages[StageName.SEGMENT].artifacts[ArtifactRole.SUBJECT_MASKS],
    ) / "000002.png"
    write_mask(mask_path, (8, 6))

    with pytest.raises(RepairableError, match="尺寸|代理"):
        CompositeWorkflowService(
            WorkflowPaths(tmp_path),
            exporter=RecordingExporter(),
            exporter_identity="recording-exporter-v1",
            prober=RecordingProber(3),
        ).run(project, CancellationToken(), discard_progress)


def test_compositor_rejects_upstream_replacement_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = completed_render_project(tmp_path)
    source_directory = artifact_path(
        tmp_path,
        project.stages[StageName.INGEST].artifacts[ArtifactRole.SOURCE_FRAMES],
    )
    original_composite = workflow_services.composite_frame
    replaced = False

    def replacing_composite(
        foreground: np.ndarray,
        background: np.ndarray,
        alpha: np.ndarray,
        *,
        edge_px: int = 1,
    ) -> np.ndarray:
        nonlocal replaced
        if not replaced:
            replace_rgb_with_same_size(source_directory / "000002.png", 200)
            replaced = True
        return original_composite(
            foreground, background, alpha, edge_px=edge_px
        )

    monkeypatch.setattr(workflow_services, "composite_frame", replacing_composite)

    with pytest.raises(RepairableError, match="源|authority|变化|fingerprint"):
        CompositeWorkflowService(
            WorkflowPaths(tmp_path),
            exporter=RecordingExporter(),
            exporter_identity="recording-exporter-v1",
            prober=RecordingProber(3),
        ).run(project, CancellationToken(), discard_progress)

    assert not any((tmp_path / "composites").glob("?" * 64))


def test_export_registers_full_composite_video(tmp_path: Path) -> None:
    project = completed_render_project(tmp_path)
    preview_exporter = RecordingExporter()
    composite = CompositeWorkflowService(
        WorkflowPaths(tmp_path),
        exporter=preview_exporter,
        exporter_identity="recording-preview-exporter-v1",
        prober=RecordingProber(3),
    ).run(project, CancellationToken(), discard_progress)
    project.stages[StageName.COMPOSITE] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=composite.cache_key,
        artifacts=dict(composite.artifacts),
    )
    final_exporter = RecordingExporter()
    final_prober = RecordingProber(3)

    result = ExportWorkflowService(
        WorkflowPaths(tmp_path),
        exporter=final_exporter,
        exporter_identity="recording-final-exporter-v1",
        prober=final_prober,
    ).run(project, CancellationToken(), discard_progress)

    assert result.artifacts[ArtifactRole.EXPORT_VIDEO] == artifact_ref(
        project, ArtifactCategory.EXPORTS, result.cache_key, "final.mp4"
    )
    assert len(final_exporter.calls) == 1
    assert final_exporter.calls[0][3] == 3
    assert final_exporter.calls[0][5] == (8, 6)
    assert len(final_prober.calls) == 1


def test_final_export_cache_hit_rejects_content_tampering(tmp_path: Path) -> None:
    project = completed_render_project(tmp_path)
    composite = CompositeWorkflowService(
        WorkflowPaths(tmp_path),
        exporter=RecordingExporter(),
        exporter_identity="recording-preview-exporter-v1",
        prober=RecordingProber(3),
    ).run(project, CancellationToken(), discard_progress)
    project.stages[StageName.COMPOSITE] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=composite.cache_key,
        artifacts=dict(composite.artifacts),
    )
    exporter = RecordingExporter()
    service = ExportWorkflowService(
        WorkflowPaths(tmp_path),
        exporter=exporter,
        exporter_identity="recording-final-exporter-v1",
        prober=RecordingProber(3),
    )
    first = service.run(project, CancellationToken(), discard_progress)
    output = artifact_path(tmp_path, first.artifacts[ArtifactRole.EXPORT_VIDEO])
    output.write_bytes(b"tampered final")

    with pytest.raises(RepairableError, match="hash|摘要|清单|manifest"):
        service.run(project, CancellationToken(), discard_progress)

    assert len(exporter.calls) == 1


@pytest.mark.parametrize("mutate_source", [False, True])
def test_final_export_rejects_upstream_replacement_before_publication(
    tmp_path: Path, mutate_source: bool
) -> None:
    project = completed_render_project(tmp_path)
    composite = CompositeWorkflowService(
        WorkflowPaths(tmp_path),
        exporter=RecordingExporter(),
        exporter_identity="recording-preview-exporter-v1",
        prober=RecordingProber(3),
    ).run(project, CancellationToken(), discard_progress)
    project.stages[StageName.COMPOSITE] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=composite.cache_key,
        artifacts=dict(composite.artifacts),
    )

    class ReplacingExporter(RecordingExporter):
        def __call__(
            self,
            frames_dir: Path,
            source_video: Path,
            fps: Fraction,
            frame_count: int,
            output: Path,
            *,
            cancellation_check: Callable[[], None] | None = None,
        ) -> ExportResult:
            if mutate_source:
                replace_bytes_with_same_size(source_video, b"source-V1deo")
            else:
                replace_rgb_with_same_size(frames_dir / "000001.png", 200)
            return super().__call__(
                frames_dir,
                source_video,
                fps,
                frame_count,
                output,
                cancellation_check=cancellation_check,
            )

    with pytest.raises(
        RepairableError, match="合成|源视频|authority|变化|fingerprint|摘要"
    ):
        ExportWorkflowService(
            WorkflowPaths(tmp_path),
            exporter=ReplacingExporter(),
            exporter_identity="replacing-final-exporter-v1",
            prober=RecordingProber(3),
        ).run(project, CancellationToken(), discard_progress)

    assert not any((tmp_path / "exports").glob("?" * 64))


def test_services_check_cancellation_before_publication(tmp_path: Path) -> None:
    project = source_project(tmp_path)
    token = CancellationToken()

    class CancellingBackend(FakeMediaBackend):
        def extract_proxy_frames(
            self, source: Path, output_dir: Path, max_height: int
        ) -> list[Path]:
            paths = super().extract_proxy_frames(source, output_dir, max_height)
            token.cancel()
            return paths

    with pytest.raises(CancelledError, match="取消"):
        MediaIngestService(WorkflowPaths(tmp_path), CancellingBackend()).run(
            project, token, discard_progress
        )

    assert not any((tmp_path / "proxies").glob("?" * 64))


def test_source_frame_inventory_failure_is_reported_as_unsupported_material(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from gs_video.media.ingest import extract_source_frames

    def fake_run(command: list[str], **_kwargs: object) -> object:
        output = Path(command[-1]).parent
        write_rgb(output / "000001.png", (8, 6), 1)
        write_rgb(output / "000003.png", (8, 6), 2)
        return object()

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(UnsupportedMaterialError, match="连续"):
        extract_source_frames(tmp_path / "source.mp4", tmp_path / "frames")


def test_source_frame_extraction_uses_unscaled_video_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from gs_video.media.ingest import extract_source_frames

    captured: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> object:
        captured.extend(command)
        output = Path(command[-1]).parent
        write_rgb(output / "000001.png", (8, 6), 1)
        return object()

    monkeypatch.setattr("subprocess.run", fake_run)

    assert len(extract_source_frames(tmp_path / "source.mp4", tmp_path / "frames")) == 1
    assert captured[1:8] == [
        "-y",
        "-i",
        str(tmp_path / "source.mp4"),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
    ]
    assert "-vf" not in captured


class CheckpointCancellationToken(CancellationToken):
    def __init__(self, cancel_at: int) -> None:
        super().__init__()
        self.cancel_at = cancel_at
        self.checks = 0

    def raise_if_cancelled(self) -> None:
        self.checks += 1
        if self.checks == self.cancel_at:
            self.cancel()
        super().raise_if_cancelled()


def test_frame_inventory_cancels_before_opening_the_second_frame(
    tmp_path: Path,
) -> None:
    for index in range(1, 4):
        write_rgb(tmp_path / f"{index:06d}.png", (8, 6), index)
    token = CheckpointCancellationToken(cancel_at=6)

    with pytest.raises(CancelledError):
        _frame_inventory(
            tmp_path,
            suffix="png",
            image_format="PNG",
            mode="RGB",
            label="test",
            token=token,
        )

    assert token.checks == 6


def test_sha256_checks_cancellation_between_one_megabyte_blocks(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = tmp_path / "large.bin"
    payload.write_bytes(b"x" * (3 * 1024 * 1024))
    token = CheckpointCancellationToken(cancel_at=3)

    with pytest.raises(CancelledError):
        _sha256(payload, "large", token)

    assert token.checks == 3


def test_frame_inventory_rejects_path_replacement_between_decode_and_hash(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "000001.png"
    parked = tmp_path.parent / f"{tmp_path.name}-parked.png"
    write_rgb(frame, (8, 6), 10)

    class ReplacingToken(CancellationToken):
        def __init__(self) -> None:
            super().__init__()
            self.checks = 0

        def raise_if_cancelled(self) -> None:
            self.checks += 1
            if self.checks == 3:
                frame.replace(parked)
                write_rgb(frame, (8, 6), 200)
            super().raise_if_cancelled()

    with pytest.raises(RepairableError, match="身份|替换|变化|不可读"):
        _frame_inventory(
            tmp_path,
            suffix="png",
            image_format="PNG",
            mode="RGB",
            label="test",
            token=ReplacingToken(),
        )


def test_preview_size_never_enlarges_and_rejects_no_even_solution() -> None:
    assert _preview_size((1920, 1080), 540) == (960, 540)
    assert _preview_size((8, 6), 540) == (8, 6)
    with pytest.raises(RepairableError, match="偶数|放大"):
        _preview_size((1, 6), 540)
    with pytest.raises(RepairableError, match="偶数|放大"):
        _preview_size((2, 1000), 540)
