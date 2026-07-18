from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from gs_video.camera.classify import CameraKind
from gs_video.camera.opencv_solver import CameraSolution
from gs_video.camera.serialization import (
    read_camera_solution,
    read_mapped_trajectory,
)
from gs_video.domain.contracts import MaskSequence, Prompt, SegmentationBackend
from gs_video.domain.errors import CancelledError, RepairableError, UnsupportedMaterialError
from gs_video.domain.models import (
    ArtifactRole,
    CameraPose,
    FootPointState,
    PreviewState,
    Project,
    StageName,
    StageState,
    StageStatus,
    SubjectPromptState,
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
    SegmentWorkflowService,
    TrajectoryMapWorkflowService,
    WorkflowPaths,
)


CacheKey = str
ProjectMutation = Callable[[Project], None]


def key(character: str) -> CacheKey:
    return character * 64


def write_rgb(path: Path, size: tuple[int, int], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value, value, value)).save(path)


def write_mask(path: Path, size: tuple[int, int], value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, value).save(path)


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

    assert result.artifacts[ArtifactRole.PROXY_FRAMES] == Path(
        f"proxies/{result.cache_key}"
    )
    assert result.artifacts[ArtifactRole.SOURCE_FRAMES] == Path(
        f"frames/{result.cache_key}"
    )
    with Image.open(tmp_path / result.artifacts[ArtifactRole.SOURCE_FRAMES] / "000001.png") as image:
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
            ArtifactRole.SOURCE_FRAMES: f"frames/{cache_key}",
            ArtifactRole.PROXY_FRAMES: f"proxies/{cache_key}",
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
    assert relative == Path(f"masks/{result.cache_key}")
    assert sorted(path.name for path in (tmp_path / relative).iterdir()) == [
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
    project.stages[StageName.INGEST].artifacts[ArtifactRole.PROXY_FRAMES] = (
        f"proxies/{key('b')}"
    )

    with pytest.raises(RepairableError, match="cache|缓存|artifact|产物"):
        SegmentWorkflowService(WorkflowPaths(tmp_path), FakeSegmenter(tmp_path)).run(
            project, CancellationToken(), discard_progress
        )


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


def test_camera_solver_publishes_serialized_solution(tmp_path: Path) -> None:
    project = ingest_succeeded(source_project(tmp_path), tmp_path)
    service = CameraSolveWorkflowService(
        WorkflowPaths(tmp_path), FakeSolver(), backend_identity="fake-opencv-1"
    )

    result = service.run(project, CancellationToken(), discard_progress)

    relative = result.artifacts[ArtifactRole.CAMERA_SOLUTION]
    assert relative == Path(f"camera/{result.cache_key}/solution.json")
    restored = read_camera_solution(tmp_path / relative)
    assert len(restored.camera_to_world) == 3
    np.testing.assert_allclose(restored.camera_to_world[2][:3, 3], [2, 0, 0])


def camera_succeeded(project: Project, root: Path, cache_key: str = key("c")) -> Project:
    project = ingest_succeeded(project, root)
    camera_dir = root / "camera" / cache_key
    camera_dir.mkdir(parents=True, exist_ok=True)
    from gs_video.camera.serialization import write_camera_solution

    write_camera_solution(camera_dir / "solution.json", FakeSolver().solve(
        [Path("1"), Path("2"), Path("3")], discard_progress, CancellationToken()
    ))
    project.stages[StageName.SOLVE_CAMERA] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=cache_key,
        artifacts={ArtifactRole.CAMERA_SOLUTION: f"camera/{cache_key}/solution.json"},
    )
    return project


def authorize_mapping(project: Project) -> None:
    project.workflow.target_camera = CameraPose(
        target=(0, 0, 0),
        distance=4,
        yaw=0,
        pitch=0,
        fov_y_degrees=55,
        revision=2,
    )
    project.workflow.confirmed_camera_revision = 2
    project.workflow.confirmed_preview_artifact_id = "preview-2"
    project.workflow.preview = PreviewState(
        artifact_id="preview-2",
        artifact_size=100,
        artifact_sha256="d" * 64,
        generation=1,
        width=4,
        height=3,
        camera_revision=2,
        pick_buffer_revision=4,
    )
    project.workflow.foot_point = FootPointState(
        image=(1, 1),
        world=(0.0, 0.0, 0.0),
        preview_artifact_id="preview-2",
        camera_revision=2,
        pick_buffer_revision=4,
    )
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
    assert relative == Path(f"trajectories/{result.cache_key}/trajectory.json")
    mapped = read_mapped_trajectory(tmp_path / relative)
    assert mapped.fov_y_degrees == 55
    assert len(mapped.camera_to_world) == 3
    assert np.linalg.norm(
        mapped.camera_to_world[1][:3, 3] - mapped.camera_to_world[0][:3, 3]
    ) == pytest.approx(0.5)


def test_trajectory_mapper_rejects_mismatched_pick_authority(tmp_path: Path) -> None:
    project = camera_succeeded(source_project(tmp_path), tmp_path)
    authorize_mapping(project)
    assert project.workflow.foot_point is not None
    project.workflow.foot_point.pick_buffer_revision = 3

    with pytest.raises(RepairableError, match="authority|授权|预览"):
        TrajectoryMapWorkflowService(WorkflowPaths(tmp_path)).run(
            project, CancellationToken(), discard_progress
        )


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
    ) -> ExportResult:
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
        artifacts={ArtifactRole.SUBJECT_MASKS: f"masks/{segment_key}"},
    )
    project.stages[StageName.RENDER] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=render_key,
        artifacts={ArtifactRole.RENDER_FRAMES: f"renders/{render_key}"},
    )
    return project


def test_compositor_uses_source_dimensions_and_registers_preview(
    tmp_path: Path,
) -> None:
    project = completed_render_project(tmp_path)
    exporter = RecordingExporter()
    service = CompositeWorkflowService(
        WorkflowPaths(tmp_path),
        preview_frame_limit=2,
        exporter=exporter,
    )

    result = service.run(project, CancellationToken(), discard_progress)

    assert result.artifacts[ArtifactRole.COMPOSITE_FRAMES] == Path(
        f"composites/{result.cache_key}"
    )
    assert result.artifacts[ArtifactRole.COMPOSITE_PREVIEW] == Path(
        f"previews/{result.cache_key}/composite-preview.mp4"
    )
    with Image.open(
        tmp_path / result.artifacts[ArtifactRole.COMPOSITE_FRAMES] / "000001.png"
    ) as image:
        assert image.mode == "RGB"
        assert image.size == (8, 6)
    assert len(exporter.calls) == 1
    assert exporter.calls[0][3] == 2
    assert exporter.calls[0][5] == (8, 6)


def test_compositor_rejects_nonconsecutive_render_inventory(tmp_path: Path) -> None:
    project = completed_render_project(tmp_path)
    render_path = tmp_path / project.stages[StageName.RENDER].artifacts[
        ArtifactRole.RENDER_FRAMES
    ]
    (render_path / "000002.png").rename(render_path / "000004.png")

    with pytest.raises(RepairableError, match="连续|inventory|帧"):
        CompositeWorkflowService(
            WorkflowPaths(tmp_path), exporter=RecordingExporter()
        ).run(project, CancellationToken(), discard_progress)


def test_compositor_rejects_masks_that_are_not_proxy_resolution(tmp_path: Path) -> None:
    project = completed_render_project(tmp_path)
    mask_path = tmp_path / project.stages[StageName.SEGMENT].artifacts[
        ArtifactRole.SUBJECT_MASKS
    ] / "000002.png"
    write_mask(mask_path, (8, 6))

    with pytest.raises(RepairableError, match="尺寸|代理"):
        CompositeWorkflowService(
            WorkflowPaths(tmp_path), exporter=RecordingExporter()
        ).run(project, CancellationToken(), discard_progress)


def test_export_registers_full_composite_video(tmp_path: Path) -> None:
    project = completed_render_project(tmp_path)
    preview_exporter = RecordingExporter()
    composite = CompositeWorkflowService(
        WorkflowPaths(tmp_path), exporter=preview_exporter
    ).run(project, CancellationToken(), discard_progress)
    project.stages[StageName.COMPOSITE] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=composite.cache_key,
        artifacts={role: str(path) for role, path in composite.artifacts.items()},
    )
    final_exporter = RecordingExporter()

    result = ExportWorkflowService(
        WorkflowPaths(tmp_path), exporter=final_exporter
    ).run(project, CancellationToken(), discard_progress)

    assert result.artifacts[ArtifactRole.EXPORT_VIDEO] == Path(
        f"exports/{result.cache_key}/final.mp4"
    )
    assert len(final_exporter.calls) == 1
    assert final_exporter.calls[0][3] == 3
    assert final_exporter.calls[0][5] == (8, 6)


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
