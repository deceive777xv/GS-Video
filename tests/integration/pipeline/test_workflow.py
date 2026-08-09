import hashlib
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from gs_video.camera.classify import CameraKind
from gs_video.camera.opencv_solver import CameraSolution
from gs_video.domain.contracts import (
    MaskSequence,
    Prompt,
    SegmentationBackend,
    StageResult,
)
from gs_video.domain.errors import RepairableError
from gs_video.domain.models import (
    ArtifactCategory,
    ArtifactRef,
    ArtifactRole,
    CameraPose,
    FootPointState,
    PreviewState,
    Project,
    StageName,
    StageStatus,
    SubjectPromptState,
    VideoSummary,
)
from gs_video.media.export import ExportResult
from gs_video.pipeline.artifacts import ArtifactPublisher
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter
from gs_video.pipeline.services import (
    CameraSolveWorkflowService,
    CompositeWorkflowService,
    ExportWorkflowService,
    MediaIngestService,
    SegmentWorkflowService,
    TrajectoryMapWorkflowService,
    WorkflowPaths,
)
from gs_video.project.cache import cache_key
import gs_video.pipeline.workflow as workflow


EXPECTED_STAGE_ORDER = [
    "ingest",
    "segment",
    "solve_camera",
    "map_trajectory",
    "render",
    "composite",
    "export",
]


class RecordingService:
    def __init__(
        self,
        name: str,
        calls: list[str],
        *,
        failure: RepairableError | None = None,
        cancel: bool = False,
    ) -> None:
        self.name = name
        self.calls = calls
        self.failure = failure
        self.cancel = cancel

    def run(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        self.calls.append(self.name)
        if self.failure is not None:
            raise self.failure
        if self.cancel:
            token.cancel()
        result_key = hashlib.sha256(self.name.encode()).hexdigest()
        reference = ArtifactRef(
            project_id=project.project_id,
            category=ArtifactCategory.FRAMES,
            cache_key=result_key,
        )
        return StageResult((reference,), result_key)


class RecordingRenderer:
    def __init__(self, calls: list[str], namespaces: list[object]) -> None:
        self.calls = calls
        self.namespaces = namespaces

    def run(
        self,
        project: Project,
        namespace: object,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        self.calls.append("render")
        self.namespaces.append(namespace)
        result_key = hashlib.sha256(b"render").hexdigest()
        reference = ArtifactRef(
            project_id=project.project_id,
            category=ArtifactCategory.RENDERS,
            cache_key=result_key,
        )
        return StageResult((reference,), result_key)


def fake_services(
    calls: list[str],
    *,
    solve_failure: RepairableError | None = None,
    cancel_solver: bool = False,
    namespaces: list[object] | None = None,
) -> object:
    render_namespaces = [] if namespaces is None else namespaces
    return workflow.WorkflowServices(
        media_ingest=RecordingService("ingest", calls),
        segmenter=RecordingService("segment", calls),
        camera_solver=RecordingService(
            "solve_camera",
            calls,
            failure=solve_failure,
            cancel=cancel_solver,
        ),
        trajectory_mapper=RecordingService("map_trajectory", calls),
        renderer=RecordingRenderer(calls, render_namespaces),
        compositor=RecordingService("composite", calls),
        exporter=RecordingService("export", calls),
    )


def test_mvp_workflow_runs_missing_dependencies_in_expected_order() -> None:
    assert hasattr(workflow, "build_mvp_workflow")
    calls: list[str] = []
    project = Project(name="demo")
    runner = workflow.build_mvp_workflow(fake_services(calls), project)

    result = runner.run(StageName.EXPORT, CancellationToken())

    assert calls == EXPECTED_STAGE_ORDER
    assert result.status is StageStatus.SUCCEEDED
    assert all(project.stages[name].status is StageStatus.SUCCEEDED for name in StageName)


def test_mvp_workflow_skips_succeeded_stages_until_explicit_invalidation() -> None:
    calls: list[str] = []
    project = Project(name="demo")
    runner = workflow.build_mvp_workflow(fake_services(calls), project)
    runner.run(StageName.EXPORT, CancellationToken())

    runner.run(StageName.EXPORT, CancellationToken())

    assert calls == EXPECTED_STAGE_ORDER

    workflow.invalidate_for_change(project, workflow.ChangeKind.TARGET_CAMERA)
    runner.run(StageName.EXPORT, CancellationToken())

    assert calls == EXPECTED_STAGE_ORDER + [
        "map_trajectory",
        "render",
        "composite",
        "export",
    ]


def test_mvp_workflow_uses_final_render_cache_namespace() -> None:
    calls: list[str] = []
    namespaces: list[object] = []
    project = Project(name="demo")
    runner = workflow.build_mvp_workflow(fake_services(calls, namespaces=namespaces), project)

    runner.run(StageName.RENDER, CancellationToken())

    assert namespaces == [workflow.RenderCacheNamespace.FINAL]
    assert workflow.RenderCacheNamespace.FINAL != workflow.RenderCacheNamespace.PREVIEW


def test_render_stage_passes_preview_cache_namespace_to_renderer() -> None:
    calls: list[str] = []
    namespaces: list[object] = []
    stage = workflow.RenderStage(
        RecordingRenderer(calls, namespaces),
        workflow.RenderCacheNamespace.PREVIEW,
    )

    result = stage.execute(Project(name="preview"), CancellationToken(), lambda *_: None)

    assert result.cache_key == hashlib.sha256(b"render").hexdigest()
    assert calls == ["render"]
    assert namespaces == [workflow.RenderCacheNamespace.PREVIEW]


@pytest.mark.parametrize("cancel_solver", [False, True], ids=["failure", "cancellation"])
def test_dependency_terminal_state_stops_downstream_stages(cancel_solver: bool) -> None:
    calls: list[str] = []
    project = Project(name="demo")
    saved: list[Project] = []
    runner = workflow.build_mvp_workflow(
        fake_services(
            calls,
            solve_failure=None if cancel_solver else RepairableError("solver failed"),
            cancel_solver=cancel_solver,
        ),
        project,
        save=lambda value: saved.append(value.model_copy(deep=True)),
    )

    result = runner.run(StageName.EXPORT, CancellationToken())

    assert calls == ["ingest", "segment", "solve_camera"]
    assert project.stages[StageName.SOLVE_CAMERA].status is (
        StageStatus.CANCELLED if cancel_solver else StageStatus.FAILED
    )
    assert result.status is StageStatus.PENDING
    assert project.stages[StageName.MAP_TRAJECTORY].status is StageStatus.PENDING
    assert project.stages[StageName.RENDER].status is StageStatus.PENDING
    assert saved
    for snapshot in saved:
        for name in (
            StageName.MAP_TRAJECTORY,
            StageName.RENDER,
            StageName.COMPOSITE,
            StageName.EXPORT,
        ):
            if name in snapshot.stages:
                assert snapshot.stages[name].status is StageStatus.PENDING
    for name in (
        StageName.MAP_TRAJECTORY,
        StageName.RENDER,
        StageName.COMPOSITE,
        StageName.EXPORT,
    ):
        assert saved[-1].stages[name].status is StageStatus.PENDING


def _write_rgb(path: Path, size: tuple[int, int], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value, value, value)).save(path)


class _IntegrationMediaBackend:
    identity = "integration-media-v1"

    def extract_source_frames(self, source: Path, output_dir: Path) -> list[Path]:
        del source
        paths: list[Path] = []
        for index, value in enumerate((40, 80), start=1):
            path = output_dir / f"{index:06d}.png"
            _write_rgb(path, (8, 6), value)
            paths.append(path)
        return paths

    def extract_proxy_frames(
        self, source: Path, output_dir: Path, max_height: int
    ) -> list[Path]:
        del source, max_height
        paths: list[Path] = []
        for index, value in enumerate((40, 80), start=1):
            path = output_dir / f"{index:06d}.jpg"
            _write_rgb(path, (4, 3), value)
            paths.append(path)
        return paths


class _IntegrationSegmenter:
    backend = SegmentationBackend.EDGETAM
    worker_prefix = ("integration-worker",)

    def __init__(self, root: Path) -> None:
        self.model_config = root / "models" / "segment.yaml"
        self.checkpoint = root / "models" / "segment.pt"
        self.model_config.parent.mkdir(parents=True)
        self.model_config.write_bytes(b"config")
        self.checkpoint.write_bytes(b"checkpoint")

    def segment(
        self,
        frames: list[Path],
        prompt: Prompt,
        output_dir: Path,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> MaskSequence:
        assert prompt.frame_index == 0
        output_dir.mkdir()
        for index, frame in enumerate(frames, start=1):
            token.raise_if_cancelled()
            with Image.open(frame) as image:
                Image.new("L", image.size, 255).save(
                    output_dir / f"{index:06d}.png"
                )
            emit(index, len(frames), f"segment {index}")
        return MaskSequence(output_dir, len(frames))


class _IntegrationSolver:
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
            emit(index, len(frame_paths), f"solve {index}")
        return CameraSolution(
            np.array([[3.0, 0.0, 2.0], [0.0, 3.0, 1.5], [0.0, 0.0, 1.0]]),
            poses,
            CameraKind.SIX_DOF,
            0.9,
        )


class _IntegrationRenderer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.publisher = ArtifactPublisher(root)

    def run(
        self,
        project: Project,
        namespace: workflow.RenderCacheNamespace,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        mapped = project.stages[StageName.MAP_TRAJECTORY]
        assert mapped.cache_key is not None
        result_key = cache_key(
            StageName.RENDER.value,
            {"mapped_cache_key": mapped.cache_key},
            {"namespace": namespace.value},
            "integration-renderer-v1",
        )

        def build(staging: Path) -> None:
            for index in range(1, 3):
                token.raise_if_cancelled()
                _write_rgb(staging / f"{index:06d}.png", (8, 6), 200)
                emit(index, 2, f"render {index}")

        output = self.publisher.publish_tree("renders", result_key, build)
        assert output.relative_to(self.root) == Path("renders", result_key)
        reference = ArtifactRef(
            project_id=project.project_id,
            category=ArtifactCategory.RENDERS,
            cache_key=result_key,
        )
        return StageResult(
            (reference,),
            result_key,
            {ArtifactRole.RENDER_FRAMES: reference},
        )


class _IntegrationExporter:
    def __init__(self) -> None:
        self.frame_counts: list[int] = []

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
        assert source_video.is_file()
        assert len(tuple(frames_dir.glob("*.png"))) == frame_count
        self.frame_counts.append(frame_count)
        output.write_bytes(b"verified integration mp4")
        return ExportResult(
            output=output,
            fps=fps,
            frame_count=frame_count,
            duration=Fraction(frame_count, 1) / fps,
            has_audio=True,
        )


class _IntegrationProber:
    def __call__(
        self,
        path: Path,
        *,
        cancellation_check: Callable[[], None] | None = None,
    ) -> ExportResult:
        if cancellation_check is not None:
            cancellation_check()
        fps = Fraction(24, 1)
        return ExportResult(
            output=path,
            fps=fps,
            frame_count=2,
            duration=Fraction(2, 1) / fps,
            has_audio=True,
        )


def test_concrete_cpu_services_progress_through_export_and_persist_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "source.mp4"
    source.parent.mkdir(parents=True)
    payload = b"integration source"
    source.write_bytes(payload)
    project = Project(name="concrete")
    project.source_video = "source/source.mp4"
    project.workflow.source_summary = VideoSummary(
        filename="source.mp4",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        width=8,
        height=6,
        duration_seconds=10,
        fps="24/1",
        has_audio=True,
        frame_count=2,
    )
    project.workflow.subject_prompt = SubjectPromptState(frame_index=0, x=1, y=1)
    project.workflow.target_camera = CameraPose(
        target=(0, 0, 0),
        distance=4,
        yaw=0,
        pitch=0,
        fov_y_degrees=55,
        revision=1,
    )
    project.workflow.confirmed_camera_revision = 1
    project.workflow.confirmed_preview_artifact_id = "preview-1"
    project.workflow.preview = PreviewState(
        artifact_id="preview-1",
        artifact_size=10,
        artifact_sha256="f" * 64,
        generation=1,
        width=4,
        height=3,
        camera_revision=1,
        pick_buffer_revision=1,
    )
    project.workflow.foot_point = FootPointState(
        image=(1, 1),
        world=(0, 0, 0),
        preview_artifact_id="preview-1",
        camera_revision=1,
        pick_buffer_revision=1,
    )
    paths = WorkflowPaths(tmp_path)
    exporter = _IntegrationExporter()
    services = workflow.WorkflowServices(
        media_ingest=MediaIngestService(paths, _IntegrationMediaBackend()),
        segmenter=SegmentWorkflowService(paths, _IntegrationSegmenter(tmp_path)),
        camera_solver=CameraSolveWorkflowService(
            paths, _IntegrationSolver(), backend_identity="integration-solver-v1"
        ),
        trajectory_mapper=TrajectoryMapWorkflowService(paths),
        renderer=_IntegrationRenderer(tmp_path),
        compositor=CompositeWorkflowService(
            paths,
            exporter=exporter,
            exporter_identity="integration-preview-exporter-v1",
            prober=_IntegrationProber(),
        ),
        exporter=ExportWorkflowService(
            paths,
            exporter=exporter,
            exporter_identity="integration-final-exporter-v1",
            prober=_IntegrationProber(),
        ),
    )
    runner = workflow.build_mvp_workflow(services, project)

    result = runner.run(StageName.EXPORT, CancellationToken())

    assert result.status is StageStatus.SUCCEEDED
    assert all(project.stages[name].status is StageStatus.SUCCEEDED for name in StageName)
    ingest = project.stages[StageName.INGEST]
    assert ingest.cache_key is not None
    assert ingest.artifacts[ArtifactRole.SOURCE_FRAMES].relative_path() == (
        f"frames/{ingest.cache_key}"
    )
    assert ingest.artifacts[ArtifactRole.PROXY_FRAMES].relative_path() == (
        f"proxies/{ingest.cache_key}"
    )
    export_relative = project.stages[StageName.EXPORT].artifacts[
        ArtifactRole.EXPORT_VIDEO
    ]
    assert export_relative.category is ArtifactCategory.EXPORTS
    assert (tmp_path / export_relative.relative_path()).read_bytes() == (
        b"verified integration mp4"
    )
    assert exporter.frame_counts == [2, 2]
