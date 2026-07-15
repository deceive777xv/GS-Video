from pathlib import Path

import pytest

from gs_video.domain.contracts import StageResult
from gs_video.domain.errors import RepairableError
from gs_video.domain.models import Project, StageName, StageStatus
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter
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
        return StageResult((Path("artifacts") / self.name,), f"{self.name}-key")


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
        return StageResult((Path("renders") / "final",), "render-key")


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

    assert result.cache_key == "render-key"
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
