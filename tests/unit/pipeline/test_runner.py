from collections.abc import Callable
from pathlib import Path

import pytest

from gs_video.domain.contracts import StageResult
from gs_video.domain.errors import GsVideoError, RepairableError, UnsupportedMaterialError
from gs_video.domain.models import (
    ArtifactRole,
    Project,
    StageName,
    StageState,
    StageStatus,
    SubjectPromptState,
)
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter
from gs_video.pipeline.runner import PipelineRunner
from gs_video.pipeline.workflow import invalidate_from
from gs_video.project.repository import ProjectRepository


class RecordingStage:
    name = StageName.RENDER

    def __init__(self, execute: Callable[[Project, CancellationToken], StageResult]) -> None:
        self._execute = execute
        self.calls = 0

    def execute(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        self.calls += 1
        emit(1, 2, "rendering")
        return self._execute(project, token)


def project_with_prior_output() -> Project:
    return Project(
        name="demo",
        stages={
            StageName.INGEST: StageState(
                status=StageStatus.SUCCEEDED,
                cache_key="ingest-key",
                output_paths=["source/meta.json"],
            )
        },
    )


def snapshots() -> tuple[list[Project], Callable[[Project], None]]:
    values: list[Project] = []

    def save(project: Project) -> None:
        values.append(project.model_copy(deep=True))

    return values, save


def test_runner_cancels_before_execute_and_saves_running_then_cancelled() -> None:
    project = project_with_prior_output()
    token = CancellationToken()
    token.cancel()
    stage = RecordingStage(
        lambda project, token: StageResult((Path("renders/frame.png"),), "render-key")
    )
    saved, save = snapshots()
    runner = PipelineRunner(project, {StageName.RENDER: stage}, save=save)

    state = runner.run(StageName.RENDER, token)

    assert stage.calls == 0
    assert [item.stages[StageName.RENDER].status for item in saved] == [
        StageStatus.RUNNING,
        StageStatus.CANCELLED,
    ]
    assert state.status is StageStatus.CANCELLED
    assert state.output_paths == []
    assert project.stages[StageName.INGEST].output_paths == ["source/meta.json"]


def test_runner_cancels_during_execute_before_registering_returned_outputs() -> None:
    project = project_with_prior_output()
    token = CancellationToken()

    def cancel_during_execute(project: Project, token: CancellationToken) -> StageResult:
        token.cancel()
        return StageResult((Path("renders/uncommitted.png"),), "uncommitted-key")

    stage = RecordingStage(cancel_during_execute)
    saved, save = snapshots()
    events: list[tuple[int, int, str]] = []
    runner = PipelineRunner(
        project,
        {StageName.RENDER: stage},
        save=save,
        emit=lambda current, total, message: events.append((current, total, message)),
    )

    state = runner.run(StageName.RENDER, token)

    assert state.status is StageStatus.CANCELLED
    assert state.output_paths == []
    assert state.cache_key is None
    assert events == [(1, 2, "rendering")]
    assert [item.stages[StageName.RENDER].status for item in saved] == [
        StageStatus.RUNNING,
        StageStatus.CANCELLED,
    ]


def test_runner_cancellation_clears_prior_same_stage_cache_key() -> None:
    project = project_with_prior_output()
    project.stages[StageName.RENDER] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key="old-render-key",
        output_paths=["renders/diagnostic.png"],
    )
    token = CancellationToken()
    token.cancel()
    saved, save = snapshots()
    runner = PipelineRunner(
        project,
        {
            StageName.RENDER: RecordingStage(
                lambda project, token: StageResult((Path("renders/new.png"),), "new-key")
            )
        },
        save=save,
    )

    state = runner.run(StageName.RENDER, token)

    assert saved[0].stages[StageName.RENDER].status is StageStatus.RUNNING
    assert saved[0].stages[StageName.RENDER].cache_key is None
    assert state.status is StageStatus.CANCELLED
    assert state.cache_key is None
    assert state.output_paths == ["renders/diagnostic.png"]


def test_runner_registers_outputs_only_after_stage_returns_successfully() -> None:
    project = project_with_prior_output()

    def execute(project: Project, token: CancellationToken) -> StageResult:
        running = project.stages[StageName.RENDER]
        assert running.status is StageStatus.RUNNING
        assert running.output_paths == []
        assert running.cache_key is None
        return StageResult((Path("renders/a.png"), Path("renders/b.png")), "render-key")

    saved, save = snapshots()
    runner = PipelineRunner(project, {StageName.RENDER: RecordingStage(execute)}, save=save)

    state = runner.run(StageName.RENDER, CancellationToken())

    assert [item.stages[StageName.RENDER].status for item in saved] == [
        StageStatus.RUNNING,
        StageStatus.SUCCEEDED,
    ]
    assert state.status is StageStatus.SUCCEEDED
    assert state.cache_key == "render-key"
    assert state.output_paths == [str(Path("renders/a.png")), str(Path("renders/b.png"))]
    assert state.error_code is None


def test_runner_persists_typed_stage_artifact_roots() -> None:
    project = project_with_prior_output()
    stage = RecordingStage(
        lambda project, token: StageResult(
            (Path("proxies"),),
            "ingest-key",
            artifacts={ArtifactRole.PROXY_FRAMES: Path("proxies")},
        )
    )
    runner = PipelineRunner(project, {StageName.RENDER: stage}, save=lambda project: None)

    state = runner.run(StageName.RENDER, CancellationToken())

    assert state.artifacts == {ArtifactRole.PROXY_FRAMES: "proxies"}


def test_stage_persistence_merges_into_latest_project_authority() -> None:
    stale_runner_project = project_with_prior_output()
    authoritative = stale_runner_project.model_copy(deep=True)

    def persist_stage(name: StageName, state: StageState) -> Project:
        authoritative.stages[name] = state.model_copy(deep=True)
        return authoritative.model_copy(deep=True)

    def patch_while_running(project: Project, token: CancellationToken) -> StageResult:
        del project, token
        authoritative.workflow.subject_prompt = SubjectPromptState(
            frame_index=4, x=100, y=120
        )
        return StageResult((Path("renders/final.png"),), "render-key")

    runner = PipelineRunner(
        stale_runner_project,
        {StageName.RENDER: RecordingStage(patch_while_running)},
        save=lambda project: None,
        persist_stage=persist_stage,
    )

    runner.run(StageName.RENDER, CancellationToken())

    assert authoritative.workflow.subject_prompt == SubjectPromptState(
        frame_index=4, x=100, y=120
    )
    assert authoritative.stages[StageName.RENDER].status is StageStatus.SUCCEEDED


def test_running_stage_cannot_publish_after_concurrent_invalidation(
    tmp_path: Path,
) -> None:
    repository = ProjectRepository(tmp_path / "project")
    repository.save(repository.create("race"))

    def invalidate_while_running(
        project: Project, token: CancellationToken
    ) -> StageResult:
        del project, token
        repository.update(lambda latest: invalidate_from(latest, StageName.RENDER))
        return StageResult((Path("renders/obsolete.png"),), "obsolete-key")

    runner = PipelineRunner(
        repository.load(),
        {StageName.RENDER: RecordingStage(invalidate_while_running)},
        save=repository.save,
        claim_stage=repository.claim_stage,
        compare_and_set_stage=repository.compare_and_set_stage,
    )

    state = runner.run(StageName.RENDER, CancellationToken())
    authoritative = repository.load().stages[StageName.RENDER]

    assert state == authoritative
    assert authoritative.status is StageStatus.STALE
    assert authoritative.input_generation == 1
    assert authoritative.run_id is None
    assert authoritative.cache_key is None
    assert authoritative.output_paths == []


def test_obsolete_dependency_does_not_run_downstream_stage(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "project")
    repository.save(repository.create("dependency-race"))

    def invalidate_dependency(
        project: Project, token: CancellationToken
    ) -> StageResult:
        del project, token
        repository.update(lambda latest: invalidate_from(latest, StageName.RENDER))
        return StageResult((Path("renders/obsolete.png"),), "obsolete-key")

    downstream = RecordingStage(
        lambda project, token: StageResult(
            (Path("exports/must-not-run.mp4"),), "export-key"
        )
    )
    runner = PipelineRunner(
        repository.load(),
        {
            StageName.RENDER: RecordingStage(invalidate_dependency),
            StageName.EXPORT: downstream,
        },
        save=repository.save,
        dependencies={StageName.EXPORT: (StageName.RENDER,)},
        claim_stage=repository.claim_stage,
        compare_and_set_stage=repository.compare_and_set_stage,
    )

    state = runner.run(StageName.EXPORT, CancellationToken())

    assert state.status is not StageStatus.SUCCEEDED
    assert downstream.calls == 0


def test_runner_claims_latest_invalidated_generation_instead_of_reusing_stale_copy(
    tmp_path: Path,
) -> None:
    repository = ProjectRepository(tmp_path / "project")
    project = repository.create("refresh-before-reuse")
    project.stages[StageName.RENDER] = StageState(
        status=StageStatus.SUCCEEDED, cache_key="old-key"
    )
    repository.save(project)
    stage = RecordingStage(
        lambda project, token: StageResult(
            (Path("renders/current.png"),), "current-key"
        )
    )
    runner = PipelineRunner(
        repository.load(),
        {StageName.RENDER: stage},
        save=repository.save,
        reuse_succeeded=True,
        claim_stage=repository.claim_stage,
        compare_and_set_stage=repository.compare_and_set_stage,
    )
    repository.update(lambda latest: invalidate_from(latest, StageName.RENDER))

    state = runner.run(StageName.RENDER, CancellationToken())

    assert stage.calls == 1
    assert state.status is StageStatus.SUCCEEDED
    assert state.input_generation == 1
    assert state.cache_key == "current-key"


def test_runner_atomically_reuses_current_succeeded_generation(
    tmp_path: Path,
) -> None:
    repository = ProjectRepository(tmp_path / "project")
    project = repository.create("reuse-current")
    project.stages[StageName.RENDER] = StageState(
        status=StageStatus.SUCCEEDED, cache_key="current-key"
    )
    repository.save(project)
    stage = RecordingStage(
        lambda project, token: StageResult((Path("unexpected"),), "unexpected")
    )
    runner = PipelineRunner(
        repository.load(),
        {StageName.RENDER: stage},
        save=repository.save,
        reuse_succeeded=True,
        claim_stage=repository.claim_stage,
        compare_and_set_stage=repository.compare_and_set_stage,
    )

    state = runner.run(StageName.RENDER, CancellationToken())

    assert stage.calls == 0
    assert state.status is StageStatus.SUCCEEDED
    assert state.cache_key == "current-key"


def test_repository_claim_does_not_replace_an_active_stage_owner(
    tmp_path: Path,
) -> None:
    repository = ProjectRepository(tmp_path / "project")
    repository.save(repository.create("single-owner"))

    first = repository.claim_stage(
        StageName.RENDER, reuse_succeeded=False, run_id="first-owner"
    )
    second = repository.claim_stage(
        StageName.RENDER, reuse_succeeded=False, run_id="second-owner"
    )

    assert first.claimed is True
    assert second.claimed is False
    assert second.project.stages[StageName.RENDER].run_id == "first-owner"


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (RepairableError("adjust input"), "repairable"),
        (UnsupportedMaterialError("unsupported"), "unsupported_material"),
        (GsVideoError("backend failed"), "system_error"),
    ],
)
def test_runner_maps_gs_video_errors_and_saves_terminal_state(
    error: GsVideoError, expected_code: str
) -> None:
    project = project_with_prior_output()

    def fail(project: Project, token: CancellationToken) -> StageResult:
        raise error

    saved, save = snapshots()
    runner = PipelineRunner(project, {StageName.RENDER: RecordingStage(fail)}, save=save)

    state = runner.run(StageName.RENDER, CancellationToken())

    assert [item.stages[StageName.RENDER].status for item in saved] == [
        StageStatus.RUNNING,
        StageStatus.FAILED,
    ]
    assert state.status is StageStatus.FAILED
    assert state.error_code == expected_code
    assert project.stages[StageName.INGEST].output_paths == ["source/meta.json"]


@pytest.mark.parametrize("error", [OSError("disk detail"), ValueError("vendor detail")])
def test_runner_commits_unexpected_exception_and_allows_retry(
    tmp_path: Path, error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    repository = ProjectRepository(tmp_path / "project")
    repository.save(repository.create("unexpected-retry"))
    attempts = 0

    def execute(project: Project, token: CancellationToken) -> StageResult:
        nonlocal attempts
        del project, token
        attempts += 1
        if attempts == 1:
            raise error
        return StageResult((Path("renders/final.png"),), "final-key")

    runner = PipelineRunner(
        repository.load(),
        {StageName.RENDER: RecordingStage(execute)},
        save=repository.save,
        claim_stage=repository.claim_stage,
        compare_and_set_stage=repository.compare_and_set_stage,
    )

    failed = runner.run(StageName.RENDER, CancellationToken())
    persisted_failure = repository.load().stages[StageName.RENDER]
    retried = runner.run(StageName.RENDER, CancellationToken())

    assert failed.status is StageStatus.FAILED
    assert failed.error_code == "unexpected_stage_failure"
    assert persisted_failure.status is StageStatus.FAILED
    assert persisted_failure.error_code == "unexpected_stage_failure"
    assert persisted_failure.run_id is None
    assert retried.status is StageStatus.SUCCEEDED
    assert attempts == 2
    assert str(error) in caplog.text


def test_runner_handled_failure_clears_prior_same_stage_cache_key() -> None:
    project = project_with_prior_output()
    project.stages[StageName.RENDER] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key="old-render-key",
        output_paths=["renders/diagnostic.png"],
    )

    def fail(project: Project, token: CancellationToken) -> StageResult:
        raise RepairableError("adjust input")

    saved, save = snapshots()
    runner = PipelineRunner(project, {StageName.RENDER: RecordingStage(fail)}, save=save)

    state = runner.run(StageName.RENDER, CancellationToken())

    assert saved[0].stages[StageName.RENDER].status is StageStatus.RUNNING
    assert saved[0].stages[StageName.RENDER].cache_key is None
    assert state.status is StageStatus.FAILED
    assert state.cache_key is None
    assert state.error_code == "repairable"
    assert state.output_paths == ["renders/diagnostic.png"]


def test_runner_clears_stale_error_on_retry_and_success() -> None:
    project = project_with_prior_output()
    project.stages[StageName.RENDER] = StageState(
        status=StageStatus.FAILED,
        error_code="repairable",
        output_paths=["renders/diagnostic.png"],
    )

    def execute(project: Project, token: CancellationToken) -> StageResult:
        assert project.stages[StageName.RENDER].error_code is None
        return StageResult((Path("renders/final.png"),), "final-key")

    saved, save = snapshots()
    runner = PipelineRunner(project, {StageName.RENDER: RecordingStage(execute)}, save=save)

    state = runner.run(StageName.RENDER, CancellationToken())

    assert saved[0].stages[StageName.RENDER].error_code is None
    assert state.status is StageStatus.SUCCEEDED
    assert state.error_code is None
    assert state.output_paths == [str(Path("renders/final.png"))]


def test_runner_rejects_unregistered_target_without_mutating_or_saving() -> None:
    project = project_with_prior_output()
    before = project.model_dump_json()
    saved, save = snapshots()
    runner = PipelineRunner(
        project,
        {
            StageName.RENDER: RecordingStage(
                lambda project, token: StageResult((Path("renders/frame.png"),), "render-key")
            )
        },
        save=save,
    )

    with pytest.raises(ValueError, match="unregistered target stage.*export"):
        runner.run(StageName.EXPORT, CancellationToken())

    assert saved == []
    assert project.model_dump_json() == before


@pytest.mark.parametrize(
    "dependencies",
    [
        {StageName.EXPORT: (StageName.RENDER,)},
        {StageName.RENDER: (StageName.EXPORT,)},
    ],
    ids=["unregistered-key", "unregistered-edge"],
)
def test_runner_rejects_dependency_reference_to_unregistered_stage_without_side_effects(
    dependencies: dict[StageName, tuple[StageName, ...]],
) -> None:
    project = project_with_prior_output()
    before = project.model_dump_json()
    saved, save = snapshots()

    with pytest.raises(ValueError, match="dependency graph.*unregistered stage.*export"):
        PipelineRunner(
            project,
            {
                StageName.RENDER: RecordingStage(
                    lambda project, token: StageResult(
                        (Path("renders/frame.png"),), "render-key"
                    )
                )
            },
            save=save,
            dependencies=dependencies,
        )

    assert saved == []
    assert project.model_dump_json() == before


def test_runner_rejects_dependency_cycle_without_mutating_or_saving() -> None:
    project = project_with_prior_output()
    before = project.model_dump_json()
    saved, save = snapshots()
    stages = {
        StageName.RENDER: RecordingStage(
            lambda project, token: StageResult((Path("renders/frame.png"),), "render-key")
        ),
        StageName.EXPORT: RecordingStage(
            lambda project, token: StageResult((Path("exports/video.mp4"),), "export-key")
        ),
    }

    with pytest.raises(ValueError, match="dependency graph contains a cycle"):
        PipelineRunner(
            project,
            stages,
            save=save,
            dependencies={
                StageName.RENDER: (StageName.EXPORT,),
                StageName.EXPORT: (StageName.RENDER,),
            },
        )

    assert saved == []
    assert project.model_dump_json() == before
