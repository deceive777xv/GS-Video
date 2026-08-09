from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging
from uuid import uuid4

from gs_video.domain.contracts import Stage
from gs_video.domain.errors import CancelledError, GsVideoError
from gs_video.domain.models import (
    Project,
    ArtifactRef,
    StageClaimResult,
    StageName,
    StageState,
    StageStatus,
    StageWriteGuard,
    StageWriteResult,
)
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter, discard_progress


logger = logging.getLogger(__name__)


SaveProject = Callable[[Project], None]
PersistStage = Callable[[StageName, StageState], Project]
CompareAndSetStage = Callable[
    [StageName, StageState, StageWriteGuard], StageWriteResult
]
ClaimStage = Callable[..., StageClaimResult]


@dataclass(frozen=True)
class PipelineOutcome:
    requested_stage: StageName
    terminal_stage: StageName
    state: StageState
    error_category: str | None = None
    retryable: bool = False


def _validate_artifact_ref(
    reference: ArtifactRef, *, project_id: str, cache_key: str
) -> ArtifactRef:
    if reference.project_id != project_id:
        raise ValueError("stage artifact belongs to another project")
    if reference.cache_key != cache_key:
        raise ValueError("stage artifact cache key does not match the stage result")
    return reference


def _validated_dependencies(
    stages: Mapping[StageName, Stage],
    dependencies: Mapping[StageName, tuple[StageName, ...]] | None,
) -> dict[StageName, tuple[StageName, ...]]:
    graph = {} if dependencies is None else dict(dependencies)
    registered = set(stages)

    for name, required in graph.items():
        for referenced in (name, *required):
            if referenced not in registered:
                raise ValueError(
                    f"dependency graph references unregistered stage: {referenced.value}"
                )

    visiting: list[StageName] = []
    visited: set[StageName] = set()

    def visit(name: StageName) -> None:
        if name in visiting:
            start = visiting.index(name)
            cycle = (*visiting[start:], name)
            description = " -> ".join(stage.value for stage in cycle)
            raise ValueError(f"dependency graph contains a cycle: {description}")
        if name in visited:
            return
        visiting.append(name)
        for dependency in graph.get(name, ()):
            visit(dependency)
        visiting.pop()
        visited.add(name)

    for name in stages:
        visit(name)
    return graph


class PipelineRunner:
    def __init__(
        self,
        project: Project,
        stages: Mapping[StageName, Stage],
        save: SaveProject,
        emit: ProgressEmitter = discard_progress,
        dependencies: Mapping[StageName, tuple[StageName, ...]] | None = None,
        reuse_succeeded: bool = False,
        persist_stage: PersistStage | None = None,
        compare_and_set_stage: CompareAndSetStage | None = None,
        claim_stage: ClaimStage | None = None,
    ) -> None:
        validated_dependencies = _validated_dependencies(stages, dependencies)
        self.project = project
        self.stages = dict(stages)
        self.save = save
        self.emit = emit
        self.dependencies = validated_dependencies
        self.reuse_succeeded = reuse_succeeded
        self.persist_stage = persist_stage
        self.compare_and_set_stage = compare_and_set_stage
        self.claim_stage = claim_stage

    def supports(self, name: StageName) -> bool:
        return name in self.stages

    def _persist(self, name: StageName, state: StageState) -> StageState:
        if self.persist_stage is None:
            self.save(self.project)
            return state
        self.project = self.persist_stage(name, state)
        return self.project.stages[name]

    def _compare_and_set(
        self,
        name: StageName,
        state: StageState,
        guard: StageWriteGuard,
    ) -> tuple[StageState, bool]:
        if self.compare_and_set_stage is None:
            self.project.stages[name] = state
            return self._persist(name, state), True
        result = self.compare_and_set_stage(name, state, guard)
        self.project = result.project
        return self.project.stages.get(name, StageState()), result.applied

    def run(
        self,
        name: StageName,
        token: CancellationToken,
        emit: ProgressEmitter | None = None,
    ) -> StageState:
        self.run_outcome(name, token, emit)
        return self.project.stages.get(name, StageState())

    def run_outcome(
        self,
        name: StageName,
        token: CancellationToken,
        emit: ProgressEmitter | None = None,
    ) -> PipelineOutcome:
        run_emit = self.emit if emit is None else emit
        if name not in self.stages:
            raise ValueError(f"unregistered target stage: {name.value}")
        state = self.project.stages.setdefault(name, StageState())
        if (
            self.claim_stage is None
            and self.reuse_succeeded
            and state.status is StageStatus.SUCCEEDED
        ):
            return PipelineOutcome(name, name, state)

        for dependency in self.dependencies.get(name, ()):
            dependency_outcome = self.run_outcome(dependency, token, run_emit)
            if dependency_outcome.state.status is not StageStatus.SUCCEEDED:
                return PipelineOutcome(
                    requested_stage=name,
                    terminal_stage=dependency_outcome.terminal_stage,
                    state=dependency_outcome.state,
                    error_category=dependency_outcome.error_category,
                    retryable=dependency_outcome.retryable,
                )

        run_id = uuid4().hex
        if self.claim_stage is not None:
            claimed = self.claim_stage(
                name, reuse_succeeded=self.reuse_succeeded, run_id=run_id
            )
            self.project = claimed.project
            state = self.project.stages.get(name, StageState())
            started = claimed.claimed
            running = state
        else:
            prior = state.model_copy(deep=True)
            running = prior.model_copy(deep=True)
            running.status = StageStatus.RUNNING
            running.cache_key = None
            running.error_code = None
            running.run_id = run_id
            state, started = self._compare_and_set(
                name,
                running,
                StageWriteGuard(
                    input_generation=prior.input_generation,
                    status=prior.status,
                    run_id=prior.run_id,
                ),
            )
        if not started:
            return PipelineOutcome(name, name, state)

        error_category: str | None = None
        retryable = False
        try:
            token.raise_if_cancelled()
            result = self.stages[name].execute(self.project, token, run_emit)
            token.raise_if_cancelled()
            terminal = state.model_copy(deep=True)
            terminal.status = StageStatus.SUCCEEDED
            terminal.cache_key = result.cache_key
            terminal.output_paths = [
                _validate_artifact_ref(
                    reference,
                    project_id=self.project.project_id,
                    cache_key=result.cache_key,
                )
                for reference in result.output_paths
            ]
            terminal.artifacts = {
                role: _validate_artifact_ref(
                    reference,
                    project_id=self.project.project_id,
                    cache_key=result.cache_key,
                )
                for role, reference in result.artifacts.items()
            }
        except CancelledError:
            terminal = state.model_copy(deep=True)
            terminal.status = StageStatus.CANCELLED
        except GsVideoError as error:
            terminal = state.model_copy(deep=True)
            terminal.status = StageStatus.FAILED
            terminal.error_code = error.code
            error_category = error.category
            retryable = error.retryable
        except Exception:
            logger.exception("Unexpected failure in pipeline stage %s", name.value)
            terminal = state.model_copy(deep=True)
            terminal.status = StageStatus.FAILED
            terminal.error_code = "unexpected_stage_failure"
            error_category = "system"
            retryable = False

        terminal.run_id = None
        state, _applied = self._compare_and_set(
            name,
            terminal,
            StageWriteGuard(
                input_generation=running.input_generation,
                status=StageStatus.RUNNING,
                run_id=running.run_id,
            ),
        )
        return PipelineOutcome(
            requested_stage=name,
            terminal_stage=name,
            state=state,
            error_category=error_category,
            retryable=retryable,
        )
