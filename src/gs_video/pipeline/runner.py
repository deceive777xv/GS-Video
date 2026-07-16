from collections.abc import Callable, Mapping

from gs_video.domain.contracts import Stage
from gs_video.domain.errors import CancelledError, GsVideoError
from gs_video.domain.models import Project, StageName, StageState, StageStatus
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter, discard_progress


SaveProject = Callable[[Project], None]
PersistStage = Callable[[StageName, StageState], Project]


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
    ) -> None:
        validated_dependencies = _validated_dependencies(stages, dependencies)
        self.project = project
        self.stages = dict(stages)
        self.save = save
        self.emit = emit
        self.dependencies = validated_dependencies
        self.reuse_succeeded = reuse_succeeded
        self.persist_stage = persist_stage

    def _persist(self, name: StageName, state: StageState) -> StageState:
        if self.persist_stage is None:
            self.save(self.project)
            return state
        self.project = self.persist_stage(name, state)
        return self.project.stages[name]

    def run(self, name: StageName, token: CancellationToken) -> StageState:
        if name not in self.stages:
            raise ValueError(f"unregistered target stage: {name.value}")
        state = self.project.stages.setdefault(name, StageState())
        if self.reuse_succeeded and state.status is StageStatus.SUCCEEDED:
            return state

        for dependency in self.dependencies.get(name, ()):
            dependency_state = self.run(dependency, token)
            if dependency_state.status is not StageStatus.SUCCEEDED:
                return state

        state.status = StageStatus.RUNNING
        state.cache_key = None
        state.error_code = None
        state = self._persist(name, state)

        try:
            token.raise_if_cancelled()
            result = self.stages[name].execute(self.project, token, self.emit)
            token.raise_if_cancelled()
            state.status = StageStatus.SUCCEEDED
            state.cache_key = result.cache_key
            state.output_paths = [str(path) for path in result.output_paths]
        except CancelledError:
            state.status = StageStatus.CANCELLED
        except GsVideoError as error:
            state.status = StageStatus.FAILED
            state.error_code = error.code

        state = self._persist(name, state)
        return state
