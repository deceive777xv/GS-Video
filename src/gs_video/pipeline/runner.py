from collections.abc import Callable, Mapping

from gs_video.domain.contracts import Stage
from gs_video.domain.errors import CancelledError, GsVideoError
from gs_video.domain.models import Project, StageName, StageState, StageStatus
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter, discard_progress


SaveProject = Callable[[Project], None]


class PipelineRunner:
    def __init__(
        self,
        project: Project,
        stages: Mapping[StageName, Stage],
        save: SaveProject,
        emit: ProgressEmitter = discard_progress,
    ) -> None:
        self.project = project
        self.stages = stages
        self.save = save
        self.emit = emit

    def run(self, name: StageName, token: CancellationToken) -> StageState:
        state = self.project.stages.setdefault(name, StageState())
        state.status = StageStatus.RUNNING
        state.cache_key = None
        state.error_code = None
        self.save(self.project)

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

        self.save(self.project)
        return state
