from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gs_video.domain.models import Project, StageName
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter


@dataclass(frozen=True)
class StageResult:
    output_paths: tuple[Path, ...]
    cache_key: str


class Stage(Protocol):
    name: StageName

    def execute(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult: ...
