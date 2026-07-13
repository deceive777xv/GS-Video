from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from gs_video.domain.models import Project, StageName
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter


@dataclass(frozen=True)
class StageResult:
    output_paths: tuple[Path, ...]
    cache_key: str


class SegmentationBackend(StrEnum):
    EDGETAM = "edgetam"
    SAM2 = "sam2"


@dataclass(frozen=True)
class Prompt:
    frame_index: int
    x: int
    y: int


@dataclass(frozen=True)
class MaskSequence:
    mask_dir: Path
    frame_count: int


class ForegroundSegmenter(Protocol):
    def segment(
        self,
        frames: list[Path],
        prompt: Prompt,
        output_dir: Path,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> MaskSequence: ...


class Stage(Protocol):
    name: StageName

    def execute(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult: ...
