from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from gs_video.domain.models import ArtifactRef, ArtifactRole, Project, StageName
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter

if TYPE_CHECKING:
    from gs_video.scene.camera import OrbitCamera
    from gs_video.scene.ply import GaussianScene


@dataclass(frozen=True)
class StageResult:
    output_paths: tuple[ArtifactRef, ...]
    cache_key: str
    artifacts: dict[ArtifactRole, ArtifactRef] = field(default_factory=dict)


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
    peak_vram_mb: int = 0


@dataclass(frozen=True)
class RenderSettings:
    width: int
    height: int = 540
    sh_degree: int = 3
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    preview_stride: int = 1

    def __post_init__(self) -> None:
        if type(self.width) is not int or type(self.height) is not int or (
            self.width <= 0 or self.height <= 0
        ):
            raise ValueError("render dimensions must be positive integers")
        if type(self.sh_degree) is not int or not 0 <= self.sh_degree <= 3:
            raise ValueError("sh_degree must be an integer between 0 and 3")
        if type(self.preview_stride) is not int or self.preview_stride < 1:
            raise ValueError("preview_stride must be an integer >= 1")
        if len(self.background) != 3 or not all(isfinite(value) for value in self.background):
            raise ValueError("background must contain three finite values")


@dataclass(frozen=True)
class RenderSequence:
    frame_dir: Path
    frame_paths: tuple[Path, ...]
    source_frame_indices: tuple[int, ...]
    width: int
    height: int
    implementation_version: str

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)


@dataclass(frozen=True)
class PickBuffer:
    rgb: npt.NDArray[np.uint8]
    expected_depth: npt.NDArray[np.float32]


class SceneRenderer(Protocol):
    def render(
        self,
        scene: GaussianScene,
        cameras: Sequence[OrbitCamera],
        output_dir: Path,
        settings: RenderSettings,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> RenderSequence: ...

    def render_pick(
        self, scene: GaussianScene, camera: OrbitCamera, width: int, height: int
    ) -> PickBuffer: ...


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
