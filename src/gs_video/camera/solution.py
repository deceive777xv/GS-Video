from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from gs_video.camera.classify import CameraKind
from gs_video.camera.mapping import validate_rigid_transform


Float64Array = npt.NDArray[np.float64]


def _intrinsics(value: npt.ArrayLike, name: str) -> Float64Array:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-12)
    ):
        raise ValueError(f"{name} must be a valid finite pinhole matrix")
    result = matrix.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SourceGroundEstimate:
    normal: tuple[float, float, float]
    offset: float
    anchor_frame_index: int
    confidence: float
    support_ratio: float
    rms_residual: float

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=np.float64)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("source ground normal must contain three finite values")
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6):
            raise ValueError("source ground normal must be unit length")
        if not isfinite(self.offset):
            raise ValueError("source ground offset must be finite")
        if type(self.anchor_frame_index) is not int or self.anchor_frame_index < 0:
            raise ValueError("source ground anchor frame must be nonnegative")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("source ground confidence must lie in [0, 1]")
        if not isfinite(self.support_ratio) or not 0 <= self.support_ratio <= 1:
            raise ValueError("source ground support ratio must lie in [0, 1]")
        if not isfinite(self.rms_residual) or self.rms_residual < 0:
            raise ValueError("source ground residual must be finite and nonnegative")


@dataclass(frozen=True)
class CameraSolution:
    """Authoritative per-frame camera/depth solve produced by ViPE.

    ``intrinsics`` remains the first-frame calibration for narrow compatibility at
    renderer boundaries. ``frame_intrinsics`` is the production authority and must
    contain one matrix per pose when supplied by the automatic solver.
    """

    intrinsics: Float64Array
    camera_to_world: tuple[Float64Array, ...] | list[Float64Array]
    kind: CameraKind
    confidence: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    frame_intrinsics: tuple[Float64Array, ...] | list[Float64Array] | None = None
    source_ground: SourceGroundEstimate | None = None

    def __post_init__(self) -> None:
        anchor_intrinsics = _intrinsics(self.intrinsics, "intrinsics")
        if not self.camera_to_world:
            raise ValueError("camera_to_world must not be empty")
        poses = tuple(
            validate_rigid_transform(pose, f"camera_to_world[{index}]")
            for index, pose in enumerate(self.camera_to_world)
        )
        for pose in poses:
            pose.setflags(write=False)
        if self.frame_intrinsics is None:
            calibrations = tuple(anchor_intrinsics.copy() for _ in poses)
        else:
            if len(self.frame_intrinsics) != len(poses):
                raise ValueError("frame_intrinsics must contain one matrix per camera pose")
            calibrations = tuple(
                _intrinsics(value, f"frame_intrinsics[{index}]")
                for index, value in enumerate(self.frame_intrinsics)
            )
        for calibration in calibrations:
            calibration.setflags(write=False)
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and between zero and one")
        try:
            kind = CameraKind(self.kind)
        except ValueError as exc:
            raise ValueError("kind must be a valid CameraKind") from exc
        if self.source_ground is not None and self.source_ground.anchor_frame_index >= len(poses):
            raise ValueError("source ground anchor frame is outside the camera trajectory")
        object.__setattr__(self, "intrinsics", anchor_intrinsics)
        object.__setattr__(self, "camera_to_world", poses)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "frame_intrinsics", calibrations)
        object.__setattr__(
            self, "diagnostics", MappingProxyType(deepcopy(dict(self.diagnostics)))
        )
