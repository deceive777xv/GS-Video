from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from gs_video.camera.opencv_solver import CameraSolution


Float64Array = npt.NDArray[np.float64]


def validate_rigid_transform(value: npt.ArrayLike, name: str) -> Float64Array:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 transform")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8) or not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-6
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} must be a finite rigid transform")
    return matrix.copy()


def map_relative_poses(
    source_camera_to_world: list[np.ndarray] | tuple[np.ndarray, ...],
    target_start: npt.ArrayLike,
    motion_scale: float,
    anchor_frame_index: int = 0,
) -> list[Float64Array]:
    if not source_camera_to_world:
        raise ValueError("source_camera_to_world must not be empty")
    if not isfinite(motion_scale) or motion_scale < 0:
        raise ValueError("motion_scale must be finite and non-negative")
    source = [
        validate_rigid_transform(pose, f"source_camera_to_world[{index}]")
        for index, pose in enumerate(source_camera_to_world)
    ]
    if not 0 <= anchor_frame_index < len(source):
        raise ValueError("anchor_frame_index must identify a source pose")
    target = validate_rigid_transform(target_start, "target_start")
    source_anchor_inverse = np.linalg.inv(source[anchor_frame_index])
    mapped: list[Float64Array] = []
    for source_pose in source:
        relative = source_anchor_inverse @ source_pose
        relative[:3, 3] *= motion_scale
        result = target @ relative
        mapped.append(validate_rigid_transform(result, "mapped pose"))
    mapped[anchor_frame_index] = target.copy()
    return mapped


def map_trajectory(
    solution: CameraSolution,
    target_start: npt.ArrayLike,
    motion_scale: float,
    anchor_frame_index: int = 0,
) -> list[Float64Array]:
    return map_relative_poses(
        solution.camera_to_world,
        target_start,
        motion_scale,
        anchor_frame_index,
    )
