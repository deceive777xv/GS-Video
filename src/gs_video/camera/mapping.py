from __future__ import annotations

from math import cos, isfinite, radians, sin
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from gs_video.camera.solution import CameraSolution


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


def _unit_tangent(
    value: npt.ArrayLike,
    normal: Float64Array,
    name: str,
) -> Float64Array:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    tangent = vector - float(np.dot(vector, normal)) * normal
    length = float(np.linalg.norm(tangent))
    if length <= 1e-9:
        raise ValueError(f"{name} must define a direction along the ground")
    return tangent / length


def map_ground_aligned_trajectory(
    solution: CameraSolution,
    *,
    target_p0: npt.ArrayLike,
    target_p1: npt.ArrayLike,
    target_normal: npt.ArrayLike,
    gs_scale: float,
    scene_azimuth_degrees: float = 0.0,
) -> list[Float64Array]:
    """Map a ViPE trajectory by aligning its source ground to target GS ground.

    The source anchor camera's orthogonal projection onto the recovered source
    ground maps to ``target_p0``. Camera height and all translation use the one
    ``gs_scale`` similarity scale; foreground pixels are not part of this mapping.
    """

    ground = solution.source_ground
    if ground is None:
        raise ValueError("camera solution requires an audited source ground")
    if not isfinite(gs_scale) or gs_scale <= 0:
        raise ValueError("gs_scale must be finite and positive")
    if not isfinite(scene_azimuth_degrees):
        raise ValueError("scene_azimuth_degrees must be finite")
    anchor_index = ground.anchor_frame_index
    if not 0 <= anchor_index < len(solution.camera_to_world):
        raise ValueError("source ground anchor frame is outside the trajectory")

    source_normal = np.asarray(ground.normal, dtype=np.float64)
    source_normal /= np.linalg.norm(source_normal)
    target_normal_array = np.asarray(target_normal, dtype=np.float64)
    if (
        target_normal_array.shape != (3,)
        or not np.all(np.isfinite(target_normal_array))
        or float(np.linalg.norm(target_normal_array)) <= 1e-9
    ):
        raise ValueError("target_normal must contain one finite nonzero direction")
    target_normal_array /= np.linalg.norm(target_normal_array)
    target_origin = np.asarray(target_p0, dtype=np.float64)
    target_direction_point = np.asarray(target_p1, dtype=np.float64)
    if (
        target_origin.shape != (3,)
        or target_direction_point.shape != (3,)
        or not np.all(np.isfinite(target_origin))
        or not np.all(np.isfinite(target_direction_point))
    ):
        raise ValueError("target ground points must contain three finite values")

    source_anchor = validate_rigid_transform(
        solution.camera_to_world[anchor_index], "source anchor camera"
    )
    source_origin = source_anchor[:3, 3] - (
        float(np.dot(source_normal, source_anchor[:3, 3])) + ground.offset
    ) * source_normal
    try:
        source_u = _unit_tangent(
            source_anchor[:3, 2], source_normal, "source camera forward"
        )
    except ValueError:
        source_u = _unit_tangent(
            source_anchor[:3, 0], source_normal, "source camera right"
        )
    source_v = np.cross(source_normal, source_u)
    source_v /= np.linalg.norm(source_v)

    target_u = _unit_tangent(
        target_direction_point - target_origin,
        target_normal_array,
        "target P0-to-P1",
    )
    target_v = np.cross(target_normal_array, target_u)
    target_v /= np.linalg.norm(target_v)
    angle = radians(scene_azimuth_degrees)
    azimuth_u = cos(angle) * target_u + sin(angle) * target_v
    azimuth_v = -sin(angle) * target_u + cos(angle) * target_v
    source_basis = np.column_stack((source_u, source_v, source_normal))
    target_basis = np.column_stack((azimuth_u, azimuth_v, target_normal_array))
    rotation = target_basis @ source_basis.T
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("ground alignment did not produce a rigid rotation")

    mapped: list[Float64Array] = []
    for index, pose_value in enumerate(solution.camera_to_world):
        pose = validate_rigid_transform(pose_value, f"camera_to_world[{index}]")
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rotation @ pose[:3, :3]
        result[:3, 3] = target_origin + gs_scale * rotation @ (
            pose[:3, 3] - source_origin
        )
        mapped.append(validate_rigid_transform(result, f"mapped camera_to_world[{index}]"))
    return mapped
