from __future__ import annotations

import numpy as np
import pytest

from gs_video.camera.classify import CameraKind
from gs_video.camera.mapping import map_relative_poses, map_trajectory
from gs_video.camera.opencv_solver import CameraSolution


def pose(*, tx: float = 0.0, yaw_degrees: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(yaw_degrees)
    result = np.eye(4)
    result[:3, :3] = [
        [np.cos(yaw), 0.0, np.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)],
    ]
    result[0, 3] = tx
    return result


def test_mapping_keeps_target_start_and_scales_only_translation() -> None:
    source = [np.eye(4), pose(tx=1.0, yaw_degrees=10)]
    target_start = pose(tx=5.0, yaw_degrees=90)

    mapped = map_relative_poses(source, target_start, motion_scale=0.25)

    np.testing.assert_allclose(mapped[0], target_start)
    assert np.linalg.norm(mapped[1][:3, 3] - mapped[0][:3, 3]) == pytest.approx(0.25)
    np.testing.assert_allclose(mapped[1][:3, :3], target_start[:3, :3] @ source[1][:3, :3])


def test_relative_mapping_uses_inverse_source_zero_before_target_start() -> None:
    source = [pose(tx=2.0, yaw_degrees=30), pose(tx=3.0, yaw_degrees=40)]
    target_start = pose(tx=7.0, yaw_degrees=-20)

    mapped = map_relative_poses(source, target_start, motion_scale=1.0)

    expected = target_start @ np.linalg.inv(source[0]) @ source[1]
    np.testing.assert_allclose(mapped[0], target_start, atol=1e-12)
    np.testing.assert_allclose(mapped[1], expected, atol=1e-12)


def test_map_trajectory_accepts_solution_and_does_not_alias_inputs() -> None:
    source = [np.eye(4), pose(tx=2.0)]
    target_start = pose(tx=4.0)
    solution = CameraSolution(np.eye(3), source, CameraKind.SIX_DOF, 0.9)

    mapped = map_trajectory(solution, target_start, 0.5)
    mapped[0][0, 3] = 99

    assert target_start[0, 3] == 4
    assert solution.camera_to_world[0][0, 3] == 0


@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
def test_mapping_rejects_invalid_motion_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="motion_scale"):
        map_relative_poses([np.eye(4)], np.eye(4), scale)


def test_mapping_rejects_empty_or_malformed_poses() -> None:
    with pytest.raises(ValueError, match="empty"):
        map_relative_poses([], np.eye(4), 1.0)
    with pytest.raises(ValueError, match="4x4"):
        map_relative_poses([np.eye(3)], np.eye(4), 1.0)
    corrupt = np.eye(4)
    corrupt[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        map_relative_poses([corrupt], np.eye(4), 1.0)
    non_rigid = np.eye(4)
    non_rigid[0, 0] = 2.0
    with pytest.raises(ValueError, match="rigid"):
        map_relative_poses([non_rigid], np.eye(4), 1.0)


def test_camera_solution_rejects_malformed_inputs_and_copies_arrays() -> None:
    intrinsics = np.eye(3)
    poses = [np.eye(4)]
    solution = CameraSolution(intrinsics, poses, CameraKind.FIXED, 0.8)
    intrinsics[0, 0] = 9
    poses[0][0, 3] = 9

    assert solution.intrinsics[0, 0] == 1
    assert solution.camera_to_world[0][0, 3] == 0
    with pytest.raises(ValueError, match="intrinsics"):
        CameraSolution(np.eye(4), [np.eye(4)], CameraKind.FIXED, 0.8)
    with pytest.raises(ValueError, match="confidence"):
        CameraSolution(np.eye(3), [np.eye(4)], CameraKind.FIXED, float("nan"))
    with pytest.raises(ValueError, match="empty"):
        CameraSolution(np.eye(3), [], CameraKind.FIXED, 0.8)


def test_camera_solution_rejects_non_pinhole_intrinsics() -> None:
    negative_focal = np.eye(3)
    negative_focal[0, 0] = -1
    malformed_last_row = np.eye(3)
    malformed_last_row[2] = [1, 0, 1]

    for intrinsics in (negative_focal, malformed_last_row):
        with pytest.raises(ValueError, match="intrinsics"):
            CameraSolution(intrinsics, [np.eye(4)], CameraKind.FIXED, 0.8)


def test_camera_solution_does_not_alias_nested_diagnostics() -> None:
    diagnostics = {"pair": {"tracked_features": 100}}
    solution = CameraSolution(np.eye(3), [np.eye(4)], CameraKind.FIXED, 0.8, diagnostics)

    diagnostics["pair"]["tracked_features"] = 0

    assert solution.diagnostics["pair"]["tracked_features"] == 100
