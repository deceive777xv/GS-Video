from __future__ import annotations

import numpy as np

from gs_video.camera.opencv_solver import _compose_world_to_camera


def _increment(axis: np.ndarray, degrees: float, translation: tuple[float, float, float]) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(degrees)
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def test_non_commuting_world_to_camera_increments_are_left_composed() -> None:
    delta1 = _increment(np.array([1.0, 0.0, 0.0]), 17, (1.0, 2.0, 3.0))
    delta2 = _increment(np.array([0.0, 1.0, 0.0]), -29, (-4.0, 0.5, 2.0))

    after_first = _compose_world_to_camera(np.eye(4), delta1)
    after_second = _compose_world_to_camera(after_first, delta2)

    expected_world_to_camera = delta2 @ delta1
    assert not np.allclose(expected_world_to_camera, delta1 @ delta2)
    np.testing.assert_allclose(after_second, expected_world_to_camera, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.inv(after_second), np.linalg.inv(delta2 @ delta1), atol=1e-12
    )
