from __future__ import annotations

import numpy as np
import pytest

from gs_video.scene.camera import OrbitCamera


def test_intrinsics_put_principal_point_at_image_center() -> None:
    camera = OrbitCamera(target=(0, 0, 0), distance=2, yaw=0, pitch=0, fov_y_degrees=60)

    intrinsics = camera.intrinsics(1920, 1080)

    assert intrinsics[0, 2] == pytest.approx(960)
    assert intrinsics[1, 2] == pytest.approx(540)
    assert intrinsics[0, 0] == pytest.approx(intrinsics[1, 1])


def test_smaller_vertical_fov_increases_focal_length() -> None:
    narrow = OrbitCamera((0, 0, 0), 2, 0, 0, 30).intrinsics(1920, 1080)
    wide = OrbitCamera((0, 0, 0), 2, 0, 0, 90).intrinsics(1920, 1080)

    assert narrow[1, 1] > wide[1, 1]


def test_view_matrix_is_inverse_of_camera_to_world() -> None:
    camera = OrbitCamera(target=(1, 2, 3), distance=4, yaw=35, pitch=-20, fov_y_degrees=60)

    assert camera.view_matrix() @ camera.camera_to_world() == pytest.approx(np.eye(4))


def test_camera_rotation_is_orthonormal_and_right_handed() -> None:
    camera = OrbitCamera(target=(1, 2, 3), distance=4, yaw=35, pitch=-20, fov_y_degrees=60)
    rotation = camera.camera_to_world()[:3, :3]

    assert rotation.T @ rotation == pytest.approx(np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_zero_orbit_uses_opencv_axes_and_looks_at_target() -> None:
    camera = OrbitCamera(target=(0, 0, 0), distance=2, yaw=0, pitch=0, fov_y_degrees=60)
    camera_to_world = camera.camera_to_world()

    assert camera_to_world[:3, :3] == pytest.approx(np.eye(3))
    assert camera_to_world[:3, 3] == pytest.approx([0, 0, -2])
    assert camera.view_matrix() @ np.array([0, 0, 0, 1]) == pytest.approx([0, 0, 2, 1])


@pytest.mark.parametrize("distance", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_distance(distance: float) -> None:
    with pytest.raises(ValueError, match="distance"):
        OrbitCamera((0, 0, 0), distance, 0, 0, 60)


@pytest.mark.parametrize("fov", [0, -1, 180, 181, float("nan"), float("inf")])
def test_rejects_invalid_vertical_fov(fov: float) -> None:
    with pytest.raises(ValueError, match="fov_y_degrees"):
        OrbitCamera((0, 0, 0), 2, 0, 0, fov)


@pytest.mark.parametrize(("width", "height"), [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_intrinsics_reject_non_positive_dimensions(width: int, height: int) -> None:
    camera = OrbitCamera((0, 0, 0), 2, 0, 0, 60)

    with pytest.raises(ValueError, match="dimensions"):
        camera.intrinsics(width, height)
