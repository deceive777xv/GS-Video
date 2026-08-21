from __future__ import annotations

import numpy as np
import pytest

from gs_video.domain.contracts import PickBuffer
from gs_video.scene.ground_fit import fit_ground_from_gaussians


def _fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    PickBuffer,
    np.ndarray,
]:
    width, height = 320, 180
    intrinsics = np.array([[180.0, 0.0, 160.0], [0.0, 180.0, 90.0], [0.0, 0.0, 1.0]])
    origin = np.array((0.0, 2.0, 0.0))
    forward = np.array((0.0, -2.0, 5.0))
    forward /= np.linalg.norm(forward)
    right = np.array((1.0, 0.0, 0.0))
    down = np.cross(forward, right)
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = np.column_stack((right, down, forward))
    camera_to_world[:3, 3] = origin

    x, z = np.meshgrid(np.linspace(-2.2, 2.2, 45), np.linspace(2.5, 7.5, 50))
    means = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    scales = np.full_like(means, 0.04)
    opacities = np.full(means.shape[0], 0.92)

    world_to_camera = np.linalg.inv(camera_to_world)
    camera_points = (world_to_camera @ np.column_stack((means, np.ones(means.shape[0]))).T).T
    projected = (intrinsics @ camera_points[:, :3].T).T
    pixels = projected[:, :2] / projected[:, 2:3]

    yy, xx = np.mgrid[:height, :width]
    rays = np.stack(
        ((xx + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0],
         (yy + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1], np.ones_like(xx)),
        axis=-1,
    )
    world_directions = rays @ camera_to_world[:3, :3].T
    depth = -origin[1] / world_directions[..., 1]
    valid = depth > 0
    depth = np.where(valid, depth, 0).astype(np.float32)
    opacity_map = np.where(valid, 0.95, 0).astype(np.float32)
    buffer = PickBuffer(
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        expected_depth=depth,
        opacity=opacity_map,
    )
    return means, scales, opacities, camera_to_world, intrinsics, buffer, pixels


def test_three_approximate_hints_fit_one_visible_gaussian_ground_plane() -> None:
    means, scales, opacities, camera_to_world, intrinsics, buffer, pixels = _fixture()
    chosen = (400, 1100, 1850)
    hints = tuple((int(pixels[index, 0]), int(pixels[index, 1])) for index in chosen)

    candidate = fit_ground_from_gaussians(
        means_world=means,
        scales_world=scales,
        opacities=opacities,
        camera_to_world=camera_to_world,
        intrinsics=intrinsics,
        pick_buffer=buffer,
        hints=hints,
    )

    np.testing.assert_allclose(candidate.plane_normal, (0.0, 1.0, 0.0), atol=1e-6)
    assert candidate.plane_offset == pytest.approx(0.0, abs=1e-6)
    assert all(count >= 8 for count in candidate.support_counts)
    assert candidate.weighted_inlier_ratio > 0.95
    assert candidate.rms_residual < 1e-6
    assert candidate.confidence > 0.9
    assert all(abs(point[1]) < 1e-6 for point in candidate.refined_points)


def test_ground_fit_rejects_hint_crossing_a_depth_discontinuity() -> None:
    means, scales, opacities, camera_to_world, intrinsics, buffer, pixels = _fixture()
    buffer.expected_depth[65:115, 130:190] *= 2.0
    chosen = (400, 1100, 1850)
    hints = tuple((int(pixels[index, 0]), int(pixels[index, 1])) for index in chosen)

    with pytest.raises(ValueError, match="mixed|discontinuous"):
        fit_ground_from_gaussians(
            means_world=means,
            scales_world=scales,
            opacities=opacities,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            pick_buffer=buffer,
            hints=hints,
        )
