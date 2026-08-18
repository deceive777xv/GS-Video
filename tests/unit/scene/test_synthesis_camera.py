from __future__ import annotations

import numpy as np
import pytest

from gs_video.scene.synthesis_camera import (
    LocalGroundPlane,
    SourcePerspective,
    SynthesisCameraRig,
)


def _reference_camera() -> np.ndarray:
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 3] = np.array([0.0, -2.0, -5.0])
    return camera_to_world


def _ground() -> LocalGroundPlane:
    return LocalGroundPlane.from_points(
        p0=(0.0, 0.0, 0.0),
        p1=(0.0, 0.0, 2.0),
        p2=(2.0, 0.0, 0.0),
        reference_camera_position=(0.0, -2.0, -5.0),
    )


def _perspective() -> SourcePerspective:
    return SourcePerspective.from_horizon(
        width=1920,
        height=1080,
        fov_y_degrees=60.0,
        horizon_start=(0.0, 540.0),
        horizon_end=(1920.0, 540.0),
    )


def _project(camera_to_world: np.ndarray, intrinsics: np.ndarray, point: np.ndarray) -> np.ndarray:
    world_to_camera = np.linalg.inv(camera_to_world)
    homogeneous = np.append(point, 1.0)
    camera_point = (world_to_camera @ homogeneous)[:3]
    pixel = intrinsics @ camera_point
    return pixel[:2] / pixel[2]


def test_local_ground_orients_normal_toward_reference_camera() -> None:
    ground = _ground()

    np.testing.assert_allclose(ground.normal, (0.0, -1.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(ground.basis_u, (0.0, 0.0, 1.0), atol=1e-9)
    np.testing.assert_allclose(ground.basis_v, (-1.0, 0.0, 0.0), atol=1e-9)


def test_local_ground_rejects_collinear_points() -> None:
    with pytest.raises(ValueError, match="not be collinear"):
        LocalGroundPlane.from_points(
            p0=(0.0, 0.0, 0.0),
            p1=(1.0, 0.0, 0.0),
            p2=(2.0, 0.0, 0.0),
            reference_camera_position=(0.0, -2.0, -5.0),
        )


def test_horizontal_horizon_recovers_camera_up_direction() -> None:
    perspective = _perspective()

    np.testing.assert_allclose(perspective.up_camera, (0.0, -1.0, 0.0), atol=1e-9)


def _guide_pair(vanishing_point: tuple[float, float]) -> tuple[
    tuple[tuple[float, float], tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float]],
]:
    def segment(start: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
        return start, (
            start[0] + 0.28 * (vanishing_point[0] - start[0]),
            start[1] + 0.28 * (vanishing_point[1] - start[1]),
        )

    return segment((220.0, 260.0)), segment((420.0, 880.0))


def test_orthogonal_guides_recover_vertical_fov_and_horizontal_plane_normal() -> None:
    width, height, expected_fov = 1920, 1080, 60.0
    focal = 0.5 * height / np.tan(np.radians(expected_fov) / 2)

    perspective = SourcePerspective.from_orthogonal_guides(
        width=width,
        height=height,
        group_a=_guide_pair((width / 2 + focal, height / 2)),
        group_b=_guide_pair((width / 2 - focal, height / 2)),
        reference_relation="both_horizontal_plane",
    )

    assert perspective.fov_y_degrees == pytest.approx(expected_fov)
    np.testing.assert_allclose(perspective.up_camera, (0.0, -1.0, 0.0), atol=1e-8)


def test_orthogonal_guides_reject_parallel_and_non_positive_focal_evidence() -> None:
    parallel = (((100.0, 50.0), (300.0, 100.0)), ((100.0, 150.0), (300.0, 200.0)))
    with pytest.raises(ValueError, match="parallel or nearly parallel"):
        SourcePerspective.from_orthogonal_guides(
            width=640,
            height=360,
            group_a=parallel,
            group_b=parallel,
            reference_relation="a_vertical_b_horizontal",
        )

    nearly_parallel = (
        ((100.0, 100.0), (500.0, 100.0)),
        ((100.0, 200.0), (500.0, 200.01)),
    )
    with pytest.raises(ValueError, match="parallel or nearly parallel"):
        SourcePerspective.from_orthogonal_guides(
            width=640,
            height=360,
            group_a=nearly_parallel,
            group_b=nearly_parallel,
            reference_relation="a_vertical_b_horizontal",
        )


@pytest.mark.parametrize("azimuth", [0.0, 45.0, 90.0])
def test_contact_mode_keeps_selected_pixel_bound_to_p0(azimuth: float) -> None:
    rig = SynthesisCameraRig(
        ground=_ground(),
        source_perspective=_perspective(),
        reference_camera_to_world=_reference_camera(),
    )
    contact_pixel = np.array([720.0, 920.0])

    camera = rig.solve_contact(
        contact_pixel=tuple(contact_pixel),
        scene_azimuth_degrees=azimuth,
        subject_to_scene_scale=1.0,
    )

    projected = _project(camera.camera_to_world(), camera.intrinsics(1920, 1080), _ground().p0)
    np.testing.assert_allclose(projected, contact_pixel, atol=1e-7)
    np.testing.assert_allclose(camera.camera_to_world()[:3, :3].T @ camera.camera_to_world()[:3, :3], np.eye(3), atol=1e-9)
    assert np.linalg.det(camera.camera_to_world()[:3, :3]) == pytest.approx(1.0)


def test_contact_scale_scales_camera_height() -> None:
    rig = SynthesisCameraRig(
        ground=_ground(),
        source_perspective=_perspective(),
        reference_camera_to_world=_reference_camera(),
    )

    camera_1 = rig.solve_contact(contact_pixel=(960.0, 900.0), subject_to_scene_scale=1.0)
    camera_2 = rig.solve_contact(contact_pixel=(960.0, 900.0), subject_to_scene_scale=1.75)
    normal = _ground().normal
    height_1 = np.dot(camera_1.camera_to_world()[:3, 3] - _ground().p0, normal)
    height_2 = np.dot(camera_2.camera_to_world()[:3, 3] - _ground().p0, normal)

    assert height_2 == pytest.approx(height_1 * 1.75)


def test_perspective_mode_offset_moves_position_without_changing_rotation_or_intrinsics() -> None:
    rig = SynthesisCameraRig(
        ground=_ground(),
        source_perspective=_perspective(),
        reference_camera_to_world=_reference_camera(),
    )

    base = rig.solve_perspective(scene_azimuth_degrees=30.0)
    shifted = rig.solve_perspective(
        scene_azimuth_degrees=30.0,
        composition_offset=(1.25, -0.5),
    )

    np.testing.assert_allclose(base.camera_to_world()[:3, :3], shifted.camera_to_world()[:3, :3])
    np.testing.assert_allclose(base.intrinsics(1920, 1080), shifted.intrinsics(1920, 1080))
    np.testing.assert_allclose(
        shifted.camera_to_world()[:3, 3] - base.camera_to_world()[:3, 3],
        _ground().basis_u * 1.25 + _ground().basis_v * -0.5,
    )


def test_perspective_scale_also_scales_local_composition_offset() -> None:
    rig = SynthesisCameraRig(
        ground=_ground(),
        source_perspective=_perspective(),
        reference_camera_to_world=_reference_camera(),
    )

    base = rig.solve_perspective(subject_to_scene_scale=2.0)
    shifted = rig.solve_perspective(
        subject_to_scene_scale=2.0,
        composition_offset=(1.25, -0.5),
    )

    np.testing.assert_allclose(
        shifted.camera_to_world()[:3, 3] - base.camera_to_world()[:3, 3],
        2.0 * (_ground().basis_u * 1.25 + _ground().basis_v * -0.5),
    )


def test_explicitly_flipped_ground_normal_remains_solvable() -> None:
    ground = _ground()
    flipped = LocalGroundPlane(
        p0=ground.p0,
        p1=ground.p1,
        p2=ground.p2,
        normal=-ground.normal,
        basis_u=ground.basis_u,
        basis_v=-ground.basis_v,
    )
    rig = SynthesisCameraRig(
        ground=flipped,
        source_perspective=_perspective(),
        reference_camera_to_world=_reference_camera(),
    )

    camera = rig.solve_perspective()

    assert np.dot(camera.camera_to_world()[:3, 3] - flipped.p0, flipped.normal) > 0


def test_perspective_azimuth_rotates_camera_about_local_normal() -> None:
    rig = SynthesisCameraRig(
        ground=_ground(),
        source_perspective=_perspective(),
        reference_camera_to_world=_reference_camera(),
    )

    camera_0 = rig.solve_perspective(scene_azimuth_degrees=0.0)
    camera_90 = rig.solve_perspective(scene_azimuth_degrees=90.0)
    horizontal_0 = camera_0.camera_to_world()[:3, 3] - _ground().p0
    horizontal_90 = camera_90.camera_to_world()[:3, 3] - _ground().p0
    horizontal_0 -= np.dot(horizontal_0, _ground().normal) * _ground().normal
    horizontal_90 -= np.dot(horizontal_90, _ground().normal) * _ground().normal

    assert np.linalg.norm(horizontal_90) == pytest.approx(np.linalg.norm(horizontal_0))
    assert np.dot(horizontal_0, horizontal_90) == pytest.approx(0.0, abs=1e-9)


def test_contact_mode_rejects_pixel_ray_that_cannot_hit_ground() -> None:
    rig = SynthesisCameraRig(
        ground=_ground(),
        source_perspective=_perspective(),
        reference_camera_to_world=_reference_camera(),
    )

    with pytest.raises(ValueError, match="does not point toward the local ground"):
        rig.solve_contact(contact_pixel=(960.0, 200.0))
