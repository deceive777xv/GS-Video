import numpy as np
import pytest

from gs_video.camera.classify import CameraKind
from gs_video.camera.mapping import map_ground_aligned_trajectory
from gs_video.camera.solution import CameraSolution, SourceGroundEstimate


def solution(*poses: np.ndarray) -> CameraSolution:
    return CameraSolution(
        intrinsics=np.eye(3),
        camera_to_world=poses,
        kind=CameraKind.SIX_DOF,
        confidence=1.0,
        source_ground=SourceGroundEstimate(
            normal=(0.0, 1.0, 0.0),
            offset=0.0,
            anchor_frame_index=0,
            confidence=1.0,
            support_ratio=1.0,
            rms_residual=0.0,
        ),
    )


def test_ground_alignment_maps_anchor_projection_and_uses_one_gs_scale() -> None:
    anchor = np.eye(4)
    anchor[:3, 3] = (2.0, 3.0, 4.0)
    moved = anchor.copy()
    moved[:3, 3] += (1.0, 0.0, 2.0)

    mapped = map_ground_aligned_trajectory(
        solution(anchor, moved),
        target_p0=(10.0, 20.0, 30.0),
        target_p1=(10.0, 20.0, 31.0),
        target_normal=(0.0, 1.0, 0.0),
        gs_scale=2.0,
    )

    np.testing.assert_allclose(mapped[0][:3, 3], (10.0, 26.0, 30.0), atol=1e-9)
    np.testing.assert_allclose(
        mapped[1][:3, 3] - mapped[0][:3, 3], (2.0, 0.0, 4.0), atol=1e-9
    )


def test_ground_alignment_applies_scene_azimuth_to_camera_trajectory() -> None:
    anchor = np.eye(4)
    moved = np.eye(4)
    moved[2, 3] = 1.0

    mapped = map_ground_aligned_trajectory(
        solution(anchor, moved),
        target_p0=(0.0, 0.0, 0.0),
        target_p1=(0.0, 0.0, 1.0),
        target_normal=(0.0, 1.0, 0.0),
        gs_scale=1.0,
        scene_azimuth_degrees=90.0,
    )

    np.testing.assert_allclose(
        mapped[1][:3, 3] - mapped[0][:3, 3], (1.0, 0.0, 0.0), atol=1e-9
    )


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_ground_alignment_rejects_invalid_gs_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="gs_scale"):
        map_ground_aligned_trajectory(
            solution(np.eye(4)),
            target_p0=(0.0, 0.0, 0.0),
            target_p1=(0.0, 0.0, 1.0),
            target_normal=(0.0, 1.0, 0.0),
            gs_scale=scale,
        )


def test_ground_alignment_requires_audited_source_ground() -> None:
    missing = CameraSolution(
        intrinsics=np.eye(3),
        camera_to_world=[np.eye(4)],
        kind=CameraKind.SIX_DOF,
        confidence=1.0,
    )

    with pytest.raises(ValueError, match="source ground"):
        map_ground_aligned_trajectory(
            missing,
            target_p0=(0.0, 0.0, 0.0),
            target_p1=(0.0, 0.0, 1.0),
            target_normal=(0.0, 1.0, 0.0),
            gs_scale=1.0,
        )
