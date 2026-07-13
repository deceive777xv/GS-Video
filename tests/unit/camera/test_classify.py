from __future__ import annotations

import pytest

from gs_video.camera.classify import (
    FIXED_FLOW_THRESHOLD_PX,
    SIX_DOF_ESSENTIAL_INLIER_THRESHOLD,
    CameraKind,
    classify_motion,
)


def test_fixed_when_median_flow_is_subpixel() -> None:
    result = classify_motion(0.3, 0.99, 0.1)

    assert result.kind is CameraKind.FIXED
    assert result.confidence >= 0.55


def test_fixed_threshold_is_inclusive() -> None:
    result = classify_motion(FIXED_FLOW_THRESHOLD_PX, 0.8, 0.8)

    assert result.kind is CameraKind.FIXED


def test_strong_homography_with_weak_essential_is_rotation() -> None:
    result = classify_motion(3.0, 0.92, SIX_DOF_ESSENTIAL_INLIER_THRESHOLD - 0.01)

    assert result.kind is CameraKind.ROTATION
    assert result.confidence == pytest.approx(0.92)


def test_essential_threshold_is_inclusive_for_six_dof() -> None:
    result = classify_motion(3.0, 0.85, SIX_DOF_ESSENTIAL_INLIER_THRESHOLD)

    assert result.kind is CameraKind.SIX_DOF
    assert result.confidence == pytest.approx(SIX_DOF_ESSENTIAL_INLIER_THRESHOLD)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_classification_rejects_invalid_flow(value: float) -> None:
    with pytest.raises(ValueError, match="median_flow"):
        classify_motion(value, 0.9, 0.9)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_classification_rejects_invalid_inlier_ratios(value: float) -> None:
    with pytest.raises(ValueError, match="inlier"):
        classify_motion(2.0, value, 0.9)
