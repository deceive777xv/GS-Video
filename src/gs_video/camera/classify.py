from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


FIXED_FLOW_THRESHOLD_PX = 0.5
ROTATION_HOMOGRAPHY_INLIER_THRESHOLD = 0.7
SIX_DOF_ESSENTIAL_INLIER_THRESHOLD = 0.6
SIX_DOF_CHEIRALITY_INLIER_THRESHOLD = 0.25
MIN_OVERALL_CONFIDENCE = 0.55
FAILED_PAIR_CONFIDENCE = 0.4
MIN_TRACKED_FEATURES = 80
MAX_CONSECUTIVE_PAIR_FAILURES = 5


class CameraKind(StrEnum):
    FIXED = "fixed"
    ROTATION = "rotation"
    SIX_DOF = "6dof"


@dataclass(frozen=True)
class MotionClassification:
    kind: CameraKind
    confidence: float


def classify_motion(
    median_flow_px: float,
    homography_inliers: float,
    essential_inliers: float,
) -> MotionClassification:
    """Classify one adjacent pair using centralized, deterministic thresholds."""

    if not isfinite(median_flow_px) or median_flow_px < 0:
        raise ValueError("median_flow_px must be finite and non-negative")
    for value in (homography_inliers, essential_inliers):
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError("inlier ratios must be finite and between zero and one")

    if median_flow_px <= FIXED_FLOW_THRESHOLD_PX:
        stillness = 1.0 - 0.4 * median_flow_px / FIXED_FLOW_THRESHOLD_PX
        return MotionClassification(CameraKind.FIXED, max(stillness, homography_inliers))
    if essential_inliers >= SIX_DOF_ESSENTIAL_INLIER_THRESHOLD:
        return MotionClassification(CameraKind.SIX_DOF, essential_inliers)
    return MotionClassification(CameraKind.ROTATION, homography_inliers)
