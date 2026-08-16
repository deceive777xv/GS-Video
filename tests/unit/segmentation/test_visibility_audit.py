from __future__ import annotations

import numpy as np
import pytest

from gs_video.segmentation.visibility_audit import audit_visibility_masks


def _mask(*, bottom: int | None) -> np.ndarray:
    mask = np.zeros((100, 80), dtype=np.uint8)
    if bottom is not None:
        mask[20:bottom, 30:50] = 255
    return mask


def test_audit_scans_all_frames_and_merges_contiguous_candidate_ranges() -> None:
    result = audit_visibility_masks(
        [
            _mask(bottom=80),
            _mask(bottom=85),
            _mask(bottom=100),
            _mask(bottom=100),
            _mask(bottom=None),
            _mask(bottom=75),
        ]
    )

    assert [(item.start_frame, item.end_frame) for item in result.fully_visible_ranges] == [
        (0, 1),
        (5, 5),
    ]
    assert [(item.start_frame, item.end_frame) for item in result.bottom_cropped_ranges] == [(2, 3)]
    assert [(item.start_frame, item.end_frame) for item in result.uncertain_ranges] == [(4, 4)]
    assert result.fully_visible_ranges[0].review_frames == (0, 1)
    assert result.recommended_anchor_frames == (1, 5)


def test_audit_does_not_call_empty_or_tiny_alpha_a_visible_foot_candidate() -> None:
    tiny = np.zeros((100, 80), dtype=np.uint8)
    tiny[30, 40] = 255

    result = audit_visibility_masks([_mask(bottom=None), tiny])

    assert len(result.uncertain_ranges) == 1
    assert result.uncertain_ranges[0].start_frame == 0
    assert result.uncertain_ranges[0].end_frame == 1
    assert result.recommended_anchor_frames == ()


def test_audit_rejects_missing_or_non_grayscale_masks() -> None:
    with pytest.raises(ValueError, match="at least one"):
        audit_visibility_masks([])
    with pytest.raises(ValueError, match="grayscale"):
        audit_visibility_masks([np.zeros((10, 10, 3), dtype=np.uint8)])
