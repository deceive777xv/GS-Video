from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


class LowerBoundaryCandidate(StrEnum):
    FULLY_VISIBLE = "fully_visible"
    BOTTOM_CROPPED = "bottom_cropped"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class VisibilityCandidateRange:
    start_frame: int
    end_frame: int
    review_frames: tuple[int, ...]


@dataclass(frozen=True)
class VisibilityAuditResult:
    fully_visible_ranges: tuple[VisibilityCandidateRange, ...]
    bottom_cropped_ranges: tuple[VisibilityCandidateRange, ...]
    uncertain_ranges: tuple[VisibilityCandidateRange, ...]
    recommended_anchor_frames: tuple[int, ...]


def _classify_mask(mask: np.ndarray) -> LowerBoundaryCandidate:
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("visibility masks must be non-empty grayscale images")
    foreground = mask > 127
    rows, columns = np.nonzero(foreground)
    if len(rows) == 0:
        return LowerBoundaryCandidate.UNCERTAIN
    height, width = foreground.shape
    area_ratio = float(foreground.mean())
    if area_ratio < 0.001 or len(np.unique(rows)) < 3 or len(np.unique(columns)) < 2:
        return LowerBoundaryCandidate.UNCERTAIN
    bottom_margin = max(2, int(round(height * 0.02)))
    if int(rows.max()) >= height - bottom_margin:
        return LowerBoundaryCandidate.BOTTOM_CROPPED
    return LowerBoundaryCandidate.FULLY_VISIBLE


def _review_frames(start: int, end: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys((start, (start + end) // 2, end)))


def _ranges(
    classifications: list[LowerBoundaryCandidate],
    target: LowerBoundaryCandidate,
) -> tuple[VisibilityCandidateRange, ...]:
    result: list[VisibilityCandidateRange] = []
    start: int | None = None
    for index, classification in enumerate((*classifications, None)):
        if classification is target and start is None:
            start = index
        elif classification is not target and start is not None:
            end = index - 1
            result.append(VisibilityCandidateRange(start, end, _review_frames(start, end)))
            start = None
    return tuple(result)


def audit_visibility_masks(masks: Iterable[np.ndarray]) -> VisibilityAuditResult:
    classifications = [_classify_mask(np.asarray(mask)) for mask in masks]
    if not classifications:
        raise ValueError("visibility audit requires at least one mask")
    fully_visible = _ranges(classifications, LowerBoundaryCandidate.FULLY_VISIBLE)
    cropped = _ranges(classifications, LowerBoundaryCandidate.BOTTOM_CROPPED)
    uncertain = _ranges(classifications, LowerBoundaryCandidate.UNCERTAIN)
    recommended = tuple(item.review_frames[len(item.review_frames) // 2] for item in fully_visible[:8])
    return VisibilityAuditResult(fully_visible, cropped, uncertain, recommended)


def audit_visibility_mask_paths(paths: Iterable[Path]) -> VisibilityAuditResult:
    def decoded_masks() -> Iterable[np.ndarray]:
        for path in paths:
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "L":
                    raise ValueError("visibility audit masks must be grayscale PNG files")
                yield np.asarray(image, dtype=np.uint8)

    return audit_visibility_masks(decoded_masks())
