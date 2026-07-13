from __future__ import annotations

import numpy as np
import pytest

from gs_video.composite.alpha import composite_frame
from gs_video.domain.errors import RepairableError


def test_alpha_endpoints_select_exact_sources() -> None:
    foreground = np.full((2, 2, 3), 200, np.uint8)
    background = np.full((2, 2, 3), 20, np.uint8)
    alpha = np.array([[0, 255], [0, 255]], np.uint8)

    output = composite_frame(foreground, background, alpha, edge_px=0)

    np.testing.assert_array_equal(output[:, 0], background[:, 0])
    np.testing.assert_array_equal(output[:, 1], foreground[:, 1])
    assert output.dtype == np.uint8


def test_alpha_blending_is_deterministic_uint8() -> None:
    foreground = np.array([[[255, 101, 0]]], np.uint8)
    background = np.array([[[0, 0, 200]]], np.uint8)
    alpha = np.array([[128]], np.uint8)

    first = composite_frame(foreground, background, alpha, edge_px=0)
    second = composite_frame(foreground, background, alpha, edge_px=0)

    np.testing.assert_array_equal(first, np.array([[[128, 51, 100]]], np.uint8))
    np.testing.assert_array_equal(second, first)


def test_edge_erosion_changes_an_originally_opaque_boundary_pixel() -> None:
    foreground = np.full((7, 7, 3), 255, np.uint8)
    background = np.zeros_like(foreground)
    alpha = np.zeros((7, 7), np.uint8)
    alpha[1:6, 1:6] = 255

    unprocessed = composite_frame(foreground, background, alpha, edge_px=0)
    processed = composite_frame(foreground, background, alpha, edge_px=1)

    np.testing.assert_array_equal(unprocessed[1, 3], foreground[1, 3])
    assert np.all(processed[1, 3] < foreground[1, 3])
    np.testing.assert_array_equal(processed[0, 0], background[0, 0])


@pytest.mark.parametrize(
    ("foreground", "background", "alpha"),
    [
        (
            np.zeros((2, 2, 3), np.uint8),
            np.zeros((2, 3, 3), np.uint8),
            np.zeros((2, 2), np.uint8),
        ),
        (
            np.zeros((2, 2, 3), np.uint8),
            np.zeros((2, 2, 3), np.uint8),
            np.zeros((3, 2), np.uint8),
        ),
        (
            np.zeros((2, 2, 3), np.uint8),
            np.zeros((2, 2, 3), np.uint8),
            np.zeros((2, 2, 1), np.uint8),
        ),
    ],
)
def test_rejects_shape_or_channel_mismatch(
    foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray
) -> None:
    with pytest.raises(RepairableError, match="尺寸|形状|通道"):
        composite_frame(foreground, background, alpha, edge_px=0)


@pytest.mark.parametrize(
    ("foreground", "background", "alpha"),
    [
        (
            np.zeros((2, 2, 3), np.float32),
            np.zeros((2, 2, 3), np.uint8),
            np.zeros((2, 2), np.uint8),
        ),
        (
            np.zeros((2, 2, 3), np.uint8),
            np.zeros((2, 2, 3), np.int16),
            np.zeros((2, 2), np.uint8),
        ),
        (
            np.zeros((2, 2, 3), np.uint8),
            np.zeros((2, 2, 3), np.uint8),
            np.zeros((2, 2), np.float32),
        ),
    ],
)
def test_rejects_non_uint8_inputs(
    foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray
) -> None:
    with pytest.raises(RepairableError, match="uint8"):
        composite_frame(foreground, background, alpha, edge_px=0)


def test_rejects_non_ndarray_inputs() -> None:
    image = np.zeros((2, 2, 3), np.uint8)
    alpha = np.zeros((2, 2), np.uint8)

    with pytest.raises(RepairableError, match="ndarray"):
        composite_frame(image.tolist(), image, alpha, edge_px=0)  # type: ignore[arg-type]


def test_rejects_empty_frame_dimensions() -> None:
    image = np.empty((0, 2, 3), np.uint8)
    alpha = np.empty((0, 2), np.uint8)

    with pytest.raises(RepairableError, match="尺寸"):
        composite_frame(image, image, alpha, edge_px=0)


@pytest.mark.parametrize("edge_px", [-1, 4, True, 1.0, "1"])
def test_rejects_invalid_edge_width(edge_px: object) -> None:
    image = np.zeros((2, 2, 3), np.uint8)
    alpha = np.zeros((2, 2), np.uint8)

    with pytest.raises(RepairableError, match="edge_px"):
        composite_frame(image, image, alpha, edge_px=edge_px)  # type: ignore[arg-type]
