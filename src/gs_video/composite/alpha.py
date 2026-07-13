from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from typing import cast

from gs_video.domain.errors import RepairableError


def _validate_inputs(
    foreground: object,
    background: object,
    alpha: object,
    edge_px: object,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], int]:
    if not all(isinstance(value, np.ndarray) for value in (foreground, background, alpha)):
        raise RepairableError("前景、背景和 Alpha 必须是 ndarray")
    assert isinstance(foreground, np.ndarray)
    assert isinstance(background, np.ndarray)
    assert isinstance(alpha, np.ndarray)
    if foreground.ndim != 3 or foreground.shape[2:] != (3,):
        raise RepairableError("前景必须是 H×W×3 RGB 图像")
    if background.ndim != 3 or background.shape[2:] != (3,):
        raise RepairableError("背景必须是 H×W×3 RGB 图像")
    if alpha.ndim != 2:
        raise RepairableError("Alpha 必须是 H×W 单通道图像")
    if foreground.shape != background.shape or foreground.shape[:2] != alpha.shape:
        raise RepairableError("前景、背景和 Alpha 尺寸不一致")
    if foreground.shape[0] <= 0 or foreground.shape[1] <= 0:
        raise RepairableError("前景、背景和 Alpha 尺寸必须为正")
    if foreground.dtype != np.uint8 or background.dtype != np.uint8 or alpha.dtype != np.uint8:
        raise RepairableError("前景、背景和 Alpha 必须使用 uint8 dtype")
    # Keep these checks explicit: uint8 currently guarantees them, and the checks document
    # the compositor's numeric contract if supported dtypes are extended later.
    if not all(np.isfinite(value).all() for value in (foreground, background, alpha)):
        raise RepairableError("前景、背景和 Alpha 必须只包含有限数值")
    if any(np.any(value < 0) or np.any(value > 255) for value in (foreground, background, alpha)):
        raise RepairableError("前景、背景和 Alpha 必须位于 0..255")
    if isinstance(edge_px, bool) or not isinstance(edge_px, int) or not 0 <= edge_px <= 3:
        raise RepairableError("edge_px 必须是 0..3 的整数")
    return foreground, background, alpha, edge_px


def composite_frame(
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,
    edge_px: int,
) -> NDArray[np.uint8]:
    """Composite exact-size RGB uint8 frames using an optional processed alpha edge."""
    foreground, background, alpha, edge_px = _validate_inputs(
        foreground, background, alpha, edge_px
    )
    matte = alpha.astype(np.float32) / np.float32(255.0)
    if edge_px:
        size = edge_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        matte = cast(
            NDArray[np.float32],
            cv2.erode(
                matte,
                kernel,
                iterations=1,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            ),
        )
        sigma = max(0.5, edge_px / 2)
        matte = cast(
            NDArray[np.float32],
            cv2.GaussianBlur(
                matte,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_CONSTANT,
            ),
        )
        matte = cast(NDArray[np.float32], np.clip(matte, 0.0, 1.0))

    weight = matte[..., None]
    blended = (
        foreground.astype(np.float32) * weight
        + background.astype(np.float32) * (np.float32(1.0) - weight)
    )
    output = np.asarray(np.rint(np.clip(blended, 0.0, 255.0)), dtype=np.uint8)
    # Preserve mathematically exact endpoints of the processed matte without consulting the
    # original alpha (which would undo erosion at originally-opaque boundary pixels).
    np.copyto(output, background, where=(weight == 0.0))
    np.copyto(output, foreground, where=(weight == 1.0))
    return output
