from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from typing import cast

from gs_video.domain.errors import RepairableError
from gs_video.domain.models import MatteRefinementSettings
from gs_video.postprocess.color import linear_to_rec709, rec709_to_linear


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
    settings = MatteRefinementSettings(
        enabled=edge_px > 0,
        edge_offset=-float(edge_px),
        feather_radius=float(edge_px),
    )
    output = composite_frame_16bit(foreground, background, alpha, settings)
    return np.asarray(np.rint(output.astype(np.float32) / 257.0), dtype=np.uint8)


def refine_matte(
    alpha: NDArray[np.uint8],
    settings: MatteRefinementSettings,
    *,
    spatial_scale: float = 1.0,
) -> NDArray[np.float32]:
    if not np.isfinite(spatial_scale) or spatial_scale <= 0:
        raise RepairableError("spatial_scale 必须是有限正数")
    matte = alpha.astype(np.float32) / np.float32(255.0)
    if not settings.enabled:
        return matte
    offset = settings.edge_offset * spatial_scale
    radius = int(np.ceil(abs(offset)))
    if radius:
        size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        operation = cv2.dilate if offset > 0 else cv2.erode
        adjusted = cast(
            NDArray[np.float32],
            operation(
                matte,
                kernel,
                iterations=1,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            ),
        )
        blend = abs(offset) / max(radius, 1)
        matte = matte * np.float32(1.0 - blend) + adjusted * np.float32(blend)
    feather = settings.feather_radius * spatial_scale
    if feather > 0:
        matte = cast(
            NDArray[np.float32],
            cv2.GaussianBlur(
                matte,
                (0, 0),
                sigmaX=max(0.25, feather / 2.0),
                sigmaY=max(0.25, feather / 2.0),
                borderType=cv2.BORDER_CONSTANT,
            ),
        )
    return cast(NDArray[np.float32], np.clip(matte, 0.0, 1.0))


def _decontaminate_foreground(
    foreground: NDArray[np.float32],
    matte: NDArray[np.float32],
    settings: MatteRefinementSettings,
    spatial_scale: float,
) -> NDArray[np.float32]:
    strength = settings.decontaminate_strength / 100.0
    if strength <= 0:
        return foreground
    radius = max(0.25, settings.decontaminate_radius * spatial_scale / 2.0)
    blurred_alpha = cv2.GaussianBlur(matte, (0, 0), radius, borderType=cv2.BORDER_REPLICATE)
    premultiplied = foreground * matte[..., None]
    sampled = cv2.GaussianBlur(
        premultiplied, (0, 0), radius, borderType=cv2.BORDER_REPLICATE
    ) / np.maximum(blurred_alpha[..., None], np.float32(1e-5))
    edge_weight = np.clip(4.0 * matte * (1.0 - matte), 0.0, 1.0)[..., None]
    return np.asarray(
        foreground * (1.0 - edge_weight * strength)
        + sampled * (edge_weight * strength),
        dtype=np.float32,
    )


def composite_frame_16bit(
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,
    settings: MatteRefinementSettings,
    *,
    spatial_scale: float = 1.0,
) -> NDArray[np.uint16]:
    """Refine the matte and composite SDR Rec.709 sources in linear light."""
    foreground, background, alpha, _edge_px = _validate_inputs(
        foreground, background, alpha, 0
    )
    matte = refine_matte(alpha, settings, spatial_scale=spatial_scale)
    foreground_encoded = foreground.astype(np.float32) / np.float32(255.0)
    foreground_encoded = _decontaminate_foreground(
        foreground_encoded, matte, settings, spatial_scale
    )
    background_encoded = background.astype(np.float32) / np.float32(255.0)
    foreground_linear = rec709_to_linear(foreground_encoded)
    background_linear = rec709_to_linear(background_encoded)
    weight = matte[..., None]
    blended_linear = (
        foreground_linear * weight
        + background_linear * (np.float32(1.0) - weight)
    )
    encoded = linear_to_rec709(blended_linear)
    output = np.asarray(
        np.rint(np.clip(encoded, 0.0, 1.0) * np.float32(65535.0)),
        dtype=np.uint16,
    )
    np.copyto(output, background.astype(np.uint16) * 257, where=(weight == 0.0))
    np.copyto(output, foreground.astype(np.uint16) * 257, where=(weight == 1.0))
    return output
