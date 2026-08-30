from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from gs_video.domain.models import (
    BloomEffect,
    EffectInstance,
    Lut3DEffect,
    PrimaryCorrectionEffect,
    SharpenEffect,
    VignetteEffect,
)
from gs_video.postprocess.color import linear_to_rec709, rec709_to_linear
from gs_video.postprocess.lut import CubeLut, apply_tetrahedral


LUMA = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
LutResolver = Callable[[str], CubeLut]


def uint16_to_float(image: NDArray[np.uint16]) -> NDArray[np.float32]:
    if image.dtype != np.uint16 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("post-process input must be an HxWx3 uint16 RGB image")
    return np.asarray(image, dtype=np.float32) / np.float32(65535.0)


def float_to_uint16(image: NDArray[np.floating]) -> NDArray[np.uint16]:
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("post-process output must be a finite HxWx3 RGB image")
    return np.asarray(
        np.rint(np.clip(rgb, 0.0, 1.0) * np.float32(65535.0)), dtype=np.uint16
    )


def _primary(
    image: NDArray[np.float32], effect: PrimaryCorrectionEffect
) -> NDArray[np.float32]:
    settings = effect.parameters
    linear = rec709_to_linear(image)
    linear *= np.float32(2.0**settings.exposure)
    corrected = linear_to_rec709(linear)

    contrast = np.float32(1.0 + settings.contrast / 100.0)
    corrected = (corrected - np.float32(0.5)) * contrast + np.float32(0.5)
    luma = np.sum(corrected * LUMA, axis=-1, keepdims=True)
    if settings.highlights:
        highlight_mask = np.clip((luma - 0.5) * 2.0, 0.0, 1.0)
        corrected += highlight_mask * np.float32(settings.highlights / 100.0) * 0.5
    if settings.shadows:
        shadow_mask = np.clip((0.5 - luma) * 2.0, 0.0, 1.0)
        corrected += shadow_mask * np.float32(settings.shadows / 100.0) * 0.5

    temperature = np.float32(settings.temperature / 100.0 * 0.15)
    tint = np.float32(settings.tint / 100.0 * 0.10)
    white_balance = np.asarray(
        (1.0 + temperature, 1.0 + tint, 1.0 - temperature), dtype=np.float32
    )
    corrected *= white_balance

    luma = np.sum(corrected * LUMA, axis=-1, keepdims=True)
    saturation = np.float32(settings.saturation / 100.0)
    corrected = luma + (corrected - luma) * saturation
    if settings.vibrance:
        chroma = np.max(corrected, axis=-1, keepdims=True) - np.min(
            corrected, axis=-1, keepdims=True
        )
        room = np.clip(1.0 - chroma, 0.0, 1.0)
        vibrance = np.float32(settings.vibrance / 100.0)
        corrected = luma + (corrected - luma) * (1.0 + vibrance * room)
    return np.asarray(corrected, dtype=np.float32)


def _bloom(
    image: NDArray[np.float32], effect: BloomEffect, spatial_scale: float
) -> NDArray[np.float32]:
    settings = effect.parameters
    luma = np.sum(image * LUMA, axis=-1, keepdims=True)
    threshold = np.float32(settings.threshold / 100.0)
    knee = np.float32(max(settings.soft_knee / 100.0 * threshold, 1e-6))
    soft = np.clip((luma - threshold + knee) / (2.0 * knee), 0.0, 1.0)
    contribution = np.maximum(luma - threshold, 0.0) + soft * soft * knee * 0.5
    highlights = image * contribution / np.maximum(luma, np.float32(1e-6))
    sigma = max(0.25, settings.radius * spatial_scale / 3.0)
    blurred = cv2.GaussianBlur(
        highlights, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101
    )
    tint = np.asarray(settings.tint, dtype=np.float32)
    return np.asarray(
        image + blurred * tint * np.float32(settings.intensity / 100.0),
        dtype=np.float32,
    )


def _vignette(
    image: NDArray[np.float32], effect: VignetteEffect
) -> NDArray[np.float32]:
    settings = effect.parameters
    height, width = image.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (x + 0.5) / max(width, 1) * 2.0 - 1.0 - settings.center_x / 100.0
    y = (y + 0.5) / max(height, 1) * 2.0 - 1.0 - settings.center_y / 100.0
    aspect = width / max(height, 1)
    roundness = settings.roundness / 100.0
    x_scale = aspect ** (-roundness * 0.5)
    y_scale = aspect ** (roundness * 0.5)
    distance = np.sqrt((x * x_scale) ** 2 + (y * y_scale) ** 2)
    start = settings.midpoint / 100.0
    feather = max(settings.feather / 100.0, 1e-4)
    transition = np.clip((distance - start) / feather, 0.0, 1.0)
    transition = transition * transition * (3.0 - 2.0 * transition)
    multiplier = 1.0 - transition[..., None] * (settings.amount / 100.0)
    return np.asarray(image * multiplier, dtype=np.float32)


def _sharpen(
    image: NDArray[np.float32], effect: SharpenEffect, spatial_scale: float
) -> NDArray[np.float32]:
    settings = effect.parameters
    luma = np.sum(image * LUMA, axis=-1)
    sigma = max(0.1, settings.radius * spatial_scale)
    blurred = cv2.GaussianBlur(
        luma, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101
    )
    detail = luma - blurred
    detail = np.where(
        np.abs(detail) >= settings.threshold / 100.0, detail, np.float32(0.0)
    )
    sharpened_luma = luma + detail * np.float32(settings.amount / 100.0)
    delta = sharpened_luma - luma
    return np.asarray(image + delta[..., None], dtype=np.float32)


def apply_effect_chain(
    image: NDArray[np.floating],
    effects: Sequence[EffectInstance],
    *,
    resolve_lut: LutResolver | None = None,
    spatial_scale: float = 1.0,
) -> NDArray[np.float32]:
    """Apply an ordered static effect chain to normalized Rec.709 RGB."""
    output = np.asarray(image, dtype=np.float32)
    if output.ndim != 3 or output.shape[-1] != 3 or not np.isfinite(output).all():
        raise ValueError("effect input must be a finite HxWx3 RGB image")
    if not np.isfinite(spatial_scale) or spatial_scale <= 0:
        raise ValueError("spatial_scale must be a finite positive value")
    output = output.copy()
    for effect in effects:
        if not effect.enabled or effect.mix <= 0:
            continue
        source = output
        if isinstance(effect, PrimaryCorrectionEffect):
            processed = _primary(source, effect)
        elif isinstance(effect, Lut3DEffect):
            if resolve_lut is None:
                raise ValueError("a managed LUT resolver is required")
            processed = apply_tetrahedral(source, resolve_lut(effect.parameters.asset_id))
        elif isinstance(effect, BloomEffect):
            processed = _bloom(source, effect, spatial_scale)
        elif isinstance(effect, VignetteEffect):
            processed = _vignette(source, effect)
        elif isinstance(effect, SharpenEffect):
            processed = _sharpen(source, effect, spatial_scale)
        else:  # pragma: no cover - the discriminated model closes this union
            raise ValueError("unsupported post-process effect")
        mix = np.float32(effect.mix / 100.0)
        output = np.asarray(source * (1.0 - mix) + processed * mix, dtype=np.float32)
        if not np.isfinite(output).all():
            raise ValueError("post-process effect produced non-finite pixels")
    return output
