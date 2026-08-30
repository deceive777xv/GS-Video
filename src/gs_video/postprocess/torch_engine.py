from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as functional

from gs_video.domain.models import (
    BloomEffect,
    EffectInstance,
    Lut3DEffect,
    PrimaryCorrectionEffect,
    SharpenEffect,
    VignetteEffect,
)
from gs_video.postprocess.lut import CubeLut


LUMA = (0.2126, 0.7152, 0.0722)
TorchLutResolver = Callable[[str], CubeLut]


def _rec709_to_linear(values: torch.Tensor) -> torch.Tensor:
    return torch.where(
        values < 0.081,
        values / 4.5,
        torch.clamp((values + 0.099) / 1.099, min=0.0).pow(1.0 / 0.45),
    )


def _linear_to_rec709(values: torch.Tensor) -> torch.Tensor:
    return torch.where(
        values < 0.018,
        values * 4.5,
        1.099 * torch.clamp(values, min=0.0).pow(0.45) - 0.099,
    )


def _luma(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor(LUMA).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = max(float(sigma), 0.1)
    radius = max(1, ceil(3.0 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    channels = image.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    padded = functional.pad(image, (radius, radius, 0, 0), mode="replicate")
    blurred = functional.conv2d(padded, horizontal, groups=channels)
    padded = functional.pad(blurred, (0, 0, radius, radius), mode="replicate")
    return functional.conv2d(padded, vertical, groups=channels)


def _primary(image: torch.Tensor, effect: PrimaryCorrectionEffect) -> torch.Tensor:
    settings = effect.parameters
    corrected = _linear_to_rec709(_rec709_to_linear(image) * (2.0**settings.exposure))
    corrected = (corrected - 0.5) * (1.0 + settings.contrast / 100.0) + 0.5
    luma = _luma(corrected)
    if settings.highlights:
        corrected = corrected + torch.clamp((luma - 0.5) * 2.0, 0.0, 1.0) * (
            settings.highlights / 200.0
        )
    if settings.shadows:
        corrected = corrected + torch.clamp((0.5 - luma) * 2.0, 0.0, 1.0) * (
            settings.shadows / 200.0
        )
    temperature = settings.temperature / 100.0 * 0.15
    tint = settings.tint / 100.0 * 0.10
    balance = corrected.new_tensor(
        (1.0 + temperature, 1.0 + tint, 1.0 - temperature)
    ).view(1, 3, 1, 1)
    corrected = corrected * balance
    luma = _luma(corrected)
    corrected = luma + (corrected - luma) * (settings.saturation / 100.0)
    if settings.vibrance:
        chroma = corrected.amax(dim=1, keepdim=True) - corrected.amin(
            dim=1, keepdim=True
        )
        room = torch.clamp(1.0 - chroma, 0.0, 1.0)
        corrected = luma + (corrected - luma) * (
            1.0 + settings.vibrance / 100.0 * room
        )
    return corrected


def _tetrahedral(image: torch.Tensor, lut: CubeLut) -> torch.Tensor:
    # Work in HWC for compact advanced indexing; return NCHW.
    rgb = image[0].permute(1, 2, 0)
    low = rgb.new_tensor(lut.domain_min)
    high = rgb.new_tensor(lut.domain_max)
    position = torch.clamp((rgb - low) / (high - low), 0.0, 1.0) * (lut.size - 1)
    base = torch.floor(position).to(torch.long).clamp(max=lut.size - 2)
    fraction = position - base
    r, g, b = (base[..., channel] for channel in range(3))
    fr, fg, fb = (fraction[..., channel : channel + 1] for channel in range(3))
    table = torch.as_tensor(lut.values, device=image.device, dtype=image.dtype)
    c000 = table[r, g, b]
    c100 = table[r + 1, g, b]
    c010 = table[r, g + 1, b]
    c001 = table[r, g, b + 1]
    c110 = table[r + 1, g + 1, b]
    c101 = table[r + 1, g, b + 1]
    c011 = table[r, g + 1, b + 1]
    c111 = table[r + 1, g + 1, b + 1]
    values = (
        c000 + fr * (c100 - c000) + fg * (c110 - c100) + fb * (c111 - c110),
        c000 + fr * (c100 - c000) + fb * (c101 - c100) + fg * (c111 - c101),
        c000 + fb * (c001 - c000) + fr * (c101 - c001) + fg * (c111 - c101),
        c000 + fg * (c010 - c000) + fr * (c110 - c010) + fb * (c111 - c110),
        c000 + fg * (c010 - c000) + fb * (c011 - c010) + fr * (c111 - c011),
        c000 + fb * (c001 - c000) + fg * (c011 - c001) + fr * (c111 - c011),
    )
    masks = (
        (fr >= fg) & (fg >= fb),
        (fr >= fg) & (fr >= fb) & (fb > fg),
        (fb > fr) & (fr >= fg),
        (fg > fr) & (fr >= fb),
        (fg > fr) & (fg >= fb) & (fb > fr),
        (fb > fg) & (fg > fr),
    )
    output = torch.zeros_like(rgb)
    for mask, value in zip(masks, values, strict=True):
        output = torch.where(mask, value, output)
    return output.permute(2, 0, 1).unsqueeze(0)


def _bloom(image: torch.Tensor, effect: BloomEffect, spatial_scale: float) -> torch.Tensor:
    settings = effect.parameters
    luma = _luma(image)
    threshold = settings.threshold / 100.0
    knee = max(settings.soft_knee / 100.0 * threshold, 1e-6)
    soft = torch.clamp((luma - threshold + knee) / (2.0 * knee), 0.0, 1.0)
    contribution = torch.clamp(luma - threshold, min=0.0) + soft.square() * knee * 0.5
    highlights = image * contribution / torch.clamp(luma, min=1e-6)
    blurred = _gaussian_blur(highlights, settings.radius * spatial_scale / 3.0)
    tint = image.new_tensor(settings.tint).view(1, 3, 1, 1)
    return image + blurred * tint * (settings.intensity / 100.0)


def _vignette(
    image: torch.Tensor,
    effect: VignetteEffect,
    *,
    origin: tuple[int, int],
    full_size: tuple[int, int],
) -> torch.Tensor:
    settings = effect.parameters
    height, width = image.shape[-2:]
    full_width, full_height = full_size
    y = torch.arange(origin[1], origin[1] + height, device=image.device, dtype=image.dtype)
    x = torch.arange(origin[0], origin[0] + width, device=image.device, dtype=image.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = (xx + 0.5) / full_width * 2.0 - 1.0 - settings.center_x / 100.0
    yy = (yy + 0.5) / full_height * 2.0 - 1.0 - settings.center_y / 100.0
    aspect = full_width / max(full_height, 1)
    roundness = settings.roundness / 100.0
    distance = torch.sqrt(
        (xx * aspect ** (-roundness * 0.5)).square()
        + (yy * aspect ** (roundness * 0.5)).square()
    )
    transition = torch.clamp(
        (distance - settings.midpoint / 100.0) / max(settings.feather / 100.0, 1e-4),
        0.0,
        1.0,
    )
    transition = transition.square() * (3.0 - 2.0 * transition)
    return image * (1.0 - transition.view(1, 1, height, width) * settings.amount / 100.0)


def _sharpen(image: torch.Tensor, effect: SharpenEffect, spatial_scale: float) -> torch.Tensor:
    settings = effect.parameters
    luma = _luma(image)
    detail = luma - _gaussian_blur(luma, settings.radius * spatial_scale)
    detail = torch.where(
        detail.abs() >= settings.threshold / 100.0, detail, torch.zeros_like(detail)
    )
    return image + detail * (settings.amount / 100.0)


def apply_effect_chain_cuda(
    image: torch.Tensor,
    effects: Sequence[EffectInstance],
    *,
    resolve_lut: TorchLutResolver,
    spatial_scale: float,
    origin: tuple[int, int],
    full_size: tuple[int, int],
) -> torch.Tensor:
    output = image
    for effect in effects:
        if not effect.enabled or effect.mix <= 0:
            continue
        source = output
        if isinstance(effect, PrimaryCorrectionEffect):
            processed = _primary(source, effect)
        elif isinstance(effect, Lut3DEffect):
            processed = _tetrahedral(source, resolve_lut(effect.parameters.asset_id))
        elif isinstance(effect, BloomEffect):
            processed = _bloom(source, effect, spatial_scale)
        elif isinstance(effect, VignetteEffect):
            processed = _vignette(
                source, effect, origin=origin, full_size=full_size
            )
        elif isinstance(effect, SharpenEffect):
            processed = _sharpen(source, effect, spatial_scale)
        else:  # pragma: no cover
            raise ValueError("unsupported CUDA post-process effect")
        mix = effect.mix / 100.0
        output = source * (1.0 - mix) + processed * mix
        if not torch.isfinite(output).all():
            raise ValueError("CUDA post-process effect produced non-finite pixels")
    return output


def required_overlap(effects: Sequence[EffectInstance], spatial_scale: float) -> int:
    overlap = 0
    for effect in effects:
        if not effect.enabled or effect.mix <= 0:
            continue
        if isinstance(effect, BloomEffect):
            overlap = max(overlap, ceil(effect.parameters.radius * spatial_scale))
        elif isinstance(effect, SharpenEffect):
            overlap = max(overlap, ceil(3.0 * effect.parameters.radius * spatial_scale))
    return overlap


def process_rgb16_cuda(
    image: NDArray[np.uint16],
    effects: Sequence[EffectInstance],
    *,
    resolve_lut: TorchLutResolver,
    spatial_scale: float,
    vram_limit_mb: int,
) -> NDArray[np.uint16]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    height, width = image.shape[:2]
    budget_bytes = int(vram_limit_mb * 1024 * 1024 * 0.65)
    estimated = height * width * 3 * 4 * 20
    overlap = required_overlap(effects, spatial_scale)
    raw_core = int((budget_bytes / (3 * 4 * 20)) ** 0.5) - overlap * 2
    if raw_core < 64:
        raise RuntimeError("minimum safe CUDA tile exceeds the VRAM budget")
    core = raw_core
    if estimated <= budget_bytes:
        core = max(height, width)
    output = np.empty_like(image)
    with torch.inference_mode():
        for y0 in range(0, height, core):
            for x0 in range(0, width, core):
                y1 = min(height, y0 + core)
                x1 = min(width, x0 + core)
                ey0 = max(0, y0 - overlap)
                ex0 = max(0, x0 - overlap)
                ey1 = min(height, y1 + overlap)
                ex1 = min(width, x1 + overlap)
                tile = torch.from_numpy(
                    image[ey0:ey1, ex0:ex1].astype(np.float32) / 65535.0
                ).to("cuda")
                tile = tile.permute(2, 0, 1).unsqueeze(0)
                processed = apply_effect_chain_cuda(
                    tile,
                    effects,
                    resolve_lut=resolve_lut,
                    spatial_scale=spatial_scale,
                    origin=(ex0, ey0),
                    full_size=(width, height),
                )
                processed = processed[0].permute(1, 2, 0)
                crop = processed[y0 - ey0 : y1 - ey0, x0 - ex0 : x1 - ex0]
                pixels = (
                    torch.clamp(crop, 0.0, 1.0)
                    .mul(65535.0)
                    .round()
                    .to(torch.int32)
                    .cpu()
                    .numpy()
                    .astype(np.uint16)
                )
                output[y0:y1, x0:x1] = pixels
                del tile, processed, crop
    return output
