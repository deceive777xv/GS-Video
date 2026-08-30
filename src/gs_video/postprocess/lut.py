from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gs_video.domain.models import LutSummary


MAX_LUT_SIZE = 65


@dataclass(frozen=True)
class CubeLut:
    size: int
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    values: NDArray[np.float32]


def _three_floats(tokens: list[str], label: str) -> tuple[float, float, float]:
    if len(tokens) != 4:
        raise ValueError(f"{label} requires exactly three values")
    try:
        values = tuple(float(value) for value in tokens[1:])
    except ValueError as error:
        raise ValueError(f"{label} contains a non-numeric value") from error
    if len(values) != 3 or not all(isfinite(value) for value in values):
        raise ValueError(f"{label} must contain three finite values")
    return values


def parse_cube(text: str) -> CubeLut:
    """Parse the supported 3D subset of the Adobe .cube text format."""
    if not text or "\x00" in text:
        raise ValueError("LUT file is empty or contains NUL bytes")
    size: int | None = None
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    samples: list[tuple[float, float, float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        directive = tokens[0].upper()
        if directive == "TITLE":
            continue
        if directive == "LUT_1D_SIZE":
            raise ValueError("1D LUTs are not supported")
        if directive == "LUT_3D_SIZE":
            if size is not None or len(tokens) != 2:
                raise ValueError("LUT_3D_SIZE must appear once with one integer")
            try:
                size = int(tokens[1])
            except ValueError as error:
                raise ValueError("LUT_3D_SIZE must be an integer") from error
            if not 2 <= size <= MAX_LUT_SIZE:
                raise ValueError(f"LUT_3D_SIZE must be between 2 and {MAX_LUT_SIZE}")
            continue
        if directive == "DOMAIN_MIN":
            domain_min = _three_floats(tokens, "DOMAIN_MIN")
            continue
        if directive == "DOMAIN_MAX":
            domain_max = _three_floats(tokens, "DOMAIN_MAX")
            continue
        if size is None:
            raise ValueError(f"LUT sample appears before LUT_3D_SIZE on line {line_number}")
        if len(tokens) != 3:
            raise ValueError(f"invalid LUT sample on line {line_number}")
        try:
            sample = tuple(float(value) for value in tokens)
        except ValueError as error:
            raise ValueError(f"invalid LUT sample on line {line_number}") from error
        if len(sample) != 3 or not all(isfinite(value) for value in sample):
            raise ValueError(f"LUT sample on line {line_number} must be finite RGB")
        samples.append(sample)
    if size is None:
        raise ValueError("LUT_3D_SIZE is required")
    if any(high <= low for low, high in zip(domain_min, domain_max, strict=True)):
        raise ValueError("DOMAIN_MAX must be greater than DOMAIN_MIN on every channel")
    expected = size**3
    if len(samples) != expected:
        raise ValueError(f"3D LUT requires exactly {expected} samples")
    # .cube ordering changes red fastest, then green, then blue. Store RGB axes.
    values = np.asarray(samples, dtype=np.float32).reshape(size, size, size, 3)
    values = np.ascontiguousarray(values.transpose(2, 1, 0, 3))
    return CubeLut(size, domain_min, domain_max, values)


def inspect_cube(path: Path, size: int, sha256: str) -> LutSummary:
    if path.suffix and path.suffix.lower() != ".cube":
        raise ValueError("3D LUT assets must use the .cube extension")
    if size <= 0 or size > 8 * 1024 * 1024:
        raise ValueError("3D LUT assets must be between 1 byte and 8 MiB")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise ValueError("3D LUT asset must be valid UTF-8 text") from error
    lut = parse_cube(text)
    return LutSummary(
        filename=path.name,
        size=size,
        sha256=sha256,
        lut_size=lut.size,
        domain_min=lut.domain_min,
        domain_max=lut.domain_max,
    )


def apply_tetrahedral(
    image: NDArray[np.floating], lut: CubeLut
) -> NDArray[np.float32]:
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("LUT input must be a finite HxWx3 image")
    low = np.asarray(lut.domain_min, dtype=np.float32)
    high = np.asarray(lut.domain_max, dtype=np.float32)
    position = np.clip((rgb - low) / (high - low), 0.0, 1.0) * (lut.size - 1)
    base = np.floor(position).astype(np.int32)
    base = np.minimum(base, lut.size - 2)
    fraction = position - base
    r, g, b = (base[..., channel] for channel in range(3))
    fr, fg, fb = (fraction[..., channel, None] for channel in range(3))
    table = lut.values
    c000 = table[r, g, b]
    c100 = table[r + 1, g, b]
    c010 = table[r, g + 1, b]
    c001 = table[r, g, b + 1]
    c110 = table[r + 1, g + 1, b]
    c101 = table[r + 1, g, b + 1]
    c011 = table[r, g + 1, b + 1]
    c111 = table[r + 1, g + 1, b + 1]

    cases = (
        (
            (fr >= fg) & (fg >= fb),
            c000 + fr * (c100 - c000) + fg * (c110 - c100) + fb * (c111 - c110),
        ),
        (
            (fr >= fg) & (fr >= fb) & (fb > fg),
            c000 + fr * (c100 - c000) + fb * (c101 - c100) + fg * (c111 - c101),
        ),
        (
            (fb > fr) & (fr >= fg),
            c000 + fb * (c001 - c000) + fr * (c101 - c001) + fg * (c111 - c101),
        ),
        (
            (fg > fr) & (fr >= fb),
            c000 + fg * (c010 - c000) + fr * (c110 - c010) + fb * (c111 - c110),
        ),
        (
            (fg > fr) & (fg >= fb) & (fb > fr),
            c000 + fg * (c010 - c000) + fb * (c011 - c010) + fr * (c111 - c011),
        ),
        (
            (fb > fg) & (fg > fr),
            c000 + fb * (c001 - c000) + fg * (c011 - c001) + fr * (c111 - c011),
        ),
    )
    output = np.empty_like(rgb, dtype=np.float32)
    assigned = np.zeros(rgb.shape[:2] + (1,), dtype=bool)
    for mask, values in cases:
        np.copyto(output, values, where=mask)
        assigned |= mask
    if not assigned.all():
        raise RuntimeError("tetrahedral interpolation left pixels unassigned")
    return output
