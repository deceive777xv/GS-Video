from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from plyfile import PlyData, PlyParseError  # type: ignore[import-untyped]

from gs_video.domain.errors import UnsupportedMaterialError


Float32Array = npt.NDArray[np.float32]

REQUIRED_PROPERTIES = {
    "x",
    "y",
    "z",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "opacity",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
}
SUPPORTED_SH_COEFFICIENT_COUNTS = {1, 4, 9, 16}


@dataclass(frozen=True)
class GaussianScene:
    """Renderer-neutral Gaussian tensors with raw log-scales and opacity logits."""

    means: Float32Array
    scales: Float32Array
    quats: Float32Array
    opacities: Float32Array
    colors: Float32Array

    @property
    def count(self) -> int:
        return int(self.means.shape[0])


@dataclass(frozen=True)
class VramAssessment:
    """Admission decision against the conservative 80% available-VRAM threshold."""

    estimated_bytes: int
    accepted: bool
    recommendation: str | None


def _columns(vertex: np.ndarray, names: tuple[str, ...]) -> Float32Array:
    return np.ascontiguousarray(np.column_stack([vertex[name] for name in names]), dtype=np.float32)


def _rest_property_names(property_names: set[str]) -> tuple[str, ...]:
    indexed: dict[int, str] = {}
    for name in property_names:
        if not name.startswith("f_rest_"):
            continue
        suffix = name.removeprefix("f_rest_")
        if not suffix.isdecimal():
            raise UnsupportedMaterialError(f"Gaussian PLY 球谐属性名称无效: {name}")
        indexed[int(suffix)] = name

    if not indexed:
        return ()

    expected_indices = set(range(max(indexed) + 1))
    missing = sorted(expected_indices - indexed.keys())
    if missing:
        missing_names = ", ".join(f"f_rest_{index}" for index in missing)
        raise UnsupportedMaterialError(f"Gaussian PLY 球谐属性不连续，缺少: {missing_names}")
    return tuple(indexed[index] for index in range(len(indexed)))


def load_gaussian_ply(path: Path | str) -> GaussianScene:
    """Load a Graphdeco Gaussian PLY into contiguous float32 domain arrays.

    Graphdeco stores non-DC spherical-harmonic coefficients channel-major. They are
    reshaped from ``[N, 3, K-1]`` to the public ``[N, K, 3]`` layout. Log-scales and
    opacity logits remain raw; renderer adapters apply exp/sigmoid at their boundary.
    """

    try:
        ply_data = PlyData.read(path)
        vertex_element = ply_data["vertex"]
    except (KeyError, OSError, PlyParseError, ValueError) as exc:
        raise UnsupportedMaterialError("Gaussian PLY 缺少或无法读取 vertex 元素") from exc

    vertex = vertex_element.data
    if len(vertex) == 0:
        raise UnsupportedMaterialError("Gaussian PLY vertex 元素为空")

    property_names = set(vertex.dtype.names or ())
    missing = sorted(REQUIRED_PROPERTIES - property_names)
    if missing:
        raise UnsupportedMaterialError(f"Gaussian PLY 缺少必需属性: {', '.join(missing)}")

    rest_names = _rest_property_names(property_names)
    for name in sorted(REQUIRED_PROPERTIES | set(rest_names)):
        field_dtype = vertex.dtype.fields[name][0]
        if field_dtype.shape != () or field_dtype.kind not in "iuf":
            raise UnsupportedMaterialError(
                f"Gaussian PLY 属性 {name} 必须是数值标量"
            )
    if len(rest_names) % 3 != 0:
        raise UnsupportedMaterialError("Gaussian PLY 球谐 f_rest 属性数量必须能被 3 整除")
    coefficient_count = 1 + len(rest_names) // 3
    if coefficient_count not in SUPPORTED_SH_COEFFICIENT_COUNTS:
        raise UnsupportedMaterialError(
            f"Gaussian PLY 球谐系数数量 {coefficient_count} 不是受支持的平方基"
        )

    means = _columns(vertex, ("x", "y", "z"))
    scales = _columns(vertex, ("scale_0", "scale_1", "scale_2"))
    quats = _columns(vertex, ("rot_0", "rot_1", "rot_2", "rot_3"))
    opacities = np.ascontiguousarray(vertex["opacity"], dtype=np.float32)
    dc = _columns(vertex, ("f_dc_0", "f_dc_1", "f_dc_2"))[:, np.newaxis, :]

    if rest_names:
        rest = _columns(vertex, rest_names)
        rest = rest.reshape(len(vertex), 3, coefficient_count - 1).transpose(0, 2, 1)
        colors = np.ascontiguousarray(np.concatenate((dc, rest), axis=1), dtype=np.float32)
    else:
        colors = np.ascontiguousarray(dc, dtype=np.float32)

    arrays = (means, scales, quats, opacities, colors)
    if not all(np.isfinite(array).all() for array in arrays):
        raise UnsupportedMaterialError("Gaussian PLY 包含非有限数值")

    return GaussianScene(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        colors=colors,
    )


def estimate_scene_vram(scene: GaussianScene, width: int, height: int) -> int:
    """Return the exact conservative byte estimate for scene rendering."""

    if width <= 0 or height <= 0:
        raise ValueError("render dimensions must be positive")
    gaussian_bytes = sum(
        array.nbytes
        for array in (scene.means, scene.scales, scene.quats, scene.opacities, scene.colors)
    )
    framebuffer_bytes = width * height * 4 * 4
    projection_bytes = scene.count * 48
    return int((gaussian_bytes + framebuffer_bytes + projection_bytes) * 1.5)


def assess_scene_vram(
    scene: GaussianScene,
    width: int,
    height: int,
    available_vram_bytes: int,
) -> VramAssessment:
    """Accept only estimates at or below 80% of available VRAM."""

    if available_vram_bytes <= 0:
        raise ValueError("available_vram_bytes must be positive")
    estimated = estimate_scene_vram(scene, width, height)
    accepted = estimated * 5 <= available_vram_bytes * 4
    recommendation = None
    if not accepted:
        recommendation = "downsample the scene or reduce resolution"
    return VramAssessment(
        estimated_bytes=estimated,
        accepted=accepted,
        recommendation=recommendation,
    )
