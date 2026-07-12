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
MAX_SUPPORTED_REST_PROPERTY_INDEX = 44


@dataclass(frozen=True)
class GaussianScene:
    """Renderer-neutral Gaussian tensors with raw log-scales and opacity logits."""

    means: Float32Array
    scales: Float32Array
    quats: Float32Array
    opacities: Float32Array
    colors: Float32Array

    def __post_init__(self) -> None:
        arrays = {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "colors": self.colors,
        }
        expected_shapes = {
            "means": (2, 3, "[N,3]"),
            "scales": (2, 3, "[N,3]"),
            "quats": (2, 4, "[N,4]"),
            "opacities": (1, None, "[N]"),
            "colors": (3, 3, "[N,K,3]"),
        }
        for name, array in arrays.items():
            if not isinstance(array, np.ndarray):
                raise ValueError(f"{name} must be a NumPy array")
            expected_rank, final_dimension, shape_description = expected_shapes[name]
            if array.ndim != expected_rank or (
                final_dimension is not None and array.shape[-1] != final_dimension
            ):
                raise ValueError(f"{name} must have shape {shape_description}")

        counts = {array.shape[0] for array in arrays.values()}
        if len(counts) != 1 or self.means.shape[0] == 0:
            raise ValueError("GaussianScene arrays must share the same nonzero N")
        if self.colors.shape[1] not in SUPPORTED_SH_COEFFICIENT_COUNTS:
            raise ValueError(
                "colors K must be one of the supported SH coefficient counts: 1, 4, 9, 16"
            )

        for name, array in arrays.items():
            if array.dtype != np.float32:
                raise ValueError(f"{name} must use float32 dtype")
            if not array.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")

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
    aliases: dict[int, list[str]] = {}
    for name in sorted(property_names):
        if not name.startswith("f_rest_"):
            continue
        suffix = name.removeprefix("f_rest_")
        if not suffix.isdecimal():
            raise UnsupportedMaterialError(f"Gaussian PLY 球谐属性名称无效: {name}")
        aliases.setdefault(int(suffix), []).append(name)

    if not aliases:
        return ()

    for index in sorted(aliases):
        names = sorted(aliases[index], key=lambda name: (len(name), name))
        if len(names) > 1:
            raise UnsupportedMaterialError(
                f"Gaussian PLY 球谐属性存在重复索引 {index}: {', '.join(names)}"
            )
        suffix = names[0].removeprefix("f_rest_")
        if suffix != str(index):
            raise UnsupportedMaterialError(
                f"Gaussian PLY 球谐属性 {names[0]} 不是规范名称 f_rest_{index}"
            )
        if index > MAX_SUPPORTED_REST_PROPERTY_INDEX:
            raise UnsupportedMaterialError(
                f"Gaussian PLY 球谐属性 {names[0]} 超过最大支持索引 "
                f"{MAX_SUPPORTED_REST_PROPERTY_INDEX}"
            )

    indexed = {index: names[0] for index, names in aliases.items()}
    sorted_indices = sorted(indexed)
    for expected_index, actual_index in enumerate(sorted_indices):
        if actual_index != expected_index:
            raise UnsupportedMaterialError(
                f"Gaussian PLY 球谐属性不连续，缺少: f_rest_{expected_index}"
            )
    return tuple(indexed[index] for index in sorted_indices)


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
    total_bytes = gaussian_bytes + framebuffer_bytes + projection_bytes
    return (total_bytes * 3) // 2


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
