from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

import gs_video.scene.ply as ply_module
from gs_video.domain.errors import UnsupportedMaterialError
from gs_video.scene.ply import (
    GaussianScene,
    assess_scene_vram,
    estimate_scene_vram,
    load_gaussian_ply,
)


FIXTURE = Path(__file__).parents[2] / "fixtures" / "scene" / "tiny_gaussians.ply"
BASE_PROPERTIES = (
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
)


def _write_ply(
    path: Path,
    properties: tuple[str, ...],
    rows: list[tuple[float, ...]],
    *,
    text: bool = True,
) -> None:
    data = np.array(rows, dtype=[(name, "f4") for name in properties])
    PlyData([PlyElement.describe(data, "vertex")], text=text).write(path)


def _valid_row(*rest: float) -> tuple[float, ...]:
    return (
        1.0,
        2.0,
        3.0,
        -1.0,
        -2.0,
        -3.0,
        1.0,
        0.0,
        0.0,
        0.0,
        -0.5,
        0.1,
        0.2,
        0.3,
        *rest,
    )


def _valid_scene_arrays() -> dict[str, np.ndarray]:
    return {
        "means": np.zeros((2, 3), dtype=np.float32),
        "scales": np.zeros((2, 3), dtype=np.float32),
        "quats": np.zeros((2, 4), dtype=np.float32),
        "opacities": np.zeros(2, dtype=np.float32),
        "colors": np.zeros((2, 4, 3), dtype=np.float32),
    }


def test_rejects_plain_xyz_point_cloud_and_lists_all_missing_properties(tmp_path: Path) -> None:
    path = tmp_path / "plain.ply"
    _write_ply(path, ("x", "y", "z"), [(0.0, 0.0, 0.0)])

    with pytest.raises(UnsupportedMaterialError) as exc_info:
        load_gaussian_ply(path)

    message = str(exc_info.value)
    for name in sorted(set(BASE_PROPERTIES) - {"x", "y", "z"}):
        assert name in message


@pytest.mark.parametrize("text", [True, False], ids=["ascii", "binary"])
def test_loads_ascii_and_binary_graphdeco_ply(tmp_path: Path, text: bool) -> None:
    path = tmp_path / "gaussians.ply"
    rest_names = tuple(f"f_rest_{index}" for index in range(9))
    rest = tuple(float(index + 1) for index in range(9))
    _write_ply(path, BASE_PROPERTIES + rest_names, [_valid_row(*rest)], text=text)

    scene = load_gaussian_ply(path)

    assert scene.count == 1
    assert scene.means.shape == (1, 3)
    assert scene.scales.shape == (1, 3)
    assert scene.quats.shape == (1, 4)
    assert scene.opacities.shape == (1,)
    assert scene.colors.shape == (1, 4, 3)
    np.testing.assert_allclose(
        scene.colors[0],
        [[0.1, 0.2, 0.3], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]],
    )


def test_committed_fixture_preserves_raw_values_and_float32_layout() -> None:
    scene = load_gaussian_ply(FIXTURE)

    assert scene.count == 2
    np.testing.assert_allclose(scene.scales, [[-1.0, -2.0, -3.0], [0.4, 0.5, 0.6]])
    assert scene.opacities.tolist() == pytest.approx([-0.5, 1.25])
    assert scene.colors.shape == (2, 4, 3)
    for array in (
        scene.means,
        scene.scales,
        scene.quats,
        scene.opacities,
        scene.colors,
    ):
        assert array.dtype == np.float32
        assert array.flags.c_contiguous


def test_rejects_discontinuous_rest_coefficients(tmp_path: Path) -> None:
    path = tmp_path / "gapped.ply"
    _write_ply(path, BASE_PROPERTIES + ("f_rest_0", "f_rest_2"), [_valid_row(1.0, 2.0)])

    with pytest.raises(UnsupportedMaterialError, match="f_rest_1"):
        load_gaussian_ply(path)


def test_rejects_noncanonical_zero_padded_rest_suffix(tmp_path: Path) -> None:
    path = tmp_path / "zero-padded.ply"
    _write_ply(path, BASE_PROPERTIES + ("f_rest_00",), [_valid_row(1.0)])

    with pytest.raises(UnsupportedMaterialError, match="f_rest_00.*规范"):
        load_gaussian_ply(path)


def test_reports_duplicate_numeric_rest_aliases_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-alias.ply"
    _write_ply(
        path,
        BASE_PROPERTIES + ("f_rest_0", "f_rest_00"),
        [_valid_row(1.0, 2.0)],
    )

    with pytest.raises(
        UnsupportedMaterialError,
        match="重复.*f_rest_0, f_rest_00",
    ):
        load_gaussian_ply(path)


def test_rejects_huge_rest_index_without_range_sized_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "huge-rest-index.ply"
    _write_ply(
        path,
        BASE_PROPERTIES + ("f_rest_1000000000",),
        [_valid_row(1.0)],
    )

    def forbid_untrusted_range(*args: int) -> range:
        raise AssertionError(f"must not allocate range from untrusted index: {args}")

    monkeypatch.setattr(ply_module, "range", forbid_untrusted_range, raising=False)

    with pytest.raises(UnsupportedMaterialError, match="f_rest_1000000000.*44"):
        load_gaussian_ply(path)


@pytest.mark.parametrize("rest_count", [1, 6])
def test_rejects_malformed_spherical_harmonic_shapes(
    tmp_path: Path, rest_count: int
) -> None:
    path = tmp_path / "malformed-sh.ply"
    rest_names = tuple(f"f_rest_{index}" for index in range(rest_count))
    _write_ply(path, BASE_PROPERTIES + rest_names, [_valid_row(*([1.0] * rest_count))])

    with pytest.raises(UnsupportedMaterialError, match="球谐"):
        load_gaussian_ply(path)


def test_rejects_empty_vertex_element(tmp_path: Path) -> None:
    path = tmp_path / "empty.ply"
    _write_ply(path, BASE_PROPERTIES, [])

    with pytest.raises(UnsupportedMaterialError, match="vertex.*为空"):
        load_gaussian_ply(path)


def test_rejects_missing_vertex_element(tmp_path: Path) -> None:
    path = tmp_path / "missing-vertex.ply"
    data = np.array([(1.0,)], dtype=[("value", "f4")])
    PlyData([PlyElement.describe(data, "face")], text=True).write(path)

    with pytest.raises(UnsupportedMaterialError, match="vertex"):
        load_gaussian_ply(path)


def test_rejects_non_finite_gaussian_values(tmp_path: Path) -> None:
    path = tmp_path / "non-finite.ply"
    row = list(_valid_row())
    row[0] = float("nan")
    _write_ply(path, BASE_PROPERTIES, [tuple(row)])

    with pytest.raises(UnsupportedMaterialError, match="有限"):
        load_gaussian_ply(path)


def test_rejects_non_scalar_gaussian_property(tmp_path: Path) -> None:
    path = tmp_path / "list-opacity.ply"
    dtype = [
        (name, "O" if name == "opacity" else "f4")
        for name in BASE_PROPERTIES
    ]
    row = list(_valid_row())
    row[10] = np.array([-0.5], dtype=np.float32)
    data = np.array([tuple(row)], dtype=dtype)
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)

    with pytest.raises(UnsupportedMaterialError, match="opacity.*数值标量"):
        load_gaussian_ply(path)


def test_maps_malformed_ply_to_stable_domain_error(tmp_path: Path) -> None:
    path = tmp_path / "malformed.ply"
    path.write_text("not a ply", encoding="ascii")

    with pytest.raises(UnsupportedMaterialError, match="PLY"):
        load_gaussian_ply(path)


def test_estimate_scene_vram_uses_exact_documented_formula() -> None:
    scene = GaussianScene(
        means=np.zeros((2, 3), dtype=np.float32),
        scales=np.zeros((2, 3), dtype=np.float32),
        quats=np.zeros((2, 4), dtype=np.float32),
        opacities=np.zeros(2, dtype=np.float32),
        colors=np.zeros((2, 4, 3), dtype=np.float32),
    )
    gaussian_bytes = 24 + 24 + 32 + 8 + 96
    framebuffer_bytes = 10 * 5 * 4 * 4
    projection_bytes = 2 * 48

    total_bytes = gaussian_bytes + framebuffer_bytes + projection_bytes
    assert estimate_scene_vram(scene, 10, 5) == (total_bytes * 3) // 2


def test_estimate_scene_vram_uses_exact_integer_safety_factor_for_large_values() -> None:
    scene = GaussianScene(**_valid_scene_arrays())
    width = 2**52 + 1
    total_bytes = (
        sum(array.nbytes for array in _valid_scene_arrays().values())
        + width * 4 * 4
        + scene.count * 48
    )

    assert estimate_scene_vram(scene, width, 1) == (total_bytes * 3) // 2


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("means", np.zeros((2, 3, 1), dtype=np.float32)),
        ("scales", np.zeros((2, 2), dtype=np.float32)),
        ("quats", np.zeros((2, 3), dtype=np.float32)),
        ("opacities", np.zeros((2, 1), dtype=np.float32)),
        ("colors", np.zeros((2, 4, 4), dtype=np.float32)),
    ],
)
def test_gaussian_scene_rejects_wrong_rank_or_final_dimensions(
    field: str, replacement: np.ndarray
) -> None:
    arrays = _valid_scene_arrays()
    arrays[field] = replacement

    with pytest.raises(ValueError, match=field):
        GaussianScene(**arrays)


def test_gaussian_scene_rejects_mismatched_gaussian_counts() -> None:
    arrays = _valid_scene_arrays()
    arrays["scales"] = np.zeros((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="same nonzero N"):
        GaussianScene(**arrays)


def test_gaussian_scene_rejects_non_float32_arrays() -> None:
    arrays = _valid_scene_arrays()
    arrays["means"] = np.zeros((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="means.*float32"):
        GaussianScene(**arrays)


def test_gaussian_scene_rejects_non_contiguous_arrays() -> None:
    arrays = _valid_scene_arrays()
    arrays["means"] = np.zeros((3, 2), dtype=np.float32).T
    assert not arrays["means"].flags.c_contiguous

    with pytest.raises(ValueError, match="means.*C-contiguous"):
        GaussianScene(**arrays)


def test_gaussian_scene_rejects_empty_gaussian_dimension() -> None:
    arrays = {
        "means": np.zeros((0, 3), dtype=np.float32),
        "scales": np.zeros((0, 3), dtype=np.float32),
        "quats": np.zeros((0, 4), dtype=np.float32),
        "opacities": np.zeros(0, dtype=np.float32),
        "colors": np.zeros((0, 4, 3), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="nonzero N"):
        GaussianScene(**arrays)


@pytest.mark.parametrize("coefficient_count", [0, 2])
def test_gaussian_scene_rejects_empty_or_unsupported_sh_basis(
    coefficient_count: int,
) -> None:
    arrays = _valid_scene_arrays()
    arrays["colors"] = np.zeros((2, coefficient_count, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="colors.*K"):
        GaussianScene(**arrays)


def test_vram_assessment_accepts_exact_eighty_percent_boundary() -> None:
    scene = load_gaussian_ply(FIXTURE)
    estimated = estimate_scene_vram(scene, 10, 5)

    assessment = assess_scene_vram(scene, 10, 5, available_vram_bytes=estimated * 5 // 4)

    assert assessment.accepted is True
    assert assessment.estimated_bytes == estimated
    assert assessment.recommendation is None


def test_vram_assessment_rejects_above_eighty_percent_with_actionable_advice() -> None:
    scene = load_gaussian_ply(FIXTURE)
    estimated = estimate_scene_vram(scene, 10, 5)

    assessment = assess_scene_vram(scene, 10, 5, available_vram_bytes=estimated)

    assert assessment.accepted is False
    assert "downsample" in assessment.recommendation
    assert "reduce resolution" in assessment.recommendation
