from __future__ import annotations

import numpy as np
import pytest

from gs_video.domain.models import (
    BloomEffect,
    PrimaryCorrectionEffect,
    PrimaryCorrectionParameters,
    SharpenEffect,
    VignetteEffect,
)
from gs_video.postprocess.color import linear_to_rec709, rec709_to_linear
from gs_video.postprocess.engine import apply_effect_chain
from gs_video.postprocess.lut import apply_tetrahedral, parse_cube


IDENTITY_CUBE = """
TITLE "identity"
LUT_3D_SIZE 2
DOMAIN_MIN 0 0 0
DOMAIN_MAX 1 1 1
0 0 0
1 0 0
0 1 0
1 1 0
0 0 1
1 0 1
0 1 1
1 1 1
"""


def test_rec709_round_trip_preserves_extended_finite_values() -> None:
    encoded = np.asarray([-0.1, 0.0, 0.05, 0.5, 1.0, 1.5], dtype=np.float32)
    restored = linear_to_rec709(rec709_to_linear(encoded))
    np.testing.assert_allclose(restored, encoded, atol=2e-6)


def test_identity_cube_uses_tetrahedral_interpolation() -> None:
    lut = parse_cube(IDENTITY_CUBE)
    image = np.asarray([[[0.2, 0.4, 0.8], [1.5, -0.5, 0.5]]], dtype=np.float32)
    output = apply_tetrahedral(image, lut)
    np.testing.assert_allclose(output[0, 0], image[0, 0], atol=1e-6)
    np.testing.assert_allclose(output[0, 1], [1.0, 0.0, 0.5], atol=1e-6)


@pytest.mark.parametrize(
    "text",
    [
        "LUT_1D_SIZE 2\n0 0 0\n1 1 1",
        "LUT_3D_SIZE 1\n0 0 0",
        "LUT_3D_SIZE 2\n0 0 nan",
        "LUT_3D_SIZE 2\n0 0 0",
        "LUT_3D_SIZE 2\nDOMAIN_MIN 1 0 0\nDOMAIN_MAX 0 1 1\n" + "0 0 0\n" * 8,
    ],
)
def test_cube_parser_rejects_unsupported_or_incomplete_content(text: str) -> None:
    with pytest.raises(ValueError):
        parse_cube(text)


def test_empty_or_disabled_chain_is_exact_bypass() -> None:
    image = np.random.default_rng(3).random((8, 9, 3), dtype=np.float32) * 1.4 - 0.2
    disabled = BloomEffect(enabled=False)
    np.testing.assert_array_equal(apply_effect_chain(image, []), image)
    np.testing.assert_array_equal(apply_effect_chain(image, [disabled]), image)


def test_universal_mix_controls_whole_effect_contribution() -> None:
    image = np.full((4, 4, 3), 0.25, dtype=np.float32)
    full = PrimaryCorrectionEffect(
        parameters=PrimaryCorrectionParameters(exposure=1)
    )
    half = full.model_copy(update={"mix": 50.0, "instance_id": full.instance_id})
    full_output = apply_effect_chain(image, [full])
    half_output = apply_effect_chain(image, [half])
    np.testing.assert_allclose(half_output, (image + full_output) * 0.5, atol=1e-6)


def test_effect_catalog_changes_pixels_and_preserves_float32() -> None:
    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[16, 16] = 1.0
    effects = [BloomEffect(), VignetteEffect(), SharpenEffect()]
    output = apply_effect_chain(image, effects)
    assert output.dtype == np.float32
    assert not np.array_equal(output, image)
    assert np.isfinite(output).all()
