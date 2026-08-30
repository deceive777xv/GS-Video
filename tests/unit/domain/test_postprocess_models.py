from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from gs_video.domain.models import (
    BloomEffect,
    Lut3DEffect,
    Lut3DParameters,
    PrimaryCorrectionEffect,
    Project,
    SharpenEffect,
    VignetteEffect,
)


def test_effect_chain_accepts_duplicates_and_preserves_order() -> None:
    first = BloomEffect()
    second = BloomEffect(display_name="Second bloom", mix=25)
    project = Project(name="effects")
    project.workflow.effect_chain = [first, second, SharpenEffect()]

    restored = Project.model_validate_json(project.model_dump_json())

    assert [effect.instance_id for effect in restored.workflow.effect_chain] == [
        first.instance_id,
        second.instance_id,
        project.workflow.effect_chain[2].instance_id,
    ]
    assert restored.workflow.effect_chain[1].display_name == "Second bloom"
    assert restored.workflow.effect_chain[1].mix == 25


def test_effect_catalog_defaults_match_product_contract() -> None:
    primary = PrimaryCorrectionEffect().parameters
    bloom = BloomEffect().parameters
    vignette = VignetteEffect().parameters
    sharpen = SharpenEffect().parameters

    assert primary.exposure == 0
    assert primary.saturation == 100
    assert bloom.threshold == 80
    assert bloom.radius == 32
    assert vignette.amount == 20
    assert sharpen.amount == 50
    assert sharpen.radius == 1


def test_lut_effect_requires_managed_asset_id() -> None:
    effect = Lut3DEffect(parameters=Lut3DParameters(asset_id=str(uuid4())))
    assert effect.parameters.asset_id

    with pytest.raises(ValidationError):
        Lut3DEffect(parameters=Lut3DParameters(asset_id=""))


def test_effect_chain_rejects_more_than_32_instances() -> None:
    with pytest.raises(ValidationError, match="32"):
        Project.model_validate(
            {
                "name": "too-many",
                "workflow": {
                    "effect_chain": [BloomEffect().model_dump() for _ in range(33)]
                },
            }
        )


@pytest.mark.parametrize(
    "effect",
    [
        {**BloomEffect().model_dump(), "mix": 101},
        {
            **SharpenEffect().model_dump(),
            "parameters": {"amount": 50, "radius": 0, "threshold": 1},
        },
        {
            **PrimaryCorrectionEffect().model_dump(),
            "parameters": {
                **PrimaryCorrectionEffect().parameters.model_dump(),
                "exposure": float("nan"),
            },
        },
    ],
)
def test_effects_reject_out_of_range_or_nonfinite_values(effect: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Project.model_validate({"name": "invalid", "workflow": {"effect_chain": [effect]}})
