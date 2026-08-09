import json
import os
from pathlib import Path

import pytest

from gs_video.environment.vram import (
    MINIMUM_VRAM_MB,
    STANDARD_VRAM_MB,
    VramBudgetManager,
    VramBudgetMode,
    VramBudgetPersistenceError,
    VramBudgetUnavailableError,
    load_user_vram_limit,
)


def test_missing_preference_uses_standard_budget_bounded_by_physical_total(
    tmp_path: Path,
) -> None:
    manager = VramBudgetManager(
        tmp_path / "user-settings.json",
        total_vram_probe=lambda: 24_576,
        initial_limit_mb=STANDARD_VRAM_MB,
    )

    snapshot = manager.snapshot()

    assert snapshot.mode is VramBudgetMode.STANDARD
    assert snapshot.minimum_vram_mb == MINIMUM_VRAM_MB
    assert snapshot.total_vram_mb == 24_576
    assert snapshot.selected_vram_mb == STANDARD_VRAM_MB
    assert snapshot.editable is True
    assert snapshot.recovered_from_invalid_preference is False

    smaller = VramBudgetManager(
        tmp_path / "other-settings.json",
        total_vram_probe=lambda: 6_144,
        initial_limit_mb=STANDARD_VRAM_MB,
    )
    assert smaller.current_limit_mb() == 6_144


def test_custom_budget_persists_and_restores_exact_physical_total(tmp_path: Path) -> None:
    path = tmp_path / "user-settings.json"
    manager = VramBudgetManager(
        path,
        total_vram_probe=lambda: 24_321,
        initial_limit_mb=STANDARD_VRAM_MB,
    )

    updated = manager.update(VramBudgetMode.CUSTOM, 24_321)

    assert updated.mode is VramBudgetMode.CUSTOM
    assert updated.selected_vram_mb == 24_321
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "vram_budget": {
            "mode": "custom",
            "selected_vram_mb": 24_321,
        },
    }

    restored = VramBudgetManager(
        path,
        total_vram_probe=lambda: 24_321,
        initial_limit_mb=STANDARD_VRAM_MB,
    )
    assert restored.snapshot() == updated


@pytest.mark.parametrize("value", [True, 1023, 24_577, 2048.0, "2048"])
def test_custom_budget_rejects_non_integer_or_out_of_range_values(
    tmp_path: Path, value: object
) -> None:
    manager = VramBudgetManager(
        tmp_path / "user-settings.json",
        total_vram_probe=lambda: 24_576,
        initial_limit_mb=STANDARD_VRAM_MB,
    )

    with pytest.raises(ValueError, match="between 1024 and 24576"):
        manager.update(VramBudgetMode.CUSTOM, value)  # type: ignore[arg-type]


def test_budget_is_read_only_when_physical_total_is_unavailable(tmp_path: Path) -> None:
    manager = VramBudgetManager(
        tmp_path / "user-settings.json",
        total_vram_probe=lambda: 0,
        initial_limit_mb=STANDARD_VRAM_MB,
    )

    snapshot = manager.snapshot()

    assert snapshot.total_vram_mb == 0
    assert snapshot.editable is False
    assert snapshot.blocked_reason == "gpu_unavailable"
    with pytest.raises(VramBudgetUnavailableError, match="physical VRAM"):
        manager.update(VramBudgetMode.STANDARD, None)


def test_invalid_preference_recovers_without_overwriting_the_file(tmp_path: Path) -> None:
    path = tmp_path / "user-settings.json"
    invalid = '{"mode":"custom","selected_vram_mb":999999}'
    path.write_text(invalid, encoding="utf-8")
    manager = VramBudgetManager(
        path,
        total_vram_probe=lambda: 12_288,
        initial_limit_mb=STANDARD_VRAM_MB,
    )

    snapshot = manager.snapshot()

    assert snapshot.mode is VramBudgetMode.STANDARD
    assert snapshot.selected_vram_mb == STANDARD_VRAM_MB
    assert snapshot.recovered_from_invalid_preference is True
    assert path.read_text(encoding="utf-8") == invalid


def test_persistence_failure_keeps_the_previous_authoritative_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "user-settings.json"
    manager = VramBudgetManager(
        path,
        total_vram_probe=lambda: 16_384,
        initial_limit_mb=STANDARD_VRAM_MB,
    )
    before = manager.snapshot()

    def fail_replace(_source: Path | str, _target: Path | str) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(VramBudgetPersistenceError, match="save VRAM preference"):
        manager.update(VramBudgetMode.CUSTOM, 12_288)

    assert manager.snapshot() == before
    assert path.exists() is False


def test_prepare_helper_reads_only_valid_machine_preference(tmp_path: Path) -> None:
    path = tmp_path / "user-settings.json"
    assert load_user_vram_limit(path) == STANDARD_VRAM_MB

    path.write_text(
        json.dumps({"mode": "custom", "selected_vram_mb": 16_384}),
        encoding="utf-8",
    )
    assert load_user_vram_limit(path) == 16_384

    path.write_text("{}", encoding="utf-8")
    assert load_user_vram_limit(path) == STANDARD_VRAM_MB
