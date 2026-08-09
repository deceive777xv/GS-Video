from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gs_video.storage.settings import read_user_settings, write_user_settings

MINIMUM_VRAM_MB = 1024
STANDARD_VRAM_MB = 8192
VramLimitProvider = Callable[[], int]


class VramBudgetMode(StrEnum):
    STANDARD = "standard"
    CUSTOM = "custom"


class VramBudgetPreference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: VramBudgetMode
    selected_vram_mb: int = Field(strict=True, ge=MINIMUM_VRAM_MB)


class VramBudgetSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    mode: VramBudgetMode
    minimum_vram_mb: int = MINIMUM_VRAM_MB
    total_vram_mb: int = Field(ge=0)
    selected_vram_mb: int = Field(ge=MINIMUM_VRAM_MB)
    editable: bool
    blocked_reason: str | None = None
    recovered_from_invalid_preference: bool = False


class VramBudgetUnavailableError(RuntimeError):
    pass


class VramBudgetPersistenceError(RuntimeError):
    pass


def validated_vram_limit_mb(value: object) -> int:
    if type(value) is not int or value < MINIMUM_VRAM_MB:
        raise ValueError("VRAM limit must be an integer of at least 1024 MiB")
    return value


def resolve_vram_limit_mb(
    fallback: int,
    provider: VramLimitProvider | None,
) -> int:
    return validated_vram_limit_mb(provider() if provider is not None else fallback)


def _read_preference(path: Path) -> tuple[VramBudgetPreference | None, bool]:
    settings, invalid = read_user_settings(path)
    if invalid:
        return None, True
    if not settings:
        return None, False
    if "vram_budget" in settings:
        value = settings["vram_budget"]
    elif "mode" in settings or "selected_vram_mb" in settings:
        value = settings
    else:
        return None, False
    try:
        return VramBudgetPreference.model_validate(value), False
    except (ValidationError, ValueError):
        return None, True


def load_user_vram_limit(path: Path) -> int:
    preference, invalid = _read_preference(path)
    if invalid or preference is None or preference.mode is VramBudgetMode.STANDARD:
        return STANDARD_VRAM_MB
    return preference.selected_vram_mb


class VramBudgetManager:
    def __init__(
        self,
        preference_path: Path,
        *,
        total_vram_probe: Callable[[], int],
        initial_limit_mb: int = STANDARD_VRAM_MB,
    ) -> None:
        validated_vram_limit_mb(initial_limit_mb)
        self._path = Path(preference_path)
        self._total_vram_probe = total_vram_probe
        self._lock = RLock()
        preference, invalid = _read_preference(self._path)
        self._preference = preference or VramBudgetPreference(
            mode=(
                VramBudgetMode.STANDARD
                if initial_limit_mb == STANDARD_VRAM_MB
                else VramBudgetMode.CUSTOM
            ),
            selected_vram_mb=initial_limit_mb,
        )
        self._invalid_preference = invalid

    def _total_vram_mb(self) -> int:
        try:
            value = self._total_vram_probe()
        except (OSError, RuntimeError, ValueError):
            return 0
        return value if type(value) is int and value > 0 else 0

    def _snapshot_for_total(
        self,
        total_vram_mb: int,
        *,
        editable: bool | None = None,
        blocked_reason: str | None = None,
    ) -> VramBudgetSnapshot:
        physical_available = total_vram_mb >= MINIMUM_VRAM_MB
        preference = self._preference
        invalid = self._invalid_preference
        if preference.mode is VramBudgetMode.STANDARD:
            selected = (
                min(STANDARD_VRAM_MB, total_vram_mb)
                if physical_available
                else STANDARD_VRAM_MB
            )
            mode = VramBudgetMode.STANDARD
        elif physical_available and preference.selected_vram_mb <= total_vram_mb:
            selected = preference.selected_vram_mb
            mode = VramBudgetMode.CUSTOM
        else:
            selected = (
                min(STANDARD_VRAM_MB, total_vram_mb)
                if physical_available
                else STANDARD_VRAM_MB
            )
            mode = VramBudgetMode.STANDARD
            invalid = True
        can_edit = physical_available if editable is None else editable and physical_available
        reason = blocked_reason
        if not physical_available:
            reason = "gpu_unavailable"
        return VramBudgetSnapshot(
            mode=mode,
            total_vram_mb=total_vram_mb,
            selected_vram_mb=selected,
            editable=can_edit,
            blocked_reason=None if can_edit else reason,
            recovered_from_invalid_preference=invalid,
        )

    def snapshot(
        self,
        *,
        editable: bool | None = None,
        blocked_reason: str | None = None,
    ) -> VramBudgetSnapshot:
        with self._lock:
            return self._snapshot_for_total(
                self._total_vram_mb(),
                editable=editable,
                blocked_reason=blocked_reason,
            )

    def current_limit_mb(self) -> int:
        return self.snapshot().selected_vram_mb

    def _save(self, preference: VramBudgetPreference) -> None:
        try:
            settings, invalid = read_user_settings(self._path)
            if invalid:
                settings = {}
            settings.pop("mode", None)
            settings.pop("selected_vram_mb", None)
            settings["schema_version"] = 1
            settings["vram_budget"] = preference.model_dump(mode="json")
            write_user_settings(self._path, settings)
        except OSError as error:
            raise VramBudgetPersistenceError("failed to save VRAM preference") from error

    def update(
        self,
        mode: VramBudgetMode,
        selected_vram_mb: int | None,
    ) -> VramBudgetSnapshot:
        with self._lock:
            total_vram_mb = self._total_vram_mb()
            if total_vram_mb < MINIMUM_VRAM_MB:
                raise VramBudgetUnavailableError("physical VRAM is unavailable")
            if not isinstance(mode, VramBudgetMode):
                raise ValueError("mode must be standard or custom")
            if mode is VramBudgetMode.STANDARD:
                selected = min(STANDARD_VRAM_MB, total_vram_mb)
            else:
                if (
                    type(selected_vram_mb) is not int
                    or not MINIMUM_VRAM_MB <= selected_vram_mb <= total_vram_mb
                ):
                    raise ValueError(
                        "selected_vram_mb must be an integer between "
                        f"{MINIMUM_VRAM_MB} and {total_vram_mb}"
                    )
                selected = selected_vram_mb
            preference = VramBudgetPreference(
                mode=mode,
                selected_vram_mb=selected,
            )
            self._save(preference)
            self._preference = preference
            self._invalid_preference = False
            return self._snapshot_for_total(total_vram_mb)


__all__ = [
    "MINIMUM_VRAM_MB",
    "STANDARD_VRAM_MB",
    "VramBudgetManager",
    "VramBudgetMode",
    "VramBudgetPersistenceError",
    "VramBudgetPreference",
    "VramBudgetSnapshot",
    "VramBudgetUnavailableError",
    "VramLimitProvider",
    "load_user_vram_limit",
    "resolve_vram_limit_mb",
    "validated_vram_limit_mb",
]
