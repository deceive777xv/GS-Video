from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gs_video.segmentation.paths import has_reparse_component

MINIMUM_VRAM_MB = 1024
STANDARD_VRAM_MB = 8192
_MAX_PREFERENCE_BYTES = 16 * 1024
VramLimitProvider = Callable[[], int]


class VramBudgetMode(StrEnum):
    STANDARD = "standard"
    CUSTOM = "custom"


class VramBudgetPreference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

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
    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return None, False
    except OSError:
        return None, True
    if (
        not path.is_file()
        or has_reparse_component(path)
        or stat_result.st_size <= 0
        or stat_result.st_size > _MAX_PREFERENCE_BYTES
    ):
        return None, True
    try:
        payload = path.read_bytes()
        return VramBudgetPreference.model_validate_json(payload), False
    except (OSError, ValidationError, ValueError):
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
        parent = self._path.parent
        temporary = self._path.with_name(f"{self._path.name}.tmp")
        try:
            parent.mkdir(parents=True, exist_ok=True)
            if has_reparse_component(parent) or (
                self._path.exists() and has_reparse_component(self._path)
            ) or (
                temporary.exists() and has_reparse_component(temporary)
            ):
                raise OSError("preference path contains a link or reparse point")
            serialized = json.dumps(
                preference.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            ) + "\n"
            temporary.write_text(serialized, encoding="utf-8", newline="\n")
            os.replace(temporary, self._path)
        except OSError as error:
            raise VramBudgetPersistenceError("failed to save VRAM preference") from error
        finally:
            temporary.unlink(missing_ok=True)

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
