from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from gs_video.segmentation.paths import has_reparse_component


MAX_USER_SETTINGS_BYTES = 64 * 1024


def read_user_settings(path: Path) -> tuple[dict[str, Any], bool]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {}, False
    except OSError:
        return {}, True
    if (
        has_reparse_component(path)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
        or metadata.st_size > MAX_USER_SETTINGS_BYTES
    ):
        return {}, True
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}, True
    return (value, False) if isinstance(value, dict) else ({}, True)


def write_user_settings(path: Path, settings: dict[str, Any]) -> None:
    parent = path.parent
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        parent.mkdir(parents=True, exist_ok=True)
        parent_stat = parent.lstat()
        if (
            has_reparse_component(parent)
            or not stat.S_ISDIR(parent_stat.st_mode)
            or (path.exists() and has_reparse_component(path))
            or (temporary.exists() and has_reparse_component(temporary))
        ):
            raise OSError("user settings path is unsafe")
        serialized = json.dumps(settings, ensure_ascii=False, indent=2) + "\n"
        temporary.write_text(serialized, encoding="utf-8", newline="\n")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
