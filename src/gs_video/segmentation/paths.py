from __future__ import annotations

from pathlib import Path, PureWindowsPath


def _is_reparse_leaf(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        return bool(path.lstat().st_file_attributes & 0x400)
    except (AttributeError, OSError):
        return False


def has_reparse_component(path: Path) -> bool:
    absolute = path if path.is_absolute() else path.absolute()
    parts = absolute.parts
    if not parts:
        return False
    current = Path(parts[0])
    if _is_reparse_leaf(current):
        return True
    for part in parts[1:]:
        current /= part
        if _is_reparse_leaf(current):
            return True
    return False


def is_wsl_prefix(prefix: tuple[str, ...]) -> bool:
    return bool(prefix) and PureWindowsPath(prefix[0]).name.lower() in {"wsl", "wsl.exe"} and "--" in prefix


def worker_path(path: Path, prefix: tuple[str, ...]) -> str:
    if not is_wsl_prefix(prefix):
        return str(path.resolve())
    original = PureWindowsPath(str(path))
    if not original.is_absolute() or not original.drive or original.drive.startswith("\\"):
        raise ValueError("WSL worker 仅支持 Windows 盘符绝对路径")
    resolved = path.resolve()
    windows = PureWindowsPath(str(resolved))
    if not windows.is_absolute() or not windows.drive or windows.drive.startswith("\\"):
        raise ValueError("WSL worker 仅支持 Windows 盘符绝对路径")
    drive = windows.drive.rstrip(":").lower()
    if len(drive) != 1 or not drive.isalpha():
        raise ValueError("WSL worker 不支持 UNC 或相对路径")
    tail = "/".join(windows.parts[1:])
    return f"/mnt/{drive}/{tail}"
