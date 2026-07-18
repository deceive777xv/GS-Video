from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from gs_video.segmentation.paths import has_reparse_component


PUBLISHABLE_CATEGORIES = frozenset(
    {
        "frames",
        "proxies",
        "masks",
        "camera",
        "trajectories",
        "renders",
        "composites",
        "previews",
        "exports",
    }
)
_HEX = frozenset("0123456789abcdef")
PathIdentity = tuple[int, int, int | None]
_DIRECTORY_MODE = 0o777 if os.name == "nt" else 0o700


def validate_cache_key(cache_key: str) -> str:
    if len(cache_key) != 64 or any(character not in _HEX for character in cache_key):
        raise ValueError("artifact cache key must be 64 lowercase hexadecimal characters")
    return cache_key


def _identity(metadata: os.stat_result) -> PathIdentity:
    device = int(metadata.st_dev)
    inode = int(metadata.st_ino)
    return (device, inode, None if inode else int(metadata.st_ctime_ns))


def _ordinary_directory(path: Path) -> PathIdentity:
    metadata = path.lstat()
    if has_reparse_component(path) or not stat.S_ISDIR(metadata.st_mode):
        raise OSError("artifact directory is not an owned ordinary directory")
    return _identity(metadata)


def _ordinary_file(path: Path) -> PathIdentity:
    metadata = path.lstat()
    if (
        has_reparse_component(path)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise OSError("artifact member is not an owned single-link regular file")
    return _identity(metadata)


def _flush_windows_directory(path: Path) -> None:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    handle = create_file(
        str(path),
        0x40000000,
        0x1 | 0x2 | 0x4,
        None,
        3,
        0x02000000,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle in {None, invalid}:
        raise ctypes.WinError(ctypes.get_last_error())
    flush = kernel32.FlushFileBuffers
    flush.argtypes = [ctypes.c_void_p]
    flush.restype = ctypes.c_int
    close = kernel32.CloseHandle
    close.argtypes = [ctypes.c_void_p]
    close.restype = ctypes.c_int
    try:
        if not flush(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        if not close(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        _flush_windows_directory(path)
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validated_inventory(root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    root_identity = _ordinary_directory(root)
    directories: list[Path] = [root]
    files: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for member in directory.iterdir():
            metadata = member.lstat()
            if has_reparse_component(member):
                raise OSError("artifact inventory contains a link or reparse point")
            if stat.S_ISDIR(metadata.st_mode):
                _identity(metadata)
                directories.append(member)
                pending.append(member)
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                _identity(metadata)
                files.append(member)
            else:
                raise OSError("artifact inventory contains a non-regular member")
    if not files:
        raise OSError("artifact inventory must contain at least one regular file")
    if _ordinary_directory(root) != root_identity:
        raise OSError("artifact directory identity changed during inventory")
    return tuple(sorted(files)), tuple(
        sorted(directories, key=lambda value: len(value.parts), reverse=True)
    )


def _persist_inventory(root: Path) -> None:
    files, directories = _validated_inventory(root)
    for path in files:
        before = path.stat()
        expected = _identity(before)
        with path.open("r+b") as stream:
            opened = os.fstat(stream.fileno())
            if (
                _identity(opened) != expected
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
            ):
                raise OSError("artifact file identity changed before fsync")
            os.fsync(stream.fileno())
            after = os.fstat(stream.fileno())
        if _identity(after) != expected or _ordinary_file(path) != expected:
            raise OSError("artifact file identity changed during fsync")
    for directory in directories:
        _fsync_directory(directory)
    _validated_inventory(root)


def _remove_owned_tree(path: Path, expected_identity: PathIdentity) -> None:
    try:
        if _ordinary_directory(path) != expected_identity:
            return
    except OSError:
        return
    try:
        members = tuple(path.iterdir())
    except OSError:
        return
    for member in members:
        try:
            metadata = member.lstat()
            if stat.S_ISDIR(metadata.st_mode) and not has_reparse_component(member):
                _remove_owned_tree(member, _identity(metadata))
            else:
                member.unlink()
        except OSError:
            return
    try:
        if _ordinary_directory(path) == expected_identity:
            path.rmdir()
    except OSError:
        return


class ArtifactPublisher:
    """Publish immutable cache-keyed directory trees through owned staging."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).absolute()
        self._root_identity = _ordinary_directory(self.root)

    def publish_tree(
        self,
        category: str,
        cache_key: str,
        build: Callable[[Path], object],
    ) -> Path:
        if category not in PUBLISHABLE_CATEGORIES:
            raise ValueError("artifact category is not publishable")
        validate_cache_key(cache_key)
        if _ordinary_directory(self.root) != self._root_identity:
            raise OSError("artifact root identity changed")

        category_root = self.root / category
        try:
            category_root.mkdir(mode=_DIRECTORY_MODE, exist_ok=False)
        except FileExistsError:
            pass
        category_identity = _ordinary_directory(category_root)
        destination = category_root / cache_key
        if destination.exists() or destination.is_symlink():
            _validated_inventory(destination)
            return destination

        staging = category_root / f".staging-{uuid4().hex}"
        staging.mkdir(mode=_DIRECTORY_MODE, exist_ok=False)
        staging_identity = _ordinary_directory(staging)
        try:
            build(staging)
            if _ordinary_directory(staging) != staging_identity:
                raise OSError("artifact staging identity changed")
            _persist_inventory(staging)
            if _ordinary_directory(self.root) != self._root_identity:
                raise OSError("artifact root identity changed before publication")
            if _ordinary_directory(category_root) != category_identity:
                raise OSError("artifact category identity changed before publication")
            if destination.exists() or destination.is_symlink():
                _validated_inventory(destination)
                return destination
            try:
                os.rename(staging, destination)
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                _validated_inventory(destination)
                return destination
            _validated_inventory(destination)
            _fsync_directory(category_root)
            return destination
        finally:
            _remove_owned_tree(staging, staging_identity)
