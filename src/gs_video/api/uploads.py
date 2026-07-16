from __future__ import annotations

import errno
import hashlib
import os
import stat
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import BinaryIO
from uuid import uuid4

from starlette.requests import Request

from gs_video.api.schemas import (
    ApiError,
    UploadComplete,
    UploadCreateRequest,
    UploadCreated,
    UploadStatus,
)


CHUNK_LIMIT = 1024 * 1024
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_IS_WINDOWS = os.name == "nt"


class _UnsafePathError(RuntimeError):
    pass


@dataclass(frozen=True)
class _PathIdentity:
    device: int
    inode: int
    fallback_created_ns: int | None = None


def _is_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return stat.S_ISLNK(metadata.st_mode) or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _identity_from_stat(metadata: os.stat_result) -> _PathIdentity:
    inode = int(metadata.st_ino)
    return _PathIdentity(
        device=int(metadata.st_dev),
        inode=inode,
        fallback_created_ns=None if inode else int(metadata.st_ctime_ns),
    )


def _read_identity(
    path: Path, *, directory: bool, single_link_file: bool = False
) -> _PathIdentity:
    metadata = path.lstat()
    if _is_reparse(metadata):
        raise _UnsafePathError("reparse points are not allowed")
    if directory and not stat.S_ISDIR(metadata.st_mode):
        raise _UnsafePathError("expected a directory")
    if not directory and not stat.S_ISREG(metadata.st_mode):
        raise _UnsafePathError("expected a regular file")
    if not directory and single_link_file and metadata.st_nlink != 1:
        raise _UnsafePathError("hard-linked upload files are not allowed")
    return _identity_from_stat(metadata)


def _open_windows_directory(path: Path) -> int:
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
        0x80 | 0x00010000,
        0x1 | 0x2,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle in {None, invalid}:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(handle)


def _close_windows_handle(handle: int) -> None:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.CloseHandle(ctypes.c_void_p(handle)):
        raise ctypes.WinError(ctypes.get_last_error())


class _DirectoryLease:
    def __init__(
        self,
        path: Path,
        identity: _PathIdentity,
        *,
        descriptor: int | None = None,
        windows_handle: int | None = None,
    ) -> None:
        self.path = path
        self.identity = identity
        self._descriptor = descriptor
        self._windows_handle = windows_handle
        self._closed = False

    @classmethod
    def acquire(
        cls, path: Path, expected: _PathIdentity | None = None
    ) -> _DirectoryLease:
        before = _read_identity(path, directory=True)
        if expected is not None and before != expected:
            raise _UnsafePathError("directory identity changed before lease")
        if _IS_WINDOWS:
            handle = _open_windows_directory(path)
            try:
                if _read_identity(path, directory=True) != before:
                    raise _UnsafePathError("directory changed during lease")
            except BaseException:
                _close_windows_handle(handle)
                raise
            return cls(path, before, windows_handle=handle)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            if _identity_from_stat(os.fstat(descriptor)) != before:
                raise _UnsafePathError("directory changed during lease")
        except BaseException:
            os.close(descriptor)
            raise
        return cls(path, before, descriptor=descriptor)

    def _require_open(self) -> None:
        if self._closed:
            raise _UnsafePathError("directory lease is closed")

    @staticmethod
    def _name(name: str) -> str:
        if not name or Path(name).name != name or "/" in name or "\\" in name:
            raise _UnsafePathError("child name must be a basename")
        return name

    def mkdir_child(self, name: str) -> None:
        self._require_open()
        name = self._name(name)
        if self._descriptor is not None:
            os.mkdir(name, 0o700, dir_fd=self._descriptor)
        else:
            (self.path / name).mkdir(parents=False, exist_ok=False)

    def rmdir_child(self, name: str) -> None:
        self._require_open()
        name = self._name(name)
        if self._descriptor is not None:
            os.rmdir(name, dir_fd=self._descriptor)
        else:
            (self.path / name).rmdir()

    def child_identity(
        self,
        name: str,
        *,
        directory: bool = False,
        single_link_file: bool = True,
    ) -> _PathIdentity:
        self._require_open()
        name = self._name(name)
        if self._descriptor is not None:
            metadata = os.stat(
                name, dir_fd=self._descriptor, follow_symlinks=False
            )
            if _is_reparse(metadata):
                raise _UnsafePathError("reparse child is not allowed")
            if directory and not stat.S_ISDIR(metadata.st_mode):
                raise _UnsafePathError("expected child directory")
            if not directory and not stat.S_ISREG(metadata.st_mode):
                raise _UnsafePathError("expected regular child")
            if not directory and single_link_file and metadata.st_nlink != 1:
                raise _UnsafePathError("expected single-link regular child")
            return _identity_from_stat(metadata)
        return _read_identity(
            self.path / name,
            directory=directory,
            single_link_file=not directory and single_link_file,
        )

    def unlink_child(self, name: str) -> None:
        self._require_open()
        name = self._name(name)
        if self._descriptor is not None:
            os.unlink(name, dir_fd=self._descriptor)
        else:
            (self.path / name).unlink()

    def child_names(self) -> set[str]:
        self._require_open()
        if self._descriptor is not None:
            return set(os.listdir(self._descriptor))
        return {entry.name for entry in self.path.iterdir()}

    def open_exclusive(self, name: str) -> BinaryIO:
        self._require_open()
        name = self._name(name)
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        if self._descriptor is not None:
            descriptor = os.open(name, flags, 0o600, dir_fd=self._descriptor)
        else:
            descriptor = os.open(self.path / name, flags, 0o600)
        return os.fdopen(descriptor, "w+b")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        if self._windows_handle is not None:
            _close_windows_handle(self._windows_handle)
            self._windows_handle = None

    def delete_if_empty(self) -> None:
        self._require_open()
        if self.child_names():
            raise OSError(errno.ENOTEMPTY, "owned directory is not empty")
        if self._windows_handle is None:
            raise RuntimeError("handle-bound directory deletion requires Windows")
        _mark_windows_handle_delete(self._windows_handle)
        self.close()


def _open_windows_owned_file(path: Path, *, delete_on_close: bool) -> BinaryIO:
    import ctypes
    import msvcrt

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
    flags = 0x80 | 0x00200000
    if delete_on_close:
        flags |= 0x04000000
    handle = create_file(
        str(path),
        0x80000000 | 0x40000000 | 0x00010000,
        0,
        None,
        1,
        flags,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle in {None, invalid}:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(
            int(handle), os.O_RDWR | getattr(os, "O_BINARY", 0)
        )
    except BaseException:
        _close_windows_handle(int(handle))
        raise
    return os.fdopen(descriptor, "w+b")


def _mark_windows_handle_delete(handle: int) -> None:
    import ctypes

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("delete_file", ctypes.c_ubyte)]

    information = FileDispositionInfo(1)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    set_information.restype = ctypes.c_int
    if not set_information(
        ctypes.c_void_p(handle),
        4,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _mark_windows_delete(stream: BinaryIO) -> None:
    import msvcrt

    _mark_windows_handle_delete(msvcrt.get_osfhandle(stream.fileno()))


@dataclass
class _OwnedFile:
    lease: _DirectoryLease
    name: str
    stream: BinaryIO
    identity: _PathIdentity
    delete_on_close: bool
    linked: bool = True
    closed: bool = False

    @classmethod
    def create(
        cls,
        lease: _DirectoryLease,
        name: str,
        *,
        delete_on_close: bool,
    ) -> _OwnedFile:
        if _IS_WINDOWS:
            stream = _open_windows_owned_file(
                lease.path / name, delete_on_close=delete_on_close
            )
        else:
            stream = lease.open_exclusive(name)
        identity = _identity_from_stat(os.fstat(stream.fileno()))
        owned = cls(lease, name, stream, identity, delete_on_close)
        if not _IS_WINDOWS and delete_on_close:
            lease.unlink_child(name)
            owned.linked = False
        return owned

    def verify_handle(self) -> None:
        if self.closed or _identity_from_stat(os.fstat(self.stream.fileno())) != self.identity:
            raise _UnsafePathError("owned file handle identity changed")

    def verify_link(self) -> None:
        self.verify_handle()
        if self.linked and self.lease.child_identity(self.name) != self.identity:
            raise _UnsafePathError("owned file link identity changed")

    def delete(self) -> None:
        if self.closed:
            if self.linked:
                raise _UnsafePathError(
                    "owned file handle was released before deletion completed"
                )
            return
        self.verify_handle()
        if _IS_WINDOWS:
            if not self.delete_on_close:
                _mark_windows_delete(self.stream)
        elif self.linked:
            if (
                self.lease.child_identity(
                    self.name, single_link_file=False
                )
                != self.identity
            ):
                raise _UnsafePathError("owned file link changed before deletion")
            self.lease.unlink_child(self.name)
            self.linked = False
        self.stream.close()
        self.closed = True
        self.linked = False

    def release(self) -> None:
        if self.closed:
            return
        self.stream.close()
        self.closed = True

    def keep(self) -> None:
        if self.closed:
            return
        self.verify_link()
        self.stream.close()
        self.closed = True


class _UploadState(StrEnum):
    ACTIVE = "active"
    PREPARED = "prepared"
    CANCELLING = "cancelling"
    COMMITTED = "committed"
    CANCELLED = "cancelled"


@dataclass
class _UploadRecord:
    id: str
    kind: str
    filename: str
    mime_type: str
    total_size: int
    sha256: str
    directory: Path
    directory_identity: _PathIdentity
    directory_lease: _DirectoryLease | None
    spool: _OwnedFile
    destination: _OwnedFile
    completed: UploadComplete
    chunk_hashes: dict[int, str] = field(default_factory=dict)
    state: _UploadState = _UploadState.ACTIVE
    persisted: bool = False
    scratch_removed: bool = False
    destination_removed: bool = False
    lock: RLock = field(default_factory=RLock, repr=False)


def _write_spool(
    stream: BinaryIO, *, offset: int, content: bytes
) -> None:
    stream.seek(offset)
    stream.write(content)
    stream.flush()


def _copy_spool_to_destination(record: _UploadRecord) -> tuple[int, str]:
    record.spool.verify_handle()
    record.destination.verify_handle()
    record.spool.stream.seek(0)
    record.destination.stream.seek(0)
    record.destination.stream.truncate(0)
    digest = hashlib.sha256()
    size = 0
    remaining = record.total_size
    while remaining:
        block = record.spool.stream.read(min(CHUNK_LIMIT, remaining))
        if not block:
            break
        record.destination.stream.write(block)
        digest.update(block)
        size += len(block)
        remaining -= len(block)
    record.destination.stream.flush()
    os.fsync(record.destination.stream.fileno())
    return size, digest.hexdigest()


async def read_bounded_body(request: Request, limit: int = CHUNK_LIMIT) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as error:
            raise ApiError(
                400,
                code="invalid_content_length",
                category="validation",
                message="The chunk Content-Length is invalid.",
            ) from error
        if declared_length < 0:
            raise ApiError(
                400,
                code="invalid_content_length",
                category="validation",
                message="The chunk Content-Length is invalid.",
            )
        if declared_length > limit:
            raise ApiError(
                413,
                code="chunk_too_large",
                category="validation",
                message="The upload chunk exceeds the configured limit.",
            )
    body = bytearray()
    async for block in request.stream():
        if len(body) + len(block) > limit:
            raise ApiError(
                413,
                code="chunk_too_large",
                category="validation",
                message="The upload chunk exceeds the configured limit.",
            )
        body.extend(block)
    return bytes(body)


class UploadManager:
    def __init__(
        self, project_root: Path, max_upload_size: int, max_active_uploads: int
    ) -> None:
        if not _IS_WINDOWS:
            raise RuntimeError(
                "Browser upload storage is supported only on Windows 11 in this MVP."
            )
        self._root = project_root.resolve()
        self._upload_root = self._root / ".uploads"
        self._source_root = self._root / "source"
        self._upload_root.mkdir(parents=True, exist_ok=True)
        self._source_root.mkdir(parents=True, exist_ok=True)
        root_lease: _DirectoryLease | None = None
        upload_root_lease: _DirectoryLease | None = None
        source_root_lease: _DirectoryLease | None = None
        try:
            self._root_identity = _read_identity(self._root, directory=True)
            self._upload_root_identity = _read_identity(
                self._upload_root, directory=True
            )
            self._source_root_identity = _read_identity(
                self._source_root, directory=True
            )
            root_lease = _DirectoryLease.acquire(
                self._root, self._root_identity
            )
            upload_root_lease = _DirectoryLease.acquire(
                self._upload_root, self._upload_root_identity
            )
            source_root_lease = _DirectoryLease.acquire(
                self._source_root, self._source_root_identity
            )
        except (OSError, _UnsafePathError) as error:
            for lease in (source_root_lease, upload_root_lease, root_lease):
                if lease is not None:
                    try:
                        lease.close()
                    except OSError:
                        pass
            raise ValueError("project upload paths must be ordinary directories") from error
        assert root_lease is not None
        assert upload_root_lease is not None
        assert source_root_lease is not None
        self._root_lease = root_lease
        self._upload_root_lease = upload_root_lease
        self._source_root_lease = source_root_lease
        self._max_upload_size = max_upload_size
        self._max_active_uploads = max_active_uploads
        self._completed_limit = max(16, max_active_uploads * 2)
        self._records: dict[str, _UploadRecord] = {}
        self._completed: OrderedDict[str, UploadComplete] = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def _path_changed(message: str) -> ApiError:
        return ApiError(
            409,
            code="upload_path_changed",
            category="filesystem",
            message=message,
        )

    @staticmethod
    def _raise_storage(error: BaseException, message: str) -> None:
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            raise ApiError(
                507,
                code="storage_full",
                category="storage",
                message=message,
            ) from error
        if isinstance(error, _UnsafePathError):
            raise UploadManager._path_changed(
                "An upload path is no longer safely owned."
            ) from error
        raise ApiError(
            500,
            code="storage_error",
            category="storage",
            message="The upload storage operation failed.",
        ) from error

    def _require_path(
        self, path: Path, expected: _PathIdentity, *, directory: bool
    ) -> None:
        try:
            actual = _read_identity(path, directory=directory)
        except (OSError, _UnsafePathError) as error:
            raise self._path_changed("An upload path is no longer safely owned.") from error
        if actual != expected:
            raise self._path_changed("An upload path identity changed during the session.")

    def _require_roots(self) -> None:
        self._require_path(self._root, self._root_identity, directory=True)
        self._require_path(
            self._upload_root, self._upload_root_identity, directory=True
        )

    def _require_source(self) -> None:
        self._require_path(
            self._source_root, self._source_root_identity, directory=True
        )

    def _require_record(self, record: _UploadRecord) -> None:
        self._require_roots()
        self._require_path(
            record.directory, record.directory_identity, directory=True
        )
        if record.directory_lease is None:
            try:
                record.directory_lease = _DirectoryLease.acquire(
                    record.directory, record.directory_identity
                )
            except (OSError, _UnsafePathError) as error:
                raise self._path_changed(
                    "The upload directory lease could not be restored."
                ) from error

    def _delete_artifacts(self, record: _UploadRecord) -> None:
        first_error: OSError | _UnsafePathError | None = None
        try:
            if not record.destination.closed and not record.destination_removed:
                try:
                    record.destination.delete()
                    record.destination_removed = True
                except (OSError, _UnsafePathError) as error:
                    first_error = error
            if not record.spool.closed:
                try:
                    record.spool.delete()
                except (OSError, _UnsafePathError) as error:
                    if first_error is None:
                        first_error = error
        finally:
            try:
                self._source_root_identity = _read_identity(
                    self._source_root, directory=True
                )
            except (OSError, _UnsafePathError):
                pass
        if first_error is not None:
            raise first_error

    @staticmethod
    def _close_lease_quietly(lease: _DirectoryLease | None) -> None:
        if lease is None:
            return
        try:
            lease.close()
        except OSError:
            pass

    def _discard_scratch_lease(self, lease: _DirectoryLease | None) -> bool:
        if lease is None:
            return False
        try:
            if lease.child_names():
                return False
            lease.delete_if_empty()
            self._upload_root_identity = _read_identity(
                self._upload_root, directory=True
            )
            return True
        except (OSError, _UnsafePathError):
            return False
        finally:
            self._close_lease_quietly(lease)

    def _rollback_allocation(
        self,
        upload_id: str,
        directory_identity: _PathIdentity | None,
        lease: _DirectoryLease | None,
        spool: _OwnedFile | None,
        destination: _OwnedFile | None,
    ) -> None:
        for owned in (destination, spool):
            if owned is not None and not owned.closed:
                try:
                    owned.delete()
                except (OSError, _UnsafePathError):
                    try:
                        owned.release()
                    except OSError:
                        pass
        if lease is None and directory_identity is not None:
            try:
                lease = _DirectoryLease.acquire(
                    self._upload_root / upload_id, directory_identity
                )
            except (OSError, _UnsafePathError):
                lease = None
        self._discard_scratch_lease(lease)
        try:
            self._source_root_identity = _read_identity(
                self._source_root, directory=True
            )
        except (OSError, _UnsafePathError):
            pass

    def _abort_record(self, record: _UploadRecord) -> None:
        try:
            self._delete_artifacts(record)
        except (OSError, _UnsafePathError):
            record.state = _UploadState.CANCELLING
            return
        lease = record.directory_lease
        record.directory_lease = None
        record.scratch_removed = self._discard_scratch_lease(lease)
        record.state = _UploadState.CANCELLED
        with self._lock:
            self._records.pop(record.id, None)

    def create(self, request: UploadCreateRequest) -> UploadCreated:
        if request.total_size > self._max_upload_size:
            raise ApiError(
                413,
                code="upload_too_large",
                category="validation",
                message="The declared upload exceeds the configured size limit.",
            )
        with self._lock:
            if len(self._records) >= self._max_active_uploads:
                raise ApiError(
                    429,
                    code="upload_limit_reached",
                    category="upload",
                    message="The active upload limit has been reached.",
                    retryable=True,
                )
            self._require_roots()
            self._require_source()
            upload_id = uuid4().hex
            lease: _DirectoryLease | None = None
            spool: _OwnedFile | None = None
            destination: _OwnedFile | None = None
            directory_identity: _PathIdentity | None = None
            try:
                self._upload_root_lease.mkdir_child(upload_id)
                directory = self._upload_root / upload_id
                directory_identity = _read_identity(directory, directory=True)
                lease = _DirectoryLease.acquire(directory, directory_identity)
                spool = _OwnedFile.create(
                    lease, "payload.tmp", delete_on_close=True
                )
                self._require_path(
                    directory, directory_identity, directory=True
                )
                directory_identity = _read_identity(directory, directory=True)
                destination_name = f"{upload_id}-{request.filename}"
                destination = _OwnedFile.create(
                    self._source_root_lease,
                    destination_name,
                    delete_on_close=False,
                )
                self._require_source()
                self._upload_root_identity = _read_identity(
                    self._upload_root, directory=True
                )
                self._source_root_identity = _read_identity(
                    self._source_root, directory=True
                )
            except ApiError:
                self._rollback_allocation(
                    upload_id, directory_identity, lease, spool, destination
                )
                raise
            except (OSError, _UnsafePathError) as error:
                self._rollback_allocation(
                    upload_id, directory_identity, lease, spool, destination
                )
                self._raise_storage(error, "Upload storage could not be allocated.")
            assert lease is not None
            assert spool is not None
            assert destination is not None
            assert directory_identity is not None
            completed = UploadComplete(
                path=f"source/{destination.name}",
                kind=request.kind,
                size=request.total_size,
                sha256=request.sha256,
            )
            self._records[upload_id] = _UploadRecord(
                id=upload_id,
                kind=request.kind,
                filename=request.filename,
                mime_type=request.mime_type,
                total_size=request.total_size,
                sha256=request.sha256,
                directory=directory,
                directory_identity=directory_identity,
                directory_lease=lease,
                spool=spool,
                destination=destination,
                completed=completed,
            )
        return UploadCreated(id=upload_id, chunk_size=CHUNK_LIMIT)

    def _get(self, upload_id: str) -> _UploadRecord:
        with self._lock:
            try:
                return self._records[upload_id]
            except KeyError as error:
                raise ApiError(
                    404,
                    code="upload_not_found",
                    category="upload",
                    message="The requested upload was not found.",
                ) from error

    def put_chunk(self, upload_id: str, index: int, content: bytes) -> None:
        record = self._get(upload_id)
        with record.lock:
            if record.state is not _UploadState.ACTIVE:
                raise ApiError(
                    409,
                    code="upload_not_active",
                    category="upload",
                    message="The upload no longer accepts chunks.",
                )
            self._require_record(record)
            offset = index * CHUNK_LIMIT
            if index < 0 or offset >= record.total_size:
                raise ApiError(
                    400,
                    code="invalid_chunk_index",
                    category="validation",
                    message="The upload chunk index is invalid.",
                )
            expected_size = min(CHUNK_LIMIT, record.total_size - offset)
            if len(content) != expected_size:
                raise ApiError(
                    400,
                    code="chunk_size_mismatch",
                    category="validation",
                    message="The upload chunk size does not match its expected offset.",
                )
            digest = hashlib.sha256(content).hexdigest()
            previous = record.chunk_hashes.get(index)
            if previous is not None:
                record.spool.stream.seek(offset)
                existing = record.spool.stream.read(expected_size)
                if previous == digest and existing == content:
                    return
                raise ApiError(
                    409,
                    code="chunk_conflict",
                    category="upload",
                    message="The upload chunk conflicts with existing bytes.",
                )
            try:
                _write_spool(record.spool.stream, offset=offset, content=content)
                record.spool.verify_handle()
                self._require_record(record)
            except ApiError:
                self._abort_record(record)
                raise
            except (OSError, _UnsafePathError) as error:
                self._raise_storage(error, "Insufficient storage for upload chunk.")
            record.chunk_hashes[index] = digest

    def status(self, upload_id: str) -> UploadStatus:
        record = self._get(upload_id)
        with record.lock:
            self._require_record(record)
            return UploadStatus(
                id=record.id,
                kind=record.kind,
                filename=record.filename,
                total_size=record.total_size,
                chunk_size=CHUNK_LIMIT,
                uploaded_chunks=sorted(record.chunk_hashes),
            )

    def _prepare(self, record: _UploadRecord) -> UploadComplete:
        expected_count = (record.total_size + CHUNK_LIMIT - 1) // CHUNK_LIMIT
        if sorted(record.chunk_hashes) != list(range(expected_count)):
            raise ApiError(
                409,
                code="upload_incomplete",
                category="upload",
                message="The upload is missing one or more chunks.",
                retryable=True,
            )
        self._require_record(record)
        self._require_source()
        try:
            record.destination.verify_link()
            size, digest = _copy_spool_to_destination(record)
            record.spool.verify_handle()
            record.destination.verify_link()
            self._require_record(record)
            self._require_source()
        except ApiError:
            self._abort_record(record)
            raise
        except (OSError, _UnsafePathError) as error:
            if isinstance(error, _UnsafePathError):
                self._abort_record(record)
            self._raise_storage(error, "The completed upload could not be stored.")
        if size != record.total_size or not secrets_compare_hex(
            digest, record.sha256
        ):
            raise ApiError(
                409,
                code="upload_hash_mismatch",
                category="upload",
                message="The completed upload did not match its declared SHA-256.",
            )
        record.state = _UploadState.PREPARED
        return record.completed

    def _prepared(self, record: _UploadRecord) -> UploadComplete:
        if record.destination_removed:
            raise ApiError(
                409,
                code="upload_cancelling",
                category="upload",
                message="The upload is being cancelled.",
                retryable=True,
            )
        self._require_source()
        if record.destination.closed:
            if (
                self._source_root_lease.child_identity(record.destination.name)
                != record.destination.identity
            ):
                raise self._path_changed(
                    "The completed upload identity changed after persistence."
                )
        else:
            record.destination.verify_link()
        return record.completed

    def _cleanup_scratch(self, record: _UploadRecord) -> None:
        if not record.spool.closed:
            record.spool.delete()
        self._require_record(record)
        lease = record.directory_lease
        assert lease is not None
        try:
            lease.delete_if_empty()
            record.directory_lease = None
            record.scratch_removed = True
        except (OSError, _UnsafePathError) as error:
            self._raise_storage(
                error, "The owned upload directory could not be removed."
            )
        try:
            self._upload_root_identity = _read_identity(
                self._upload_root, directory=True
            )
        except (OSError, _UnsafePathError):
            pass

    def _remember_completed(self, record: _UploadRecord) -> None:
        with self._lock:
            record.state = _UploadState.COMMITTED
            self._records.pop(record.id, None)
            self._completed[record.id] = record.completed
            self._completed.move_to_end(record.id)
            while len(self._completed) > self._completed_limit:
                self._completed.popitem(last=False)

    def complete(
        self,
        upload_id: str,
        persist: Callable[[UploadComplete], None] | None = None,
    ) -> UploadComplete:
        with self._lock:
            completed = self._completed.get(upload_id)
        if completed is not None:
            return completed
        record = self._get(upload_id)
        with record.lock:
            if record.state is _UploadState.CANCELLING:
                raise ApiError(
                    409,
                    code="upload_cancelling",
                    category="upload",
                    message="The upload is being cancelled.",
                    retryable=True,
                )
            completed = (
                self._prepare(record)
                if record.state is _UploadState.ACTIVE
                else self._prepared(record)
            )
            if persist is None:
                return completed
            if not record.persisted:
                persist(completed)
                record.persisted = True
            if not record.destination.closed:
                record.destination.keep()
            if not record.scratch_removed:
                self._cleanup_scratch(record)
            self._remember_completed(record)
            return completed

    def cancel(self, upload_id: str) -> None:
        with self._lock:
            if upload_id in self._completed:
                return
        record = self._get(upload_id)
        with record.lock:
            record.state = _UploadState.CANCELLING
            if record.persisted:
                if not record.destination.closed:
                    record.destination.keep()
                if not record.scratch_removed:
                    self._cleanup_scratch(record)
                self._remember_completed(record)
                return
            if not record.destination_removed:
                record.destination.delete()
                record.destination_removed = True
            if not record.scratch_removed:
                self._cleanup_scratch(record)
            record.state = _UploadState.CANCELLED
            with self._lock:
                self._records.pop(upload_id, None)

    def close(self) -> None:
        for upload_id in tuple(self._records):
            try:
                self.cancel(upload_id)
            except (ApiError, OSError, _UnsafePathError):
                record = self._records.get(upload_id)
                if record is not None:
                    try:
                        self._delete_artifacts(record)
                    except (OSError, _UnsafePathError):
                        pass
                    for owned in (record.destination, record.spool):
                        try:
                            owned.release()
                        except OSError:
                            pass
                    if record.directory_lease is not None:
                        self._close_lease_quietly(record.directory_lease)
                        record.directory_lease = None
                    with self._lock:
                        self._records.pop(upload_id, None)
        for lease in (
            self._source_root_lease,
            self._upload_root_lease,
            self._root_lease,
        ):
            self._close_lease_quietly(lease)


def secrets_compare_hex(left: str, right: str) -> bool:
    import secrets

    return secrets.compare_digest(left, right)
