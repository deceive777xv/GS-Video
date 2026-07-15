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


class _UnsafePathError(RuntimeError):
    pass


@dataclass(frozen=True)
class _PathIdentity:
    device: int
    inode: int
    fallback_created_ns: int | None = None


class _UploadState(StrEnum):
    ACTIVE = "active"
    PREPARED = "prepared"
    COMMITTED = "committed"
    CANCELLED = "cancelled"


@dataclass
class _UploadRecord:
    id: str
    filename: str
    mime_type: str
    total_size: int
    sha256: str
    directory: Path
    directory_identity: _PathIdentity
    chunk_hashes: dict[int, str] = field(default_factory=dict)
    chunk_identities: dict[int, _PathIdentity] = field(default_factory=dict)
    state: _UploadState = _UploadState.ACTIVE
    completed: UploadComplete | None = None
    destination: Path | None = None
    destination_identity: _PathIdentity | None = None
    lock: RLock = field(default_factory=RLock, repr=False)


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


def _write_chunk_atomic(path: Path, content: bytes) -> _PathIdentity:
    temporary = path.with_suffix(".tmp")
    created = False
    identity: _PathIdentity | None = None
    try:
        with temporary.open("xb") as stream:
            created = True
            stream.write(content)
            identity = _identity_from_stat(os.fstat(stream.fileno()))
        if _read_identity(
            temporary, directory=False, single_link_file=True
        ) != identity:
            raise _UnsafePathError("temporary upload file identity changed")
        os.replace(temporary, path)
        created = False
        if _read_identity(path, directory=False, single_link_file=True) != identity:
            raise _UnsafePathError("published upload chunk identity changed")
        return identity
    finally:
        if created:
            try:
                if identity is not None and _read_identity(
                    temporary, directory=False, single_link_file=True
                ) == identity:
                    temporary.unlink()
            except (FileNotFoundError, OSError, _UnsafePathError):
                pass


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
        self._root = project_root.resolve()
        self._upload_root = self._root / ".uploads"
        self._source_root = self._root / "source"
        if not self._upload_root.is_relative_to(self._root):
            raise ValueError("upload root must be confined to project root")
        self._upload_root.mkdir(parents=True, exist_ok=True)
        self._source_root.mkdir(parents=True, exist_ok=True)
        try:
            self._root_identity = _read_identity(self._root, directory=True)
            self._upload_root_identity = _read_identity(
                self._upload_root, directory=True
            )
            self._source_root_identity = _read_identity(
                self._source_root, directory=True
            )
        except (OSError, _UnsafePathError) as error:
            raise ValueError("project upload paths must be ordinary directories") from error
        self._max_upload_size = max_upload_size
        self._max_active_uploads = max_active_uploads
        self._completed_limit = max(16, max_active_uploads * 2)
        self._records: dict[str, _UploadRecord] = {}
        self._completed: OrderedDict[str, UploadComplete] = OrderedDict()
        self._lock = RLock()
        self._source_lock = RLock()

    @staticmethod
    def _path_changed(message: str) -> ApiError:
        return ApiError(
            409,
            code="upload_path_changed",
            category="filesystem",
            message=message,
        )

    def _require_identity(
        self,
        path: Path,
        expected: _PathIdentity,
        *,
        directory: bool,
        single_link_file: bool = False,
    ) -> None:
        try:
            actual = _read_identity(
                path,
                directory=directory,
                single_link_file=single_link_file,
            )
        except (OSError, _UnsafePathError) as error:
            raise self._path_changed("An upload path is no longer safely owned.") from error
        if actual != expected:
            raise self._path_changed("An upload path identity changed during the session.")

    def _require_roots(self) -> None:
        self._require_identity(self._root, self._root_identity, directory=True)
        self._require_identity(
            self._upload_root, self._upload_root_identity, directory=True
        )

    def _require_record_directory(self, record: _UploadRecord) -> None:
        self._require_roots()
        if record.directory.parent != self._upload_root:
            raise self._path_changed("The upload directory escaped its owned root.")
        self._require_identity(
            record.directory, record.directory_identity, directory=True
        )

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
            upload_id = uuid4().hex
            directory = self._upload_root / upload_id
            try:
                directory.mkdir(parents=False, exist_ok=False)
                identity = _read_identity(directory, directory=True)
                self._upload_root_identity = _read_identity(
                    self._upload_root, directory=True
                )
            except (OSError, _UnsafePathError) as error:
                self._raise_storage_error(
                    error, "Upload storage could not be allocated."
                )
            record = _UploadRecord(
                id=upload_id,
                filename=request.filename,
                mime_type=request.mime_type,
                total_size=request.total_size,
                sha256=request.sha256,
                directory=directory,
                directory_identity=identity,
            )
            self._records[upload_id] = record
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

    def _completed_result(self, upload_id: str) -> UploadComplete | None:
        with self._lock:
            return self._completed.get(upload_id)

    @staticmethod
    def _raise_storage_error(error: BaseException, message: str) -> None:
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            raise ApiError(
                507,
                code="storage_full",
                category="storage",
                message=message,
            ) from error
        if isinstance(error, _UnsafePathError):
            raise UploadManager._path_changed(
                "An upload file is no longer safely owned."
            ) from error
        raise ApiError(
            500,
            code="storage_error",
            category="storage",
            message="The upload storage operation failed.",
        ) from error

    @staticmethod
    def _require_active(record: _UploadRecord) -> None:
        if record.state is not _UploadState.ACTIVE:
            raise ApiError(
                409,
                code="upload_not_active",
                category="upload",
                message="The upload no longer accepts chunks.",
            )

    def put_chunk(self, upload_id: str, index: int, content: bytes) -> None:
        record = self._get(upload_id)
        with record.lock:
            self._require_active(record)
            self._require_record_directory(record)
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
            path = record.directory / f"{index}.chunk"
            previous = record.chunk_hashes.get(index)
            if previous is not None:
                expected_identity = record.chunk_identities[index]
                self._require_identity(
                    path,
                    expected_identity,
                    directory=False,
                    single_link_file=True,
                )
                if previous == digest and self._read_verified_bytes(
                    path, expected_identity
                ) == content:
                    return
                raise ApiError(
                    409,
                    code="chunk_conflict",
                    category="upload",
                    message="The upload chunk conflicts with existing bytes.",
                )
            self._require_record_directory(record)
            try:
                identity = _write_chunk_atomic(path, content)
                record.directory_identity = _read_identity(
                    record.directory, directory=True
                )
            except (OSError, _UnsafePathError) as error:
                self._raise_storage_error(
                    error, "Insufficient storage for upload chunk."
                )
            record.chunk_hashes[index] = digest
            record.chunk_identities[index] = identity

    def _read_verified_bytes(
        self, path: Path, expected: _PathIdentity
    ) -> bytes:
        self._require_identity(
            path, expected, directory=False, single_link_file=True
        )
        try:
            with path.open("rb") as reader:
                if _identity_from_stat(os.fstat(reader.fileno())) != expected:
                    raise self._path_changed(
                        "An upload file identity changed while it was opened."
                    )
                content = reader.read()
        except OSError as error:
            self._raise_storage_error(error, "The upload chunk could not be read.")
        self._require_identity(
            path, expected, directory=False, single_link_file=True
        )
        return content

    def status(self, upload_id: str) -> UploadStatus:
        record = self._get(upload_id)
        with record.lock:
            self._require_record_directory(record)
            return UploadStatus(
                id=record.id,
                filename=record.filename,
                total_size=record.total_size,
                chunk_size=CHUNK_LIMIT,
                uploaded_chunks=sorted(record.chunk_hashes),
            )

    def _verify_chunk(self, record: _UploadRecord, index: int) -> Path:
        path = record.directory / f"{index}.chunk"
        self._require_identity(
            path,
            record.chunk_identities[index],
            directory=False,
            single_link_file=True,
        )
        return path

    def _remove_verified_file(self, path: Path, identity: _PathIdentity) -> None:
        self._require_identity(
            path, identity, directory=False, single_link_file=True
        )
        try:
            path.unlink()
        except OSError as error:
            self._raise_storage_error(error, "The owned upload file could not be removed.")

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
        self._require_record_directory(record)
        assembled = record.directory / "assembled.tmp"
        digest = hashlib.sha256()
        size = 0
        assembled_identity: _PathIdentity | None = None
        try:
            with assembled.open("xb") as writer:
                assembled_identity = _identity_from_stat(os.fstat(writer.fileno()))
                for index in range(expected_count):
                    chunk_path = self._verify_chunk(record, index)
                    with chunk_path.open("rb") as reader:
                        if (
                            _identity_from_stat(os.fstat(reader.fileno()))
                            != record.chunk_identities[index]
                        ):
                            raise _UnsafePathError("upload chunk identity changed")
                        while block := reader.read(1024 * 1024):
                            writer.write(block)
                            digest.update(block)
                            size += len(block)
                    self._require_identity(
                        chunk_path,
                        record.chunk_identities[index],
                        directory=False,
                        single_link_file=True,
                    )
            if _read_identity(
                assembled, directory=False, single_link_file=True
            ) != assembled_identity:
                raise _UnsafePathError("assembled upload identity changed")
            record.directory_identity = _read_identity(record.directory, directory=True)
        except (ApiError, OSError, _UnsafePathError) as error:
            if assembled_identity is not None:
                try:
                    self._remove_verified_file(assembled, assembled_identity)
                    record.directory_identity = _read_identity(
                        record.directory, directory=True
                    )
                except ApiError as cleanup_error:
                    raise cleanup_error from error
            if isinstance(error, ApiError):
                raise
            self._raise_storage_error(
                error, "Insufficient storage to complete the upload."
            )
        if size != record.total_size or not secrets_compare_hex(
            digest.hexdigest(), record.sha256
        ):
            assert assembled_identity is not None
            self._remove_verified_file(assembled, assembled_identity)
            record.directory_identity = _read_identity(record.directory, directory=True)
            raise ApiError(
                409,
                code="upload_hash_mismatch",
                category="upload",
                message="The completed upload did not match its declared SHA-256.",
            )
        with self._source_lock:
            self._require_identity(
                self._source_root, self._source_root_identity, directory=True
            )
            self._require_record_directory(record)
            destination = self._source_root / f"{record.id}-{record.filename}"
            if destination.exists() or destination.is_symlink():
                raise ApiError(
                    409,
                    code="upload_destination_conflict",
                    category="filesystem",
                    message="The completed upload destination already exists.",
                )
            try:
                os.replace(assembled, destination)
                assert assembled_identity is not None
                if _read_identity(
                    destination, directory=False, single_link_file=True
                ) != assembled_identity:
                    raise _UnsafePathError("destination identity changed")
                destination_identity = assembled_identity
                record.directory_identity = _read_identity(
                    record.directory, directory=True
                )
                self._source_root_identity = _read_identity(
                    self._source_root, directory=True
                )
            except (OSError, _UnsafePathError) as error:
                self._raise_storage_error(
                    error, "The completed upload could not be stored."
                )
        relative = destination.relative_to(self._root).as_posix()
        completed = UploadComplete(path=relative)
        record.destination = destination
        record.destination_identity = destination_identity
        record.completed = completed
        record.state = _UploadState.PREPARED
        return completed

    def _prepared(self, record: _UploadRecord) -> UploadComplete:
        if (
            record.completed is None
            or record.destination is None
            or record.destination_identity is None
        ):
            raise RuntimeError("prepared upload is missing completion metadata")
        with self._source_lock:
            self._require_identity(
                self._source_root, self._source_root_identity, directory=True
            )
            self._require_identity(
                record.destination,
                record.destination_identity,
                directory=False,
                single_link_file=True,
            )
        return record.completed

    def _cleanup_directory(self, record: _UploadRecord) -> None:
        with self._lock:
            self._preflight_directory(record)
            for index, identity in sorted(record.chunk_identities.items()):
                self._remove_verified_file(
                    record.directory / f"{index}.chunk", identity
                )
                record.directory_identity = _read_identity(
                    record.directory, directory=True
                )
            self._require_record_directory(record)
            try:
                record.directory.rmdir()
                self._upload_root_identity = _read_identity(
                    self._upload_root, directory=True
                )
            except OSError as error:
                self._raise_storage_error(
                    error, "The owned upload directory could not be removed."
                )

    def _preflight_directory(self, record: _UploadRecord) -> None:
        self._require_record_directory(record)
        expected_names = {
            f"{index}.chunk" for index in record.chunk_identities
        }
        try:
            actual_names = {entry.name for entry in record.directory.iterdir()}
        except OSError as error:
            self._raise_storage_error(
                error, "The upload directory could not be inspected."
            )
        if actual_names != expected_names:
            raise self._path_changed(
                "The upload directory contains files not owned by this upload."
            )
        for index, identity in record.chunk_identities.items():
            self._require_identity(
                record.directory / f"{index}.chunk",
                identity,
                directory=False,
                single_link_file=True,
            )

    def _remember_completed(
        self, record: _UploadRecord, completed: UploadComplete
    ) -> None:
        with self._lock:
            record.state = _UploadState.COMMITTED
            self._records.pop(record.id, None)
            self._completed[record.id] = completed
            self._completed.move_to_end(record.id)
            while len(self._completed) > self._completed_limit:
                self._completed.popitem(last=False)

    def complete(
        self,
        upload_id: str,
        persist: Callable[[UploadComplete], None] | None = None,
    ) -> UploadComplete:
        already_completed = self._completed_result(upload_id)
        if already_completed is not None:
            return already_completed
        record = self._get(upload_id)
        with record.lock:
            if record.state is _UploadState.COMMITTED:
                assert record.completed is not None
                return record.completed
            if record.state is _UploadState.CANCELLED:
                raise ApiError(
                    404,
                    code="upload_not_found",
                    category="upload",
                    message="The requested upload was not found.",
                )
            completed = (
                self._prepare(record)
                if record.state is _UploadState.ACTIVE
                else self._prepared(record)
            )
            if persist is None:
                return completed
            self._preflight_directory(record)
            persist(completed)
            self._cleanup_directory(record)
            self._remember_completed(record, completed)
            return completed

    def cancel(self, upload_id: str) -> None:
        if self._completed_result(upload_id) is not None:
            return
        record = self._get(upload_id)
        with record.lock:
            if record.state is _UploadState.COMMITTED:
                return
            if record.state is _UploadState.PREPARED:
                assert record.destination is not None
                assert record.destination_identity is not None
                with self._source_lock:
                    self._require_identity(
                        self._source_root,
                        self._source_root_identity,
                        directory=True,
                    )
                    self._remove_verified_file(
                        record.destination, record.destination_identity
                    )
                    self._source_root_identity = _read_identity(
                        self._source_root, directory=True
                    )
            self._cleanup_directory(record)
            record.state = _UploadState.CANCELLED
            with self._lock:
                self._records.pop(upload_id, None)


def secrets_compare_hex(left: str, right: str) -> bool:
    import secrets

    return secrets.compare_digest(left, right)
