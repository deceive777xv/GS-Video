from __future__ import annotations

import errno
import hashlib
import os
import shutil
from dataclasses import dataclass, field
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


@dataclass
class _UploadRecord:
    id: str
    filename: str
    mime_type: str
    total_size: int
    sha256: str
    directory: Path
    chunk_hashes: dict[int, str] = field(default_factory=dict)


def _write_chunk_atomic(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
        self._upload_root = (self._root / ".uploads").resolve()
        if not self._upload_root.is_relative_to(self._root):
            raise ValueError("upload root must be confined to project root")
        self._upload_root.mkdir(parents=True, exist_ok=True)
        self._max_upload_size = max_upload_size
        self._max_active_uploads = max_active_uploads
        self._records: dict[str, _UploadRecord] = {}
        self._lock = RLock()

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
            upload_id = uuid4().hex
            directory = (self._upload_root / upload_id).resolve()
            if not directory.is_relative_to(self._upload_root):
                raise ApiError(
                    400,
                    code="invalid_upload_id",
                    category="validation",
                    message="The upload identifier is invalid.",
                )
            try:
                directory.mkdir(parents=False, exist_ok=False)
            except OSError as error:
                self._raise_storage_error(error, "Upload storage could not be allocated.")
            record = _UploadRecord(
                id=upload_id,
                filename=request.filename,
                mime_type=request.mime_type,
                total_size=request.total_size,
                sha256=request.sha256,
                directory=directory,
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

    @staticmethod
    def _raise_storage_error(error: OSError, message: str) -> None:
        if error.errno == errno.ENOSPC:
            raise ApiError(
                507,
                code="storage_full",
                category="storage",
                message=message,
            ) from error
        raise ApiError(
            500,
            code="storage_error",
            category="storage",
            message="The upload storage operation failed.",
        ) from error

    def put_chunk(self, upload_id: str, index: int, content: bytes) -> None:
        record = self._get(upload_id)
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
        with self._lock:
            previous = record.chunk_hashes.get(index)
            if previous is not None:
                if previous == digest and path.is_file() and path.read_bytes() == content:
                    return
                raise ApiError(
                    409,
                    code="chunk_conflict",
                    category="upload",
                    message="The upload chunk conflicts with existing bytes.",
                )
            try:
                _write_chunk_atomic(path, content)
            except OSError as error:
                self._raise_storage_error(error, "Insufficient storage for upload chunk.")
            record.chunk_hashes[index] = digest

    def status(self, upload_id: str) -> UploadStatus:
        record = self._get(upload_id)
        return UploadStatus(
            id=record.id,
            filename=record.filename,
            total_size=record.total_size,
            chunk_size=CHUNK_LIMIT,
            uploaded_chunks=sorted(record.chunk_hashes),
        )

    def complete(self, upload_id: str) -> UploadComplete:
        record = self._get(upload_id)
        expected_count = (record.total_size + CHUNK_LIMIT - 1) // CHUNK_LIMIT
        if sorted(record.chunk_hashes) != list(range(expected_count)):
            raise ApiError(
                409,
                code="upload_incomplete",
                category="upload",
                message="The upload is missing one or more chunks.",
                retryable=True,
            )
        assembled = record.directory / "assembled.tmp"
        digest = hashlib.sha256()
        size = 0
        try:
            with assembled.open("xb") as writer:
                for index in range(expected_count):
                    chunk = (record.directory / f"{index}.chunk").read_bytes()
                    writer.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as error:
            assembled.unlink(missing_ok=True)
            self._raise_storage_error(error, "Insufficient storage to complete the upload.")
        if size != record.total_size or not secrets_compare_hex(
            digest.hexdigest(), record.sha256
        ):
            assembled.unlink(missing_ok=True)
            raise ApiError(
                409,
                code="upload_hash_mismatch",
                category="upload",
                message="The completed upload did not match its declared SHA-256.",
            )
        source_dir = (self._root / "source").resolve()
        if not source_dir.is_relative_to(self._root):
            assembled.unlink(missing_ok=True)
            raise ApiError(
                400,
                code="invalid_project_root",
                category="filesystem",
                message="The project input directory is invalid.",
            )
        source_dir.mkdir(parents=True, exist_ok=True)
        destination = (source_dir / f"{record.id}-{record.filename}").resolve()
        if not destination.is_relative_to(source_dir):
            assembled.unlink(missing_ok=True)
            raise ApiError(
                400,
                code="invalid_upload_path",
                category="validation",
                message="The upload destination is invalid.",
            )
        try:
            os.replace(assembled, destination)
        except OSError as error:
            assembled.unlink(missing_ok=True)
            self._raise_storage_error(error, "The completed upload could not be stored.")
        relative = destination.relative_to(self._root).as_posix()
        shutil.rmtree(record.directory)
        with self._lock:
            self._records.pop(upload_id, None)
        return UploadComplete(path=relative)

    def cancel(self, upload_id: str) -> None:
        record = self._get(upload_id)
        directory = record.directory.resolve()
        if not directory.is_relative_to(self._upload_root) or directory.parent != self._upload_root:
            raise ApiError(
                400,
                code="invalid_upload_path",
                category="validation",
                message="The upload directory is invalid.",
            )
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
        except OSError as error:
            self._raise_storage_error(error, "The upload could not be cancelled.")
        with self._lock:
            self._records.pop(upload_id, None)


def secrets_compare_hex(left: str, right: str) -> bool:
    import secrets

    return secrets.compare_digest(left, right)
