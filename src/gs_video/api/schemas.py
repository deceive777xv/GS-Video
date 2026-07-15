from __future__ import annotations

from enum import StrEnum
from ipaddress import ip_address
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gs_video.domain.models import Project, StageName
from gs_video.environment.doctor import EnvironmentReport


API_VERSION = "v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ApiSettings(StrictModel):
    bind_host: str
    port: int = Field(ge=0, le=65535)
    session_token: str = Field(min_length=1)
    allowed_origins: tuple[str, ...]
    event_window: int = Field(default=256, ge=1, le=4096)
    max_tasks: int = Field(default=128, ge=1, le=1024)
    max_active_uploads: int = Field(default=64, ge=1, le=1024)
    task_workers: int = Field(default=2, ge=1, le=8)
    websocket_auth_timeout: float = Field(default=3.0, gt=0, le=3.0)
    shutdown_timeout: float = Field(default=5.0, gt=0, le=30.0)
    max_upload_size: int = Field(default=4 * 1024 * 1024 * 1024, ge=0)

    @field_validator("bind_host")
    @classmethod
    def validate_loopback_ip(cls, value: str) -> str:
        try:
            address = ip_address(value)
        except ValueError as error:
            raise ValueError("bind_host must be an IP loopback address") from error
        if not address.is_loopback:
            raise ValueError("bind_host must be an IP loopback address")
        return value

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(value == "*" for value in values):
            raise ValueError("wildcard origins are not allowed")
        if len(set(values)) != len(values):
            raise ValueError("allowed origins must be unique")
        return values


class ApiError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        *,
        code: str,
        category: str,
        message: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.envelope = ErrorEnvelope(
            code=code,
            category=category,
            message=message,
            retryable=retryable,
        )


class ErrorEnvelope(StrictModel):
    code: str
    category: str
    message: str
    retryable: bool


class HealthResponse(StrictModel):
    status: str


class BootstrapResponse(StrictModel):
    api_version: str
    capabilities: tuple[str, ...]
    project: Project
    environment: EnvironmentReport


class ProjectPatch(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)


class AssetKind(StrEnum):
    SOURCE_VIDEO = "source_video"
    SCENE_PLY = "scene_ply"


class AssetImportRequest(StrictModel):
    path: str = Field(min_length=1)
    kind: str

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        try:
            return AssetKind(value).value
        except ValueError as error:
            raise ValueError("unsupported asset kind") from error


class AssetResponse(StrictModel):
    kind: str
    path: str
    size: int
    sha256: str


class TaskCreateRequest(StrictModel):
    target_stage: str

    @field_validator("target_stage")
    @classmethod
    def validate_target_stage(cls, value: str) -> str:
        try:
            return StageName(value).value
        except ValueError as error:
            raise ValueError("unsupported target stage") from error


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskSnapshot(StrictModel):
    id: str
    target_stage: str
    status: str
    revision: int
    error: str | None = None


class TaskEvent(StrictModel):
    type: str = "task_event"
    task_id: str
    revision: int
    stage: str
    progress: float = Field(ge=0.0, le=1.0)
    error: dict[str, Any] | None = None


class UploadCreateRequest(StrictModel):
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=255)
    total_size: int = Field(ge=0)
    sha256: str

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        if value in {".", ".."} or Path(value).name != value or PurePath(value).name != value:
            raise ValueError("filename must be a basename")
        if PureWindowsPath(value).name != value:
            raise ValueError("filename must be a basename")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        return normalized


class UploadCreated(StrictModel):
    id: str
    chunk_size: int


class UploadStatus(StrictModel):
    id: str
    filename: str
    total_size: int
    chunk_size: int
    uploaded_chunks: list[int]


class UploadComplete(StrictModel):
    path: str
