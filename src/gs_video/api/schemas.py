from __future__ import annotations

from enum import StrEnum
from ipaddress import ip_address
from math import isfinite
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from gs_video.domain.models import Project, StageName
from gs_video.environment.doctor import EnvironmentReport


API_VERSION = "v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ApiSettings(StrictModel):
    bind_host: str
    port: int = Field(ge=0, le=65535)
    session_token: SecretStr = Field(min_length=1, exclude=True, repr=False)
    allowed_origins: tuple[str, ...]
    event_window: int = Field(default=256, ge=1, le=4096)
    max_tasks: int = Field(default=128, ge=1, le=1024)
    max_active_uploads: int = Field(default=64, ge=1, le=1024)
    task_workers: int = Field(default=2, ge=1, le=8)
    websocket_auth_timeout: float = Field(default=3.0, gt=0, le=3.0)
    shutdown_timeout: float = Field(default=5.0, gt=0, le=30.0)
    max_upload_size: int = Field(default=4 * 1024 * 1024 * 1024, ge=0)
    max_artifact_response_size: int = Field(
        default=512 * 1024 * 1024, ge=1024, le=1024 * 1024 * 1024
    )
    max_json_body_size: int = Field(default=64 * 1024, ge=1024, le=1024 * 1024)
    max_chunk_body_size: int = Field(default=1024 * 1024, ge=1024, le=16 * 1024 * 1024)

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


class EnvironmentRepairState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EnvironmentRepairError(StrictModel):
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=512)
    retryable: bool


class EnvironmentRepairSnapshot(StrictModel):
    state: EnvironmentRepairState
    job_id: str | None = Field(default=None, max_length=128)
    step: str | None = Field(default=None, max_length=64)
    resource_id: str | None = Field(default=None, max_length=128)
    resource_name: str | None = Field(default=None, max_length=256)
    progress: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    downloaded_bytes: int = Field(default=0, ge=0)
    total_bytes: int | None = Field(default=None, ge=0)
    message: str | None = Field(default=None, max_length=512)
    resume_available: bool = False
    restart_required: bool = False
    error: EnvironmentRepairError | None = None
    environment: EnvironmentReport | None = None


class SubjectPromptInput(StrictModel):
    frame_index: int = Field(ge=0)
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class ProjectPatch(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    subject_prompt: SubjectPromptInput | None = None
    motion_scale: float | None = Field(default=None, ge=0.1, le=4.0)
    preview_height: int | None = Field(default=None, ge=180, le=540)


class CameraInput(StrictModel):
    target: list[float] = Field(min_length=3, max_length=3)
    distance: float = Field(gt=0, allow_inf_nan=False)
    yaw: float = Field(allow_inf_nan=False)
    pitch: float = Field(gt=-90, lt=90, allow_inf_nan=False)
    fov_y_degrees: float = Field(gt=1, lt=179, allow_inf_nan=False)

    @field_validator("target")
    @classmethod
    def validate_target(
        cls, value: list[float]
    ) -> list[float]:
        import math

        if not all(math.isfinite(component) for component in value):
            raise ValueError("target must contain finite values")
        return value


class PreviewFrameRequest(StrictModel):
    generation: int = Field(ge=1)
    width: int = Field(gt=0, le=960)
    height: int = Field(gt=0, le=540)
    camera: CameraInput


class LivePreviewRequest(StrictModel):
    request_id: int = Field(ge=1)
    width: int = Field(gt=0, le=960)
    height: int = Field(gt=0, le=540)
    camera: CameraInput


class PreviewFrameResponse(StrictModel):
    artifact_id: str
    generation: int
    width: int
    height: int
    camera_revision: int
    pick_buffer_revision: int


class PickRequest(StrictModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    preview_artifact_id: str = Field(min_length=1, max_length=128)
    camera_revision: int = Field(ge=1)
    pick_buffer_revision: int = Field(ge=1)


class PickResponse(StrictModel):
    image: tuple[int, int]
    world: tuple[float, float, float]
    preview_artifact_id: str
    camera_revision: int
    pick_buffer_revision: int


class CameraConfirmationRequest(StrictModel):
    camera_revision: int = Field(ge=1)


class VerifiedExportResponse(StrictModel):
    artifact_id: str
    filename: str
    size: int
    duration_seconds: float
    fps: str
    frame_count: int
    has_audio: bool
    verified: bool


class CompositePreviewResponse(StrictModel):
    artifact_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    filename: Literal["composite-preview.mp4"]
    size: int = Field(gt=0, le=256 * 1024 * 1024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    fps: str
    frame_count: int = Field(gt=0)


class ExportCopyRequest(StrictModel):
    destination: str = Field(min_length=1, max_length=32767)


class SubjectMediaRole(StrEnum):
    PROXY = "proxy"
    ALPHA = "alpha"


class SubjectMediaResponse(StrictModel):
    role: SubjectMediaRole
    artifact_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    frame_index: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    size: int = Field(gt=0, le=16 * 1024 * 1024)
    mime_type: Literal["image/jpeg", "image/png"]


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
    progress: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    current: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, gt=0)
    message: str | None = Field(default=None, max_length=512)
    elapsed_seconds: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    eta_seconds: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_progress_authority(self) -> TaskSnapshot:
        _validate_task_progress_fields(self)
        return self


class TaskEvent(StrictModel):
    type: str = "task_event"
    task_id: str
    revision: int
    stage: str
    progress: float = Field(ge=0.0, le=1.0)
    current: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, gt=0)
    message: str | None = Field(default=None, max_length=512)
    elapsed_seconds: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    eta_seconds: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    error: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_progress_authority(self) -> TaskEvent:
        _validate_task_progress_fields(self)
        return self


def _validate_task_progress_fields(value: TaskSnapshot | TaskEvent) -> None:
    if (value.current is None) != (value.total is None):
        raise ValueError("current and total must be reported together")
    if value.current is not None and value.total is not None:
        if value.current > value.total:
            raise ValueError("current must not exceed total")
        expected = value.current / value.total
        if abs(value.progress - expected) > 1e-12:
            raise ValueError("progress must equal current / total")
    if not isfinite(value.elapsed_seconds) or (
        value.eta_seconds is not None and not isfinite(value.eta_seconds)
    ):
        raise ValueError("task times must be finite")
    if value.message is not None and any(
        (ord(character) < 32 and character != "\t") or ord(character) == 127
        for character in value.message
    ):
        raise ValueError("task messages must not contain control characters")


class UploadCreateRequest(StrictModel):
    kind: str = AssetKind.SOURCE_VIDEO.value
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=255)
    total_size: int = Field(ge=0)
    sha256: str

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        try:
            return AssetKind(value).value
        except ValueError as error:
            raise ValueError("unsupported asset kind") from error

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
    kind: str
    filename: str
    total_size: int
    chunk_size: int
    uploaded_chunks: list[int]


class UploadComplete(StrictModel):
    path: str
    kind: str = AssetKind.SOURCE_VIDEO.value
    size: int = Field(default=0, ge=0)
    sha256: str = "0" * 64
