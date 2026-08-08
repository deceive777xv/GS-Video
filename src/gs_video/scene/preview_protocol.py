from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Annotated, Literal, TypeAlias
import uuid

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from gs_video.scene.worker_protocol import OrbitCameraPayload
from gs_video.scene.worker_protocol import assert_safe_directory, ensure_safe_directory
from gs_video.segmentation.paths import has_reparse_component


MAX_PREVIEW_MESSAGE_BYTES = 64 * 1024


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _absolute(value: Path) -> Path:
    if not value.is_absolute():
        raise ValueError("preview session paths must be absolute")
    return value


class OpenPreviewSessionRequest(_StrictModel):
    type: Literal["open_preview_session"]
    scene_path: Path
    output_root: Path
    scene_size: int = Field(strict=True, gt=0)
    scene_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sh_degree: int = Field(strict=True, ge=0, le=3)
    maximum_width: int = Field(strict=True, gt=0, le=960)
    maximum_height: int = Field(strict=True, gt=0, le=540)
    available_vram_limit_mb: int = Field(strict=True, ge=1024, le=8192)
    initial_camera: OrbitCameraPayload

    @field_validator("scene_path", "output_root")
    @classmethod
    def validate_absolute(cls, value: Path) -> Path:
        return _absolute(value)


class RenderLiveCommand(_StrictModel):
    type: Literal["render_live"]
    request_id: int = Field(strict=True, ge=1)
    output_path: Path
    camera: OrbitCameraPayload
    width: int = Field(strict=True, gt=0, le=960)
    height: int = Field(strict=True, gt=0, le=540)

    @field_validator("output_path")
    @classmethod
    def validate_output(cls, value: Path) -> Path:
        return _absolute(value)


class RenderPickCommand(_StrictModel):
    type: Literal["render_pick"]
    request_id: int = Field(strict=True, ge=1)
    output_path: Path
    camera: OrbitCameraPayload
    width: int = Field(strict=True, gt=0, le=960)
    height: int = Field(strict=True, gt=0, le=540)

    @field_validator("output_path")
    @classmethod
    def validate_output(cls, value: Path) -> Path:
        return _absolute(value)


PreviewCommand: TypeAlias = Annotated[
    RenderLiveCommand | RenderPickCommand, Field(discriminator="type")
]


class ReadyEvent(_StrictModel):
    type: Literal["ready"]
    implementation_version: str = Field(min_length=1, max_length=128)


class PreviewTimingPayload(_StrictModel):
    gpu_raster_ms: float = Field(ge=0)
    readback_ms: float = Field(ge=0)
    jpeg_ms: float = Field(ge=0)


class CompleteEvent(_StrictModel):
    type: Literal["complete"]
    request_id: int = Field(strict=True, ge=1)
    output: str = Field(min_length=1, max_length=255)
    timings: PreviewTimingPayload | None = None


class ErrorEvent(_StrictModel):
    type: Literal["error"]
    request_id: int | None = Field(default=None, ge=1)
    code: Literal[
        "invalid_request",
        "resource_exhausted",
        "scene_changed",
        "system_error",
    ]
    message: str = Field(min_length=1, max_length=512)


PreviewEvent: TypeAlias = Annotated[
    ReadyEvent | CompleteEvent | ErrorEvent, Field(discriminator="type")
]

_COMMAND: TypeAdapter[PreviewCommand] = TypeAdapter(PreviewCommand)
_EVENT: TypeAdapter[PreviewEvent] = TypeAdapter(PreviewEvent)


def _decode(payload: bytes) -> object:
    if not payload or len(payload) > MAX_PREVIEW_MESSAGE_BYTES:
        raise ValueError("preview protocol message size is invalid")
    return json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicates)


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("preview protocol contains a duplicate key")
        result[key] = value
    return result


def parse_command(payload: bytes) -> PreviewCommand:
    return _COMMAND.validate_python(_decode(payload))


def parse_event(payload: bytes) -> PreviewEvent:
    return _EVENT.validate_python(_decode(payload))


def encode_message(message: BaseModel) -> bytes:
    payload = json.dumps(
        message.model_dump(mode="json"), separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    if len(payload) > MAX_PREVIEW_MESSAGE_BYTES:
        raise ValueError("preview protocol message exceeds the size limit")
    return payload


def read_open_request(path: Path) -> OpenPreviewSessionRequest:
    requested = Path(path).absolute()
    if has_reparse_component(requested):
        raise ValueError("preview open request links are forbidden")
    before = requested.lstat()
    identity = _file_identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > MAX_PREVIEW_MESSAGE_BYTES
    ):
        raise ValueError("preview open request size is invalid")
    with requested.open("rb") as stream:
        if _file_identity(os.fstat(stream.fileno())) != identity:
            raise ValueError("preview open request identity changed before read")
        payload = stream.read(before.st_size + 1)
        after_handle = _file_identity(os.fstat(stream.fileno()))
    if (
        len(payload) != before.st_size
        or after_handle != identity
        or _file_identity(requested.lstat()) != identity
    ):
        raise ValueError("preview open request identity changed during read")
    return OpenPreviewSessionRequest.model_validate(_decode(payload))


def write_open_request(path: Path, request: OpenPreviewSessionRequest) -> None:
    payload = json.dumps(
        request.model_dump(mode="json"), separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(payload) > MAX_PREVIEW_MESSAGE_BYTES:
        raise ValueError("preview open request exceeds the size limit")
    destination = Path(path).absolute()
    parent_identity = ensure_safe_directory(destination.parent)
    temporary = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        assert_safe_directory(destination.parent, parent_identity)
    finally:
        temporary.unlink(missing_ok=True)


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_nlink),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )
