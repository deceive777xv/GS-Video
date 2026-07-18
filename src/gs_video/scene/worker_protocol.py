from __future__ import annotations

import json
import os
import stat
import uuid
from collections.abc import Callable
from math import isfinite
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from gs_video.segmentation.paths import has_reparse_component


MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_EVENT_BYTES = 64 * 1024
MAX_EVENT_MESSAGE_CHARS = 512
DirectoryIdentity: TypeAlias = tuple[int, int]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _finite(value: object, name: str) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _absolute_path(value: Path, name: str) -> Path:
    if not value.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return value


class OrbitCameraPayload(_StrictModel):
    target: tuple[float, float, float]
    distance: float = Field(gt=0)
    yaw: float
    pitch: float = Field(gt=-90, lt=90)
    fov_y_degrees: float = Field(gt=0, lt=180)

    @field_validator("target", mode="before")
    @classmethod
    def validate_target(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("target must contain three values")
        for item in value:
            _finite(item, "target")
        return value

    @field_validator("distance", "yaw", "pitch", "fov_y_degrees", mode="before")
    @classmethod
    def validate_number(cls, value: object) -> object:
        return _finite(value, "camera value")


class RenderSequenceRequest(_StrictModel):
    type: Literal["render_sequence"]
    scene_path: Path
    camera_manifest: Path
    output_dir: Path
    width: int = Field(strict=True, gt=0, le=16384)
    height: int = Field(strict=True, gt=0, le=16384)
    sh_degree: int = Field(strict=True, ge=0, le=3)
    background: tuple[float, float, float]
    preview_stride: int = Field(strict=True, ge=1)

    @field_validator("scene_path", "camera_manifest", "output_dir")
    @classmethod
    def validate_path(cls, value: Path, info: object) -> Path:
        field_name = getattr(info, "field_name", "path")
        return _absolute_path(value, str(field_name))

    @field_validator("background", mode="before")
    @classmethod
    def validate_background(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("background must contain three values")
        for item in value:
            _finite(item, "background")
        return value

    @model_validator(mode="after")
    def validate_distinct_paths(self) -> RenderSequenceRequest:
        if self.output_dir in (self.scene_path, self.camera_manifest):
            raise ValueError("output_dir must differ from inputs")
        return self


class RenderPickRequest(_StrictModel):
    type: Literal["render_pick"]
    scene_path: Path
    output_npz: Path
    camera: OrbitCameraPayload
    width: int = Field(strict=True, gt=0, le=16384)
    height: int = Field(strict=True, gt=0, le=16384)

    @field_validator("scene_path", "output_npz")
    @classmethod
    def validate_path(cls, value: Path, info: object) -> Path:
        field_name = getattr(info, "field_name", "path")
        return _absolute_path(value, str(field_name))

    @model_validator(mode="after")
    def validate_distinct_paths(self) -> RenderPickRequest:
        if self.output_npz == self.scene_path:
            raise ValueError("output_npz must differ from scene_path")
        if self.output_npz.suffix.lower() != ".npz":
            raise ValueError("output_npz must have .npz suffix")
        return self


class ProbeRequest(_StrictModel):
    type: Literal["probe"]


WorkerRequest: TypeAlias = Annotated[
    RenderSequenceRequest | RenderPickRequest | ProbeRequest,
    Field(discriminator="type"),
]
_REQUEST_ADAPTER: TypeAdapter[WorkerRequest] = TypeAdapter(WorkerRequest)


class ProgressEvent(_StrictModel):
    type: Literal["progress"]
    current: int = Field(strict=True, ge=1)
    total: int = Field(strict=True, ge=1)
    message: str = Field(min_length=1, max_length=MAX_EVENT_MESSAGE_CHARS)

    @model_validator(mode="after")
    def validate_range(self) -> ProgressEvent:
        if self.current > self.total:
            raise ValueError("progress current exceeds total")
        return self


class CompleteEvent(_StrictModel):
    type: Literal["complete"]
    implementation_version: str = Field(min_length=1, max_length=128)
    outputs: list[str] = Field(max_length=100_000)

    @field_validator("outputs")
    @classmethod
    def validate_outputs(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("terminal outputs must be unique")
        for item in value:
            path = Path(item)
            if (
                not item
                or len(item) > 255
                or path.is_absolute()
                or len(path.parts) != 1
                or item in {".", ".."}
            ):
                raise ValueError("terminal outputs must be relative basenames")
        return value


class ErrorEvent(_StrictModel):
    type: Literal["error"]
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    message: str = Field(min_length=1, max_length=MAX_EVENT_MESSAGE_CHARS)


class ProbeEvent(_StrictModel):
    type: Literal["probe"]
    torch: str = Field(min_length=1, max_length=128)
    gsplat: str = Field(min_length=1, max_length=128)
    device: Literal["cuda"]


WorkerEvent: TypeAlias = Annotated[
    ProgressEvent | CompleteEvent | ErrorEvent | ProbeEvent,
    Field(discriminator="type"),
]
_EVENT_ADAPTER: TypeAdapter[WorkerEvent] = TypeAdapter(WorkerEvent)


def _parse_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(payload: bytes) -> object:
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=_parse_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("renderer worker JSON is invalid UTF-8 JSON") from exc


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_nlink),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _directory_identity(metadata: os.stat_result) -> DirectoryIdentity:
    return int(metadata.st_dev), int(metadata.st_ino)


def _directory_chain(path: Path) -> tuple[Path, ...]:
    absolute = Path(path).absolute()
    parts = absolute.parts
    if not parts:
        raise OSError("directory path is empty")
    current = Path(parts[0])
    chain = [current]
    for part in parts[1:]:
        current /= part
        chain.append(current)
    return tuple(chain)


def _validate_existing_directory_chain(path: Path) -> None:
    for component in _directory_chain(path):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if has_reparse_component(component) or not stat.S_ISDIR(metadata.st_mode):
            raise OSError("directory chain contains a link, reparse point, or non-directory")


def ensure_safe_directory(path: Path) -> DirectoryIdentity:
    directory = Path(path).absolute()
    _validate_existing_directory_chain(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _validate_existing_directory_chain(directory)
    metadata = directory.lstat()
    if has_reparse_component(directory) or not stat.S_ISDIR(metadata.st_mode):
        raise OSError("created directory is not an ordinary non-reparse directory")
    identity = _directory_identity(metadata)
    if _directory_identity(directory.lstat()) != identity:
        raise OSError("directory identity changed during validation")
    return identity


def assert_safe_directory(path: Path, expected: DirectoryIdentity) -> None:
    directory = Path(path).absolute()
    _validate_existing_directory_chain(directory)
    metadata = directory.lstat()
    if (
        has_reparse_component(directory)
        or not stat.S_ISDIR(metadata.st_mode)
        or _directory_identity(metadata) != expected
    ):
        raise OSError("directory identity changed during operation")


def read_worker_request(path: Path) -> WorkerRequest:
    requested = Path(path).absolute()
    if has_reparse_component(requested):
        raise ValueError("renderer request links and reparse points are forbidden")
    before = requested.lstat()
    expected = _identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
    ):
        raise ValueError("renderer request must be a non-empty owned regular file")
    if before.st_size > MAX_REQUEST_BYTES:
        raise ValueError("renderer request exceeds the 16 MiB limit")
    with requested.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if _identity(opened) != expected:
            raise ValueError("renderer request identity changed before read")
        payload = stream.read(before.st_size + 1)
        after_handle = os.fstat(stream.fileno())
    after_path = requested.lstat()
    if (
        len(payload) != before.st_size
        or _identity(after_handle) != expected
        or _identity(after_path) != expected
    ):
        raise ValueError("renderer request identity changed during read")
    return _REQUEST_ADAPTER.validate_python(_decode_json(payload))


def write_worker_request(path: Path, request: WorkerRequest) -> None:
    encoded = json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if not encoded or len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("renderer request exceeds the 16 MiB limit")
    destination = Path(path).absolute()
    parent_identity = ensure_safe_directory(destination.parent)
    temporary = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        assert_safe_directory(destination.parent, parent_identity)
    finally:
        temporary.unlink(missing_ok=True)


def parse_worker_event(payload: bytes) -> WorkerEvent:
    if len(payload) > MAX_EVENT_BYTES:
        raise ValueError("renderer worker event exceeds the 64 KiB limit")
    if not payload:
        raise ValueError("renderer worker event is empty")
    return _EVENT_ADAPTER.validate_python(_decode_json(payload))


def encode_worker_event(event: WorkerEvent) -> bytes:
    encoded = (
        json.dumps(
            event.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_EVENT_BYTES:
        raise ValueError("renderer worker event exceeds the 64 KiB limit")
    return encoded


EventEmitter: TypeAlias = Callable[[WorkerEvent], None]
