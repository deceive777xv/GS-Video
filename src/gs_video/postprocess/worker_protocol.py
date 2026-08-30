from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gs_video.domain.models import EffectInstance
from gs_video.segmentation.paths import has_reparse_component


MAX_REQUEST_BYTES = 16 * 1024 * 1024


class ProcessSequenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: Literal["process_sequence"] = "process_sequence"
    frame_paths: list[Path] = Field(min_length=1, max_length=100_000)
    output_dir: Path
    effects: list[EffectInstance] = Field(max_length=32)
    lut_paths: dict[str, Path] = Field(default_factory=dict, max_length=32)
    spatial_scale: float = Field(default=1.0, gt=0, le=16)
    vram_limit_mb: int = Field(ge=1024, le=1024 * 1024)

    @field_validator("frame_paths", "output_dir")
    @classmethod
    def validate_paths(cls, value: object) -> object:
        paths = value if isinstance(value, list) else [value]
        if any(not isinstance(path, Path) or not path.is_absolute() for path in paths):
            raise ValueError("post-process worker paths must be absolute")
        return value

    @field_validator("lut_paths")
    @classmethod
    def validate_lut_paths(cls, value: dict[str, Path]) -> dict[str, Path]:
        if any(not asset_id or not path.is_absolute() for asset_id, path in value.items()):
            raise ValueError("managed LUT paths must be absolute and keyed by asset ID")
        return value

    @model_validator(mode="after")
    def validate_inventory(self) -> ProcessSequenceRequest:
        if len(set(self.frame_paths)) != len(self.frame_paths):
            raise ValueError("post-process frame inventory contains duplicates")
        if any(path == self.output_dir for path in self.frame_paths):
            raise ValueError("post-process output must differ from its inputs")
        return self


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def write_request(path: Path, request: ProcessSequenceRequest) -> None:
    encoded = json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if not encoded or len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("post-process request exceeds 16 MiB")
    path.write_bytes(encoded)


def read_request(path: Path) -> ProcessSequenceRequest:
    requested = path.absolute()
    if has_reparse_component(requested):
        raise ValueError("post-process request cannot contain links")
    before = requested.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > MAX_REQUEST_BYTES
    ):
        raise ValueError("post-process request must be an owned bounded file")
    with requested.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("post-process request changed before read")
        payload = stream.read(MAX_REQUEST_BYTES + 1)
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            parse_constant=_parse_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("post-process request is invalid JSON") from error
    return ProcessSequenceRequest.model_validate(raw)
