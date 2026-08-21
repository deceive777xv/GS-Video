from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gs_video.camera.classify import CameraKind
from gs_video.camera.mapping import validate_rigid_transform
from gs_video.camera.solution import CameraSolution, SourceGroundEstimate
from gs_video.segmentation.paths import has_reparse_component


MAX_CAMERA_JSON_BYTES = 64 * 1024 * 1024
Float64Array = npt.NDArray[np.float64]


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_nlink),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _json_native(value: object, name: str) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{name} must contain only finite JSON numbers")
        return value
    if isinstance(value, list):
        return [_json_native(item, f"{name}[]") for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {
            key: _json_native(item, f"{name}.{key}")
            for key, item in value.items()
        }
    raise TypeError(f"{name} must contain only JSON-native values")


def _finite_matrix(value: list[list[float]], shape: tuple[int, int], name: str) -> Float64Array:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix")
    return matrix


def _numeric_lists(value: object, depth: int, name: str) -> object:
    if depth:
        if not isinstance(value, list):
            raise ValueError(f"{name} must contain JSON numeric lists")
        return [
            _numeric_lists(item, depth - 1, f"{name}[]") for item in value
        ]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError(f"{name} must contain only finite JSON numbers")
    return value


def _finite_number(value: object, name: str) -> object:
    return _numeric_lists(value, 0, name)


class _CameraSolutionDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1, 2]
    intrinsics: list[list[float]]
    frame_intrinsics: list[list[list[float]]] | None = None
    camera_to_world: list[list[list[float]]]
    kind: CameraKind
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    diagnostics: dict[str, Any]
    source_ground: dict[str, Any] | None = None

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        if isinstance(value, bool) or value not in (1, 2):
            raise ValueError("version must be the integer 1 or 2")
        return value

    @field_validator("intrinsics", mode="before")
    @classmethod
    def validate_intrinsics_numbers(cls, value: object) -> object:
        return _numeric_lists(value, 2, "intrinsics")

    @field_validator("frame_intrinsics", mode="before")
    @classmethod
    def validate_frame_intrinsics_numbers(cls, value: object) -> object:
        if value is None:
            return None
        return _numeric_lists(value, 3, "frame_intrinsics")

    @field_validator("camera_to_world", mode="before")
    @classmethod
    def validate_pose_numbers(cls, value: object) -> object:
        return _numeric_lists(value, 3, "camera_to_world")

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence_number(cls, value: object) -> object:
        return _finite_number(value, "confidence")

    @field_validator("diagnostics", mode="before")
    @classmethod
    def validate_diagnostics(cls, value: object) -> object:
        return _json_native(value, "diagnostics")

    @field_validator("source_ground", mode="before")
    @classmethod
    def validate_source_ground(cls, value: object) -> object:
        if value is None:
            return None
        return _json_native(value, "source_ground")

    @model_validator(mode="after")
    def validate_matrices(self) -> _CameraSolutionDocument:
        intrinsics = _finite_matrix(self.intrinsics, (3, 3), "intrinsics")
        if (
            intrinsics[0, 0] <= 0
            or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], [0, 0, 1], atol=1e-12)
        ):
            raise ValueError("intrinsics must be a valid pinhole calibration matrix")
        if not self.camera_to_world:
            raise ValueError("camera_to_world must not be empty")
        for index, pose in enumerate(self.camera_to_world):
            validate_rigid_transform(pose, f"camera_to_world[{index}]")
        if self.version == 2:
            if self.frame_intrinsics is None or len(self.frame_intrinsics) != len(
                self.camera_to_world
            ):
                raise ValueError("version 2 requires one intrinsic matrix per pose")
            for index, calibration in enumerate(self.frame_intrinsics):
                matrix = _finite_matrix(calibration, (3, 3), f"frame_intrinsics[{index}]")
                if (
                    matrix[0, 0] <= 0
                    or matrix[1, 1] <= 0
                    or not np.allclose(matrix[2], [0, 0, 1], atol=1e-12)
                ):
                    raise ValueError("frame intrinsics must be valid pinhole matrices")
        return self


class _MappedTrajectoryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1, 2]
    fov_y_degrees: float = Field(gt=0, lt=180, allow_inf_nan=False)
    camera_to_world: list[list[list[float]]]
    frame_intrinsics: list[list[list[float]]] | None = None
    source_width: int | None = Field(default=None, gt=0)
    source_height: int | None = Field(default=None, gt=0)

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        if isinstance(value, bool) or value not in (1, 2):
            raise ValueError("version must be the integer 1 or 2")
        return value

    @field_validator("fov_y_degrees", mode="before")
    @classmethod
    def validate_fov_number(cls, value: object) -> object:
        return _finite_number(value, "fov_y_degrees")

    @field_validator("camera_to_world", mode="before")
    @classmethod
    def validate_pose_numbers(cls, value: object) -> object:
        return _numeric_lists(value, 3, "camera_to_world")

    @field_validator("frame_intrinsics", mode="before")
    @classmethod
    def validate_frame_intrinsics_numbers(cls, value: object) -> object:
        if value is None:
            return None
        return _numeric_lists(value, 3, "frame_intrinsics")

    @model_validator(mode="after")
    def validate_matrices(self) -> _MappedTrajectoryDocument:
        if not self.camera_to_world:
            raise ValueError("camera_to_world must not be empty")
        for index, pose in enumerate(self.camera_to_world):
            validate_rigid_transform(pose, f"camera_to_world[{index}]")
        if self.version == 1:
            if (
                self.frame_intrinsics is not None
                or self.source_width is not None
                or self.source_height is not None
            ):
                raise ValueError("mapped trajectory version 1 must not contain intrinsics")
        elif (
            self.frame_intrinsics is None
            or len(self.frame_intrinsics) != len(self.camera_to_world)
            or self.source_width is None
            or self.source_height is None
        ):
            raise ValueError("mapped trajectory version 2 requires per-frame intrinsics and source size")
        else:
            for index, calibration in enumerate(self.frame_intrinsics):
                matrix = _finite_matrix(
                    calibration, (3, 3), f"frame_intrinsics[{index}]"
                )
                if (
                    matrix[0, 0] <= 0
                    or matrix[1, 1] <= 0
                    or not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-12)
                ):
                    raise ValueError("mapped frame intrinsics must be valid pinhole matrices")
        return self


@dataclass(frozen=True)
class MappedTrajectory:
    fov_y_degrees: float
    camera_to_world: tuple[Float64Array, ...] | list[Float64Array]
    frame_intrinsics: tuple[Float64Array, ...] | list[Float64Array] | None = None
    source_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.fov_y_degrees) or not 0 < self.fov_y_degrees < 180:
            raise ValueError("fov_y_degrees must be finite and between 0 and 180")
        if not self.camera_to_world:
            raise ValueError("camera_to_world must not be empty")
        poses = tuple(
            validate_rigid_transform(pose, f"camera_to_world[{index}]")
            for index, pose in enumerate(self.camera_to_world)
        )
        for pose in poses:
            pose.setflags(write=False)
        if self.frame_intrinsics is None:
            calibrations = None
            if self.source_size is not None:
                raise ValueError("source_size requires frame_intrinsics")
        else:
            if len(self.frame_intrinsics) != len(poses):
                raise ValueError("frame_intrinsics must contain one matrix per mapped pose")
            if (
                self.source_size is None
                or len(self.source_size) != 2
                or any(type(value) is not int or value <= 0 for value in self.source_size)
            ):
                raise ValueError("frame_intrinsics require a positive integer source_size")
            calibrations = tuple(
                _finite_matrix(
                    np.asarray(value, dtype=np.float64).tolist(),
                    (3, 3),
                    f"frame_intrinsics[{index}]",
                )
                for index, value in enumerate(self.frame_intrinsics)
            )
            for calibration in calibrations:
                if (
                    calibration[0, 0] <= 0
                    or calibration[1, 1] <= 0
                    or not np.allclose(calibration[2], (0.0, 0.0, 1.0), atol=1e-12)
                ):
                    raise ValueError("frame_intrinsics must be valid pinhole matrices")
                calibration.setflags(write=False)
        object.__setattr__(self, "camera_to_world", poses)
        object.__setattr__(self, "frame_intrinsics", calibrations)


def _parse_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate camera JSON key: {key}")
        result[key] = value
    return result


def _read_document(path: Path) -> object:
    requested = Path(path).absolute()
    if has_reparse_component(requested):
        raise OSError("camera JSON links and reparse points are not allowed")
    before = requested.lstat()
    expected = _identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
    ):
        raise OSError("camera JSON must be a non-empty single-link regular file")
    if before.st_size > MAX_CAMERA_JSON_BYTES:
        raise OSError("camera JSON is too large")
    with requested.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if _identity(opened) != expected:
            raise OSError("camera JSON identity changed before read")
        payload = stream.read(before.st_size + 1)
        after_handle = os.fstat(stream.fileno())
    after_path = requested.lstat()
    if (
        len(payload) != before.st_size
        or _identity(after_handle) != expected
        or _identity(after_path) != expected
    ):
        raise OSError("camera JSON identity changed during read")
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=_parse_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("camera JSON is invalid") from error


def _write_document(path: Path, payload: dict[str, object]) -> None:
    validated = _json_native(payload, "camera JSON")
    encoded = json.dumps(
        validated,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if not encoded or len(encoded) > MAX_CAMERA_JSON_BYTES:
        raise ValueError("camera JSON exceeds the 64 MiB limit")
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if has_reparse_component(destination.parent):
        raise OSError("camera JSON parent contains a link or reparse point")
    if destination.exists() or destination.is_symlink():
        existing = destination.lstat()
        if (
            has_reparse_component(destination)
            or not stat.S_ISREG(existing.st_mode)
            or existing.st_nlink != 1
        ):
            raise OSError("camera JSON destination is not an owned regular file")
    temporary = destination.parent / f".{destination.name}.staging-{uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_stat = temporary.lstat()
        if (
            has_reparse_component(temporary)
            or not stat.S_ISREG(temporary_stat.st_mode)
            or temporary_stat.st_nlink != 1
        ):
            raise OSError("camera JSON staging identity is unsafe")
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def write_camera_solution(path: Path, solution: CameraSolution) -> None:
    frame_intrinsics = solution.frame_intrinsics
    assert frame_intrinsics is not None
    payload: dict[str, object] = {
        "version": 2,
        "intrinsics": solution.intrinsics.tolist(),
        "frame_intrinsics": [matrix.tolist() for matrix in frame_intrinsics],
        "camera_to_world": [pose.tolist() for pose in solution.camera_to_world],
        "kind": solution.kind.value,
        "confidence": solution.confidence,
        "diagnostics": dict(solution.diagnostics),
        "source_ground": (
            None
            if solution.source_ground is None
            else {
                "normal": list(solution.source_ground.normal),
                "offset": solution.source_ground.offset,
                "anchor_frame_index": solution.source_ground.anchor_frame_index,
                "confidence": solution.source_ground.confidence,
                "support_ratio": solution.source_ground.support_ratio,
                "rms_residual": solution.source_ground.rms_residual,
            }
        ),
    }
    document = _CameraSolutionDocument.model_validate(payload)
    _write_document(path, document.model_dump(mode="json"))


def read_camera_solution(path: Path) -> CameraSolution:
    document = _CameraSolutionDocument.model_validate(_read_document(path))
    source_ground = (
        None
        if document.source_ground is None
        else SourceGroundEstimate(**document.source_ground)
    )
    return CameraSolution(
        intrinsics=np.asarray(document.intrinsics, dtype=np.float64),
        camera_to_world=[
            np.asarray(pose, dtype=np.float64) for pose in document.camera_to_world
        ],
        kind=document.kind,
        confidence=document.confidence,
        diagnostics=document.diagnostics,
        frame_intrinsics=(
            None
            if document.frame_intrinsics is None
            else [
                np.asarray(matrix, dtype=np.float64)
                for matrix in document.frame_intrinsics
            ]
        ),
        source_ground=source_ground,
    )


def write_mapped_trajectory(path: Path, trajectory: MappedTrajectory) -> None:
    payload: dict[str, object] = {
        "version": 1 if trajectory.frame_intrinsics is None else 2,
        "fov_y_degrees": trajectory.fov_y_degrees,
        "camera_to_world": [pose.tolist() for pose in trajectory.camera_to_world],
    }
    if trajectory.frame_intrinsics is not None:
        assert trajectory.source_size is not None
        payload.update(
            {
                "frame_intrinsics": [
                    matrix.tolist() for matrix in trajectory.frame_intrinsics
                ],
                "source_width": trajectory.source_size[0],
                "source_height": trajectory.source_size[1],
            }
        )
    document = _MappedTrajectoryDocument.model_validate(payload)
    _write_document(path, document.model_dump(mode="json", exclude_none=True))


def read_mapped_trajectory(path: Path) -> MappedTrajectory:
    document = _MappedTrajectoryDocument.model_validate(_read_document(path))
    return MappedTrajectory(
        fov_y_degrees=document.fov_y_degrees,
        camera_to_world=[
            np.asarray(pose, dtype=np.float64) for pose in document.camera_to_world
        ],
        frame_intrinsics=(
            None
            if document.frame_intrinsics is None
            else [
                np.asarray(matrix, dtype=np.float64)
                for matrix in document.frame_intrinsics
            ]
        ),
        source_size=(
            None
            if document.source_width is None or document.source_height is None
            else (document.source_width, document.source_height)
        ),
    )
