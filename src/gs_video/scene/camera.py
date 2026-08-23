from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin, tan
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt


Float64Array = npt.NDArray[np.float64]
Matrix3Tuple: TypeAlias = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Matrix4Tuple: TypeAlias = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


class CameraMatrices(Protocol):
    def view_matrix(self) -> Float64Array: ...

    def intrinsics(self, width: int, height: int) -> Float64Array: ...


def matrix3_tuple(value: npt.ArrayLike) -> Matrix3Tuple:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must be a finite 3x3 array")
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
        (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])),
    )


def matrix4_tuple(value: npt.ArrayLike) -> Matrix4Tuple:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must be a finite 4x4 array")
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[0, 3])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2]), float(matrix[1, 3])),
        (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2]), float(matrix[2, 3])),
        (float(matrix[3, 0]), float(matrix[3, 1]), float(matrix[3, 2]), float(matrix[3, 3])),
    )


@dataclass(frozen=True)
class OrbitCamera:
    """Orbit camera using OpenCV-style x-right, y-down, z-forward coordinates.

    The domain convention is camera-to-world. Rotation columns are camera right,
    down, and forward axes in world coordinates. ``view_matrix`` is its inverse.
    Positive yaw rotates the view toward +x; positive pitch moves the camera upward.
    """

    target: tuple[float, float, float]
    distance: float
    yaw: float
    pitch: float
    fov_y_degrees: float

    def __post_init__(self) -> None:
        if len(self.target) != 3 or not all(isfinite(value) for value in self.target):
            raise ValueError("target must contain three finite values")
        if not isfinite(self.distance) or self.distance <= 0:
            raise ValueError("distance must be finite and positive")
        if not isfinite(self.yaw):
            raise ValueError("yaw must be finite")
        if not isfinite(self.pitch) or not -90 < self.pitch < 90:
            raise ValueError("pitch must be finite and between -90 and 90 degrees")
        if not isfinite(self.fov_y_degrees) or not 0 < self.fov_y_degrees < 180:
            raise ValueError("fov_y_degrees must be finite and between 0 and 180")

    def camera_to_world(self) -> Float64Array:
        """Return the domain camera-to-world transform."""

        yaw = radians(self.yaw)
        pitch = radians(self.pitch)
        forward = np.array(
            [sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)],
            dtype=np.float64,
        )
        right = np.array([cos(yaw), 0.0, -sin(yaw)], dtype=np.float64)
        down = np.cross(forward, right)
        target = np.asarray(self.target, dtype=np.float64)
        position = target - forward * self.distance

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.column_stack((right, down, forward))
        transform[:3, 3] = position
        return transform

    def view_matrix(self) -> Float64Array:
        """Return the OpenCV-style world-to-camera inverse transform."""

        return np.asarray(np.linalg.inv(self.camera_to_world()), dtype=np.float64)

    def intrinsics(self, width: int, height: int) -> Float64Array:
        """Return pinhole intrinsics with centered principal point and square pixels."""

        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        focal_length = 0.5 * height / tan(radians(self.fov_y_degrees) * 0.5)
        return np.array(
            [
                [focal_length, 0.0, width / 2],
                [0.0, focal_length, height / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class MatrixCamera:
    """Renderer-compatible camera with an authoritative camera-to-world pose."""

    camera_to_world_matrix: Float64Array
    fov_y_degrees: float
    intrinsics_matrix: Float64Array | None = None
    source_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_to_world_matrix, dtype=np.float64)
        rotation = matrix[:3, :3] if matrix.shape == (4, 4) else np.empty((0, 0))
        if (
            matrix.shape != (4, 4)
            or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        ):
            raise ValueError("camera_to_world_matrix must be a finite rigid transform")
        if not isfinite(self.fov_y_degrees) or not 0 < self.fov_y_degrees < 180:
            raise ValueError("fov_y_degrees must be finite and between 0 and 180")
        intrinsics = self.intrinsics_matrix
        source_size = self.source_size
        if (intrinsics is None) != (source_size is None):
            raise ValueError("authoritative intrinsics require a source size")
        if intrinsics is not None:
            calibration = np.asarray(intrinsics, dtype=np.float64)
            if (
                calibration.shape != (3, 3)
                or not np.all(np.isfinite(calibration))
                or calibration[0, 0] <= 0
                or calibration[1, 1] <= 0
                or not np.allclose(calibration[2], (0.0, 0.0, 1.0), atol=1e-12)
                or source_size is None
                or type(source_size[0]) is not int
                or type(source_size[1]) is not int
                or source_size[0] <= 0
                or source_size[1] <= 0
            ):
                raise ValueError("intrinsics_matrix must be a valid pinhole calibration")
            object.__setattr__(self, "intrinsics_matrix", calibration.copy())
        object.__setattr__(self, "camera_to_world_matrix", matrix.copy())

    def camera_to_world(self) -> Float64Array:
        return np.array(self.camera_to_world_matrix, copy=True)

    def view_matrix(self) -> Float64Array:
        return np.asarray(np.linalg.inv(self.camera_to_world_matrix), dtype=np.float64)

    def intrinsics(self, width: int, height: int) -> Float64Array:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.intrinsics_matrix is not None and self.source_size is not None:
            source_width, source_height = self.source_size
            matrix = np.asarray(self.intrinsics_matrix, dtype=np.float64).copy()
            matrix[0, :] *= width / source_width
            matrix[1, :] *= height / source_height
            return matrix
        focal_length = 0.5 * height / tan(radians(self.fov_y_degrees) * 0.5)
        return np.array(
            [
                [focal_length, 0.0, width / 2],
                [0.0, focal_length, height / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
