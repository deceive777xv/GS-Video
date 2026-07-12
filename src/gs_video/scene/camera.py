from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin, tan

import numpy as np
import numpy.typing as npt


Float64Array = npt.NDArray[np.float64]


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
