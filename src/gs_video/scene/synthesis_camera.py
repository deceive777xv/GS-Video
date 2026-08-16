from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin, tan

import numpy as np
import numpy.typing as npt


Float64Array = npt.NDArray[np.float64]
_EPSILON = 1e-9


def _vector3(value: tuple[float, float, float] | Float64Array, *, name: str) -> Float64Array:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _unit(vector: Float64Array, *, name: str) -> Float64Array:
    length = float(np.linalg.norm(vector))
    if not isfinite(length) or length <= _EPSILON:
        raise ValueError(f"{name} must have non-zero length")
    return np.asarray(vector / length, dtype=np.float64)


def _camera_matrix(value: Float64Array, *, name: str) -> Float64Array:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{name} must be an affine transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-7
    ):
        raise ValueError(f"{name} rotation must be rigid")
    return np.array(matrix, dtype=np.float64, copy=True)


def _intrinsics(width: int, height: int, fov_y_degrees: float) -> Float64Array:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    focal_length = 0.5 * height / tan(radians(fov_y_degrees) * 0.5)
    return np.array(
        [
            [focal_length, 0.0, width / 2],
            [0.0, focal_length, height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class LocalGroundPlane:
    """A user-defined local support plane with a stable, camera-facing normal."""

    p0: Float64Array
    p1: Float64Array
    p2: Float64Array
    normal: Float64Array
    basis_u: Float64Array
    basis_v: Float64Array

    @classmethod
    def from_points(
        cls,
        *,
        p0: tuple[float, float, float] | Float64Array,
        p1: tuple[float, float, float] | Float64Array,
        p2: tuple[float, float, float] | Float64Array,
        reference_camera_position: tuple[float, float, float] | Float64Array,
    ) -> LocalGroundPlane:
        origin = _vector3(p0, name="p0")
        point_u = _vector3(p1, name="p1")
        point_v = _vector3(p2, name="p2")
        camera = _vector3(reference_camera_position, name="reference_camera_position")
        edge_u = point_u - origin
        edge_v = point_v - origin
        if np.linalg.norm(edge_u) <= _EPSILON or np.linalg.norm(edge_v) <= _EPSILON:
            raise ValueError("local ground points must be distinct")
        raw_normal = np.cross(edge_u, edge_v)
        if np.linalg.norm(raw_normal) <= _EPSILON * np.linalg.norm(edge_u) * np.linalg.norm(edge_v):
            raise ValueError("local ground points must not be collinear")
        normal = _unit(raw_normal, name="local ground normal")
        camera_side = float(np.dot(camera - origin, normal))
        if abs(camera_side) <= _EPSILON:
            raise ValueError("reference camera must not lie on the local ground plane")
        if camera_side < 0:
            normal = -normal
        basis_u = _unit(edge_u, name="p0-to-p1 direction")
        basis_v = _unit(np.cross(normal, basis_u), name="local ground secondary axis")
        return cls(
            p0=np.array(origin, copy=True),
            p1=np.array(point_u, copy=True),
            p2=np.array(point_v, copy=True),
            normal=np.array(normal, copy=True),
            basis_u=np.array(basis_u, copy=True),
            basis_v=np.array(basis_v, copy=True),
        )


@dataclass(frozen=True)
class SourcePerspective:
    """Source-frame perspective authority independent of subject visibility."""

    width: int
    height: int
    fov_y_degrees: float
    up_camera: Float64Array
    horizon_line: Float64Array

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("source dimensions must be positive")
        if not isfinite(self.fov_y_degrees) or not 0 < self.fov_y_degrees < 180:
            raise ValueError("fov_y_degrees must be finite and between 0 and 180")
        up = _unit(_vector3(self.up_camera, name="up_camera"), name="up_camera")
        horizon = np.asarray(self.horizon_line, dtype=np.float64)
        if horizon.shape != (3,) or not np.all(np.isfinite(horizon)) or np.linalg.norm(horizon[:2]) <= _EPSILON:
            raise ValueError("horizon_line must be a finite image line")
        object.__setattr__(self, "up_camera", up)
        object.__setattr__(self, "horizon_line", horizon / np.linalg.norm(horizon[:2]))

    @classmethod
    def from_horizon(
        cls,
        *,
        width: int,
        height: int,
        fov_y_degrees: float,
        horizon_start: tuple[float, float],
        horizon_end: tuple[float, float],
    ) -> SourcePerspective:
        if len(horizon_start) != 2 or len(horizon_end) != 2:
            raise ValueError("horizon endpoints must contain two values")
        start = np.array((*horizon_start, 1.0), dtype=np.float64)
        end = np.array((*horizon_end, 1.0), dtype=np.float64)
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            raise ValueError("horizon endpoints must be finite")
        line = np.cross(start, end)
        if np.linalg.norm(line[:2]) <= _EPSILON:
            raise ValueError("horizon endpoints must be distinct")
        intrinsics = _intrinsics(width, height, fov_y_degrees)
        up_camera = _unit(intrinsics.T @ line, name="horizon-derived up direction")
        # Real-world footage is expected to be upright within 90 degrees. Resolve
        # the homogeneous line sign so camera-up points toward the top of the image.
        if up_camera[1] > 0:
            up_camera = -up_camera
            line = -line
        return cls(
            width=width,
            height=height,
            fov_y_degrees=fov_y_degrees,
            up_camera=up_camera,
            horizon_line=line,
        )

    def intrinsics(self) -> Float64Array:
        return _intrinsics(self.width, self.height, self.fov_y_degrees)


@dataclass(frozen=True)
class MatrixCamera:
    """Renderer-compatible camera whose pose is already fully constrained."""

    camera_to_world_matrix: Float64Array
    fov_y_degrees: float

    def __post_init__(self) -> None:
        matrix = _camera_matrix(self.camera_to_world_matrix, name="camera_to_world_matrix")
        if not isfinite(self.fov_y_degrees) or not 0 < self.fov_y_degrees < 180:
            raise ValueError("fov_y_degrees must be finite and between 0 and 180")
        object.__setattr__(self, "camera_to_world_matrix", matrix)

    def camera_to_world(self) -> Float64Array:
        return np.array(self.camera_to_world_matrix, copy=True)

    def view_matrix(self) -> Float64Array:
        return np.asarray(np.linalg.inv(self.camera_to_world_matrix), dtype=np.float64)

    def intrinsics(self, width: int, height: int) -> Float64Array:
        return _intrinsics(width, height, self.fov_y_degrees)


@dataclass(frozen=True)
class SynthesisCameraRig:
    """Solve a target camera from source perspective and a local GS plane."""

    ground: LocalGroundPlane
    source_perspective: SourcePerspective
    reference_camera_to_world: Float64Array

    def __post_init__(self) -> None:
        matrix = _camera_matrix(self.reference_camera_to_world, name="reference_camera_to_world")
        height = abs(float(np.dot(matrix[:3, 3] - self.ground.p0, self.ground.normal)))
        if height <= _EPSILON:
            raise ValueError("reference camera must not lie on the local ground")
        object.__setattr__(self, "reference_camera_to_world", matrix)

    @property
    def reference_height(self) -> float:
        return abs(
            float(np.dot(
                self.reference_camera_to_world[:3, 3] - self.ground.p0,
                self.ground.normal,
            ))
        )

    @property
    def reference_horizontal_distance(self) -> float:
        displacement = self.reference_camera_to_world[:3, 3] - self.ground.p0
        horizontal = displacement - np.dot(displacement, self.ground.normal) * self.ground.normal
        return float(np.linalg.norm(horizontal))

    def _rotation(self, scene_azimuth_degrees: float) -> Float64Array:
        if not isfinite(scene_azimuth_degrees):
            raise ValueError("scene_azimuth_degrees must be finite")
        source_up = self.source_perspective.up_camera
        optical_forward = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        source_forward = optical_forward - np.dot(optical_forward, source_up) * source_up
        source_forward = _unit(source_forward, name="source horizontal forward direction")
        source_right = _unit(np.cross(source_forward, source_up), name="source horizontal right direction")

        angle = radians(scene_azimuth_degrees)
        target_forward = cos(angle) * self.ground.basis_u + sin(angle) * self.ground.basis_v
        target_forward = _unit(target_forward, name="target horizontal forward direction")
        target_right = _unit(
            np.cross(target_forward, self.ground.normal),
            name="target horizontal right direction",
        )
        source_basis = np.column_stack((source_right, source_up, source_forward))
        target_basis = np.column_stack((target_right, self.ground.normal, target_forward))
        rotation = target_basis @ source_basis.T
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-8
        ):
            raise ValueError("perspective constraints did not produce a rigid camera rotation")
        return np.asarray(rotation, dtype=np.float64)

    def solve_contact(
        self,
        *,
        contact_pixel: tuple[float, float],
        scene_azimuth_degrees: float = 0.0,
        subject_to_scene_scale: float = 1.0,
    ) -> MatrixCamera:
        if len(contact_pixel) != 2 or not all(isfinite(value) for value in contact_pixel):
            raise ValueError("contact_pixel must contain two finite values")
        pixel_x, pixel_y = contact_pixel
        if not 0 <= pixel_x < self.source_perspective.width or not 0 <= pixel_y < self.source_perspective.height:
            raise ValueError("contact_pixel must lie inside the source frame")
        self._validate_scale(subject_to_scene_scale)
        rotation = self._rotation(scene_azimuth_degrees)
        ray_camera = np.linalg.inv(self.source_perspective.intrinsics()) @ np.array(
            (pixel_x, pixel_y, 1.0), dtype=np.float64
        )
        ray_world = _unit(rotation @ ray_camera, name="contact ray")
        normal_component = float(np.dot(ray_world, self.ground.normal))
        if normal_component >= -_EPSILON:
            raise ValueError("contact pixel ray does not point toward the local ground")
        height = self.reference_height * subject_to_scene_scale
        distance = height / -normal_component
        position = self.ground.p0 - distance * ray_world
        return self._camera(rotation, position)

    def solve_perspective(
        self,
        *,
        scene_azimuth_degrees: float = 0.0,
        subject_to_scene_scale: float = 1.0,
        composition_offset: tuple[float, float] = (0.0, 0.0),
    ) -> MatrixCamera:
        self._validate_scale(subject_to_scene_scale)
        if len(composition_offset) != 2 or not all(isfinite(value) for value in composition_offset):
            raise ValueError("composition_offset must contain two finite values")
        rotation = self._rotation(scene_azimuth_degrees)
        forward = rotation[:, 2]
        horizontal_forward = forward - np.dot(forward, self.ground.normal) * self.ground.normal
        horizontal_forward = _unit(horizontal_forward, name="target horizontal forward direction")
        position = (
            self.ground.p0
            - horizontal_forward * self.reference_horizontal_distance * subject_to_scene_scale
            + self.ground.normal * self.reference_height * subject_to_scene_scale
            + self.ground.basis_u * composition_offset[0] * subject_to_scene_scale
            + self.ground.basis_v * composition_offset[1] * subject_to_scene_scale
        )
        return self._camera(rotation, position)

    @staticmethod
    def _validate_scale(subject_to_scene_scale: float) -> None:
        if not isfinite(subject_to_scene_scale) or subject_to_scene_scale <= 0:
            raise ValueError("subject_to_scene_scale must be finite and positive")

    def _camera(self, rotation: Float64Array, position: Float64Array) -> MatrixCamera:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        return MatrixCamera(
            camera_to_world_matrix=transform,
            fov_y_degrees=self.source_perspective.fov_y_degrees,
        )
