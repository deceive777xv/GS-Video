from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from gs_video.camera.mapping import validate_rigid_transform
from gs_video.domain.contracts import PickBuffer


Float64Array = npt.NDArray[np.float64]
ImagePoint: TypeAlias = tuple[int, int]
WorldPoint: TypeAlias = tuple[float, float, float]


@dataclass(frozen=True)
class GroundFitSettings:
    neighborhood_radius_pixels: int = 28
    minimum_opacity: float = 0.2
    minimum_candidates_per_hint: int = 8
    ransac_iterations: int = 768
    maximum_relative_residual: float = 0.012
    maximum_relative_depth_spread: float = 0.12

    def __post_init__(self) -> None:
        if self.neighborhood_radius_pixels < 4:
            raise ValueError("ground neighborhood radius must be at least 4 pixels")
        if not 0 < self.minimum_opacity <= 1:
            raise ValueError("ground minimum opacity must lie in (0, 1]")
        if self.minimum_candidates_per_hint < 3:
            raise ValueError("ground fit requires at least three candidates per hint")
        if self.ransac_iterations < 1:
            raise ValueError("ground RANSAC iterations must be positive")
        if self.maximum_relative_residual <= 0:
            raise ValueError("ground residual threshold must be positive")
        if self.maximum_relative_depth_spread <= 0:
            raise ValueError("ground depth spread threshold must be positive")


@dataclass(frozen=True)
class GroundFitCandidate:
    plane_normal: WorldPoint
    plane_offset: float
    refined_points: tuple[WorldPoint, WorldPoint, WorldPoint]
    support_counts: tuple[int, int, int]
    weighted_inlier_ratio: float
    rms_residual: float
    confidence: float


def _validate_intrinsics(value: npt.ArrayLike) -> Float64Array:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-8)
    ):
        raise ValueError("ground fit intrinsics must be a finite pinhole matrix")
    return matrix


def _validate_pick_buffer(buffer: PickBuffer) -> tuple[int, int]:
    if (
        buffer.rgb.ndim != 3
        or buffer.rgb.shape[2] != 3
        or buffer.rgb.dtype != np.uint8
        or buffer.expected_depth.shape != buffer.rgb.shape[:2]
        or buffer.expected_depth.dtype != np.float32
        or buffer.opacity.shape != buffer.rgb.shape[:2]
        or buffer.opacity.dtype != np.float32
        or not np.isfinite(buffer.expected_depth).all()
        or np.any(buffer.expected_depth < 0)
        or not np.isfinite(buffer.opacity).all()
        or np.any((buffer.opacity < 0) | (buffer.opacity > 1))
    ):
        raise ValueError("ground fit pick buffer is invalid")
    return buffer.rgb.shape[1], buffer.rgb.shape[0]


def _weighted_plane(points: Float64Array, weights: Float64Array) -> tuple[Float64Array, float]:
    total = float(np.sum(weights))
    if not isfinite(total) or total <= 0:
        raise ValueError("ground support weights are invalid")
    centroid = np.sum(points * weights[:, None], axis=0) / total
    centered = points - centroid
    covariance = (centered * weights[:, None]).T @ centered / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[1] <= 1e-12 or eigenvalues[2] <= 1e-12:
        raise ValueError("ground support is geometrically degenerate")
    normal = eigenvectors[:, 0]
    normal /= np.linalg.norm(normal)
    return normal, -float(np.dot(normal, centroid))


def _ray_plane_point(
    pixel: ImagePoint,
    intrinsics: Float64Array,
    camera_to_world: Float64Array,
    normal: Float64Array,
    offset: float,
) -> WorldPoint:
    camera_ray = np.linalg.inv(intrinsics) @ np.asarray(
        (float(pixel[0]) + 0.5, float(pixel[1]) + 0.5, 1.0), dtype=np.float64
    )
    direction = camera_to_world[:3, :3] @ camera_ray
    origin = camera_to_world[:3, 3]
    denominator = float(np.dot(normal, direction))
    if abs(denominator) <= 1e-9:
        raise ValueError("ground hint ray is parallel to the fitted plane")
    distance = -(float(np.dot(normal, origin)) + offset) / denominator
    if not isfinite(distance) or distance <= 0:
        raise ValueError("fitted ground lies behind the exploration camera")
    point = origin + distance * direction
    return float(point[0]), float(point[1]), float(point[2])


def fit_ground_from_gaussians(
    *,
    means_world: npt.ArrayLike,
    scales_world: npt.ArrayLike,
    opacities: npt.ArrayLike,
    camera_to_world: npt.ArrayLike,
    intrinsics: npt.ArrayLike,
    pick_buffer: PickBuffer,
    hints: tuple[ImagePoint, ImagePoint, ImagePoint],
    settings: GroundFitSettings = GroundFitSettings(),
) -> GroundFitCandidate:
    """Fit one visible GS support plane around three approximate viewport hints.

    Expected depth is used only as an occlusion/depth-continuity gate. The plane is
    fitted to actual Gaussian centers, never to three sampled depth pixels.
    """

    width, height = _validate_pick_buffer(pick_buffer)
    transform = validate_rigid_transform(camera_to_world, "camera_to_world")
    calibration = _validate_intrinsics(intrinsics)
    means = np.asarray(means_world, dtype=np.float64)
    scales = np.asarray(scales_world, dtype=np.float64)
    opacity = np.asarray(opacities, dtype=np.float64)
    if (
        means.ndim != 2
        or means.shape[1] != 3
        or scales.shape != means.shape
        or opacity.shape != (means.shape[0],)
        or not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
        or not np.isfinite(opacity).all()
        or np.any((opacity < 0) | (opacity > 1))
    ):
        raise ValueError("ground fit Gaussian parameters are invalid")
    if len(set(hints)) != 3 or any(
        x < 0 or y < 0 or x >= width or y >= height for x, y in hints
    ):
        raise ValueError("ground fit requires three distinct in-frame hints")

    world_to_camera = np.linalg.inv(transform)
    homogeneous = np.column_stack((means, np.ones(means.shape[0], dtype=np.float64)))
    camera_points = (world_to_camera @ homogeneous.T).T[:, :3]
    visible = camera_points[:, 2] > 1e-6
    pixels_h = (calibration @ camera_points.T).T
    pixels = pixels_h[:, :2] / np.maximum(pixels_h[:, 2:3], 1e-12)
    projected_scale = (
        max(float(calibration[0, 0]), float(calibration[1, 1]))
        * np.max(scales, axis=1)
        / np.maximum(camera_points[:, 2], 1e-12)
    )

    groups: list[npt.NDArray[np.int64]] = []
    group_weights: list[Float64Array] = []
    radius = settings.neighborhood_radius_pixels
    yy, xx = np.ogrid[:height, :width]
    for hint_x, hint_y in hints:
        patch = (
            (xx - hint_x) ** 2 + (yy - hint_y) ** 2 <= radius**2
        ) & (pick_buffer.opacity >= settings.minimum_opacity) & (pick_buffer.expected_depth > 0)
        depths = pick_buffer.expected_depth[patch].astype(np.float64)
        if depths.size < settings.minimum_candidates_per_hint:
            raise ValueError("ground hint has insufficient opaque depth support")
        median_depth = float(np.median(depths))
        horizontal_pairs = patch[:, 1:] & patch[:, :-1]
        vertical_pairs = patch[1:, :] & patch[:-1, :]
        discontinuities = np.concatenate(
            (
                np.abs(np.diff(pick_buffer.expected_depth, axis=1))[horizontal_pairs],
                np.abs(np.diff(pick_buffer.expected_depth, axis=0))[vertical_pairs],
            )
        )
        if (
            discontinuities.size == 0
            or float(np.max(discontinuities))
            > settings.maximum_relative_depth_spread * median_depth
        ):
            raise ValueError("ground hint crosses mixed or discontinuous depth")

        image_distance = np.linalg.norm(pixels - (hint_x + 0.5, hint_y + 0.5), axis=1)
        depth_tolerance = np.maximum(
            settings.maximum_relative_depth_spread * median_depth,
            3.0 * np.max(scales, axis=1),
        )
        selected = (
            visible
            & (opacity >= settings.minimum_opacity)
            & (image_distance <= radius + np.clip(3.0 * projected_scale, 0.0, radius))
            & (np.abs(camera_points[:, 2] - median_depth) <= depth_tolerance)
        )
        indices = np.flatnonzero(selected).astype(np.int64)
        if indices.size < settings.minimum_candidates_per_hint:
            raise ValueError("ground hint has insufficient visible Gaussian support")
        sigma = max(radius * 0.55, 1.0)
        weights = opacity[indices] * np.exp(-0.5 * (image_distance[indices] / sigma) ** 2)
        groups.append(indices)
        group_weights.append(weights)

    indices = np.concatenate(groups)
    points = means[indices]
    weights = np.concatenate(group_weights)
    group_ids = np.concatenate(
        [np.full(group.shape[0], index, dtype=np.int8) for index, group in enumerate(groups)]
    )
    scene_distance = float(np.median(camera_points[indices, 2]))
    threshold = max(settings.maximum_relative_residual * scene_distance, 1e-6)
    rng = np.random.default_rng(0)
    best_score = -1.0
    best_inliers: npt.NDArray[np.bool_] | None = None
    for _ in range(settings.ransac_iterations):
        sample = rng.choice(points.shape[0], size=3, replace=False)
        normal = np.cross(points[sample[1]] - points[sample[0]], points[sample[2]] - points[sample[0]])
        length = float(np.linalg.norm(normal))
        if length <= 1e-10:
            continue
        normal /= length
        offset = -float(np.dot(normal, points[sample[0]]))
        residuals = np.abs(points @ normal + offset)
        inliers = residuals <= threshold
        if any(np.count_nonzero(inliers & (group_ids == group)) < 3 for group in range(3)):
            continue
        score = float(np.sum(weights[inliers]))
        if score > best_score:
            best_score = score
            best_inliers = inliers
    if best_inliers is None:
        raise ValueError("ground hints do not share one supported plane")

    normal, offset = _weighted_plane(points[best_inliers], weights[best_inliers])
    residuals = np.abs(points @ normal + offset)
    inliers = residuals <= threshold
    support_counts = tuple(
        int(np.count_nonzero(inliers & (group_ids == group))) for group in range(3)
    )
    if any(count < 3 for count in support_counts):
        raise ValueError("fitted ground lacks support near every hint")
    weighted_ratio = float(np.sum(weights[inliers]) / np.sum(weights))
    rms = float(
        np.sqrt(np.average(np.square(residuals[inliers]), weights=weights[inliers]))
    )
    if weighted_ratio < 0.55 or rms > threshold:
        raise ValueError("ground plane confidence is too low")

    camera_position = transform[:3, 3]
    centroid = np.average(points[inliers], axis=0, weights=weights[inliers])
    if np.dot(normal, camera_position - centroid) < 0:
        normal = -normal
        offset = -offset
    refined = tuple(
        _ray_plane_point(hint, calibration, transform, normal, offset) for hint in hints
    )
    confidence = float(np.clip(weighted_ratio * np.exp(-rms / threshold), 0.0, 1.0))
    return GroundFitCandidate(
        plane_normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        plane_offset=float(offset),
        refined_points=(refined[0], refined[1], refined[2]),
        support_counts=(support_counts[0], support_counts[1], support_counts[2]),
        weighted_inlier_ratio=weighted_ratio,
        rms_residual=rms,
        confidence=confidence,
    )
