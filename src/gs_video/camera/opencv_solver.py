from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite, radians, tan
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, cast

import cv2
import numpy as np
import numpy.typing as npt

from gs_video.camera.classify import (
    FAILED_PAIR_CONFIDENCE,
    FIXED_FLOW_THRESHOLD_PX,
    MAX_CONSECUTIVE_PAIR_FAILURES,
    MIN_OVERALL_CONFIDENCE,
    MIN_TRACKED_FEATURES,
    ROTATION_HOMOGRAPHY_INLIER_THRESHOLD,
    ROTATION_HOMOGRAPHY_RESIDUAL_THRESHOLD,
    SIX_DOF_CHEIRALITY_INLIER_THRESHOLD,
    CameraKind,
    classify_motion,
)
from gs_video.camera.mapping import validate_rigid_transform
from gs_video.domain.errors import UnsupportedMaterialError
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter


Float64Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CameraSolution:
    intrinsics: Float64Array
    camera_to_world: tuple[Float64Array, ...] | list[Float64Array]
    kind: CameraKind
    confidence: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
            raise ValueError("intrinsics must be a finite 3x3 matrix")
        if (
            intrinsics[0, 0] <= 0
            or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], [0, 0, 1], atol=1e-12)
        ):
            raise ValueError("intrinsics must be a valid pinhole calibration matrix")
        if not self.camera_to_world:
            raise ValueError("camera_to_world must not be empty")
        poses = tuple(
            validate_rigid_transform(pose, f"camera_to_world[{index}]")
            for index, pose in enumerate(self.camera_to_world)
        )
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and between zero and one")
        try:
            kind = CameraKind(self.kind)
        except ValueError as exc:
            raise ValueError("kind must be a valid CameraKind") from exc
        intrinsics = intrinsics.copy()
        intrinsics.setflags(write=False)
        for pose in poses:
            pose.setflags(write=False)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", poses)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "diagnostics", MappingProxyType(deepcopy(dict(self.diagnostics))))


@dataclass(frozen=True)
class _PairPose:
    world_to_camera_increment: Float64Array
    kind: CameraKind
    confidence: float
    diagnostics: Mapping[str, Any]


class _PairSolveFailure(RuntimeError):
    pass


def _project_to_rotation(matrix: npt.ArrayLike) -> Float64Array:
    candidate = np.asarray(matrix, dtype=np.float64)
    u, _, vt = np.linalg.svd(candidate)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return cast(Float64Array, np.asarray(rotation, dtype=np.float64))


def _intrinsics(width: int, height: int) -> Float64Array:
    focal = 0.5 * height / tan(radians(60.0) * 0.5)
    return np.array(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _compose_world_to_camera(
    accumulated: npt.ArrayLike, increment: npt.ArrayLike
) -> Float64Array:
    """Left-compose an adjacent OpenCV world-to-camera increment."""

    current = validate_rigid_transform(accumulated, "accumulated world-to-camera")
    delta = validate_rigid_transform(increment, "world-to-camera increment")
    return validate_rigid_transform(delta @ current, "composed world-to-camera")


def _homography_rotation(
    homography: npt.ArrayLike, intrinsics: npt.ArrayLike
) -> tuple[Float64Array, float]:
    normalized = np.linalg.inv(np.asarray(intrinsics, dtype=np.float64)) @ np.asarray(
        homography, dtype=np.float64
    ) @ np.asarray(intrinsics, dtype=np.float64)
    scale = np.cbrt(abs(np.linalg.det(normalized)))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise _PairSolveFailure("homography normalization failed")
    normalized /= scale
    rotation = _project_to_rotation(normalized)
    residual = float(np.linalg.norm(normalized - rotation, ord="fro"))
    return rotation, residual


def smooth_translations(translations: list[np.ndarray] | tuple[np.ndarray, ...]) -> list[Float64Array]:
    """Fit a local quadratic over at most five samples and evaluate at each frame."""

    if not translations:
        return []
    values = np.asarray(translations, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("translations must be finite three-vectors")
    count = len(values)
    result = np.empty_like(values)
    for index in range(count):
        start = max(0, min(index - 2, count - 5))
        stop = min(count, start + 5)
        sample_indices = np.arange(start, stop, dtype=np.float64)
        degree = min(2, len(sample_indices) - 1)
        design = np.vander(sample_indices - index, degree + 1, increasing=True)
        coefficients, *_ = np.linalg.lstsq(design, values[start:stop], rcond=None)
        result[index] = coefficients[0]
    return [point.copy() for point in result]


class OpenCvCameraSolver:
    """Solve adjacent OpenCV camera motion with a deterministic 60-degree FOV prior."""

    def solve(
        self,
        frame_paths: list[Path] | tuple[Path, ...],
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> CameraSolution:
        if not frame_paths:
            raise ValueError("frame_paths must not be empty")
        frames = self._read_frames(frame_paths)
        if len(frames) < 2:
            raise UnsupportedMaterialError("相机轨迹可信度过低：至少需要两帧")
        height, width = frames[0].shape
        intrinsics = _intrinsics(width, height)
        total = len(frames) - 1
        world_to_camera = np.eye(4, dtype=np.float64)
        poses = [np.eye(4, dtype=np.float64)]
        pairs: list[_PairPose | None] = []
        consecutive_failures = 0

        for index, (previous, current) in enumerate(zip(frames, frames[1:]), 1):
            token.raise_if_cancelled()
            try:
                pair = self._solve_pair(previous, current, intrinsics)
            except _PairSolveFailure:
                pair = None
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_PAIR_FAILURES:
                    raise UnsupportedMaterialError("相机轨迹可信度过低") from None
            else:
                consecutive_failures = 0
                assert pair is not None
                world_to_camera = _compose_world_to_camera(
                    world_to_camera, pair.world_to_camera_increment
                )
            pairs.append(pair)
            poses.append(validate_rigid_transform(np.linalg.inv(world_to_camera), "camera pose"))
            emit(index, total, f"求解相机运动 {index}/{total}")
            token.raise_if_cancelled()

        kind = self._sequence_kind(pairs)
        poses = self._simplify_poses(poses, kind)
        confidences = [
            pair.confidence if pair is not None else FAILED_PAIR_CONFIDENCE for pair in pairs
        ]
        confidence = float(np.mean(confidences))
        if confidence < MIN_OVERALL_CONFIDENCE:
            raise UnsupportedMaterialError("相机轨迹可信度过低")
        diagnostics: dict[str, Any] = {
            "intrinsics_prior": "square pixels, centered principal point, assumed 60 degree vertical FOV",
            "coordinate_convention": "camera-to-world; OpenCV x-right/y-down/z-forward",
            "pair_failures": sum(pair is None for pair in pairs),
            "pairs": tuple({} if pair is None else dict(pair.diagnostics) for pair in pairs),
        }
        return CameraSolution(intrinsics, poses, kind, confidence, diagnostics)

    @staticmethod
    def _read_frames(frame_paths: list[Path] | tuple[Path, ...]) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        expected_shape: tuple[int, int] | None = None
        for raw_path in frame_paths:
            path = Path(raw_path)
            frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if frame is None or frame.ndim != 2:
                raise UnsupportedMaterialError("相机轨迹可信度过低：无法读取代理帧")
            if expected_shape is None:
                expected_shape = frame.shape
            elif frame.shape != expected_shape:
                raise UnsupportedMaterialError("相机轨迹可信度过低：代理帧尺寸不一致")
            frames.append(frame)
        return frames

    @staticmethod
    def _solve_pair(previous: np.ndarray, current: np.ndarray, intrinsics: np.ndarray) -> _PairPose:
        points0 = cv2.goodFeaturesToTrack(previous, 2000, 0.01, 8)
        if points0 is None or len(points0) < MIN_TRACKED_FEATURES:
            raise UnsupportedMaterialError("相机轨迹可信度过低：特征不足")
        points1, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, points0, np.empty_like(points0)
        )
        if points1 is None or status is None:
            raise _PairSolveFailure("forward optical flow failed")
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, points1, np.empty_like(points1)
        )
        if back is None or back_status is None:
            raise _PairSolveFailure("backward optical flow failed")
        error = np.linalg.norm(points0 - back, axis=2).reshape(-1)
        keep = (
            status.reshape(-1).astype(bool)
            & back_status.reshape(-1).astype(bool)
            & np.isfinite(error)
            & (error < 1.0)
        )
        matched0 = np.asarray(points0[keep, 0], dtype=np.float64)
        matched1 = np.asarray(points1[keep, 0], dtype=np.float64)
        if len(matched0) < MIN_TRACKED_FEATURES:
            raise UnsupportedMaterialError("相机轨迹可信度过低：特征不足")
        flow = np.linalg.norm(matched1 - matched0, axis=1)
        median_flow = float(np.median(flow))
        tracking_support = len(matched0) / len(points0)
        try:
            homography, homography_mask = cv2.findHomography(
                matched0, matched1, cv2.RANSAC, 2.0
            )
        except cv2.error:
            homography, homography_mask = None, None
        homography_evaluated = homography is not None and homography_mask is not None
        if homography_mask is not None:
            homography_ratio = float(np.count_nonzero(homography_mask)) / len(matched0)
        else:
            homography_ratio = 0.0
        if median_flow <= FIXED_FLOW_THRESHOLD_PX:
            classification = classify_motion(
                median_flow, homography_ratio, 0.0, tracking_support
            )
            return _PairPose(
                np.eye(4), classification.kind, classification.confidence,
                {
                    "detected_features": len(points0),
                    "tracked_features": len(matched0),
                    "tracking_support": tracking_support,
                    "median_flow_px": median_flow,
                    "homography_inliers": homography_ratio if homography_evaluated else None,
                    "homography_evaluated": homography_evaluated,
                    "essential_inliers": None,
                    "essential_evaluated": False,
                    "cheirality_inliers": None,
                },
            )

        essential, essential_mask = cv2.findEssentialMat(
            matched0, matched1, intrinsics, cv2.RANSAC, 0.999, 1.0
        )
        essential_ratio = (
            float(np.count_nonzero(essential_mask)) / len(matched0)
            if essential is not None and essential_mask is not None
            else 0.0
        )
        classification = classify_motion(median_flow, homography_ratio, essential_ratio)
        increment = np.eye(4, dtype=np.float64)
        cheirality_ratio = 0.0

        final_kind = classification.kind
        final_confidence = classification.confidence
        homography_rotation: Float64Array | None = None
        homography_rotation_residual: float | None = None
        if homography is not None and homography_ratio >= ROTATION_HOMOGRAPHY_INLIER_THRESHOLD:
            homography_rotation, homography_rotation_residual = _homography_rotation(
                homography, intrinsics
            )

        if (
            homography_rotation is not None
            and homography_rotation_residual is not None
            and homography_rotation_residual <= ROTATION_HOMOGRAPHY_RESIDUAL_THRESHOLD
        ):
            increment[:3, :3] = homography_rotation
            final_kind = CameraKind.ROTATION
            final_confidence = homography_ratio
        elif classification.kind is CameraKind.SIX_DOF and essential is not None and essential_mask is not None:
            try:
                inliers, rotation, translation, pose_mask = cv2.recoverPose(
                    essential, matched0, matched1, intrinsics, mask=essential_mask.copy()
                )
            except cv2.error:
                inliers = 0
                pose_mask = None
                rotation = np.eye(3)
                translation = np.zeros((3, 1))
            cheirality_ratio = float(inliers) / len(matched0)
            if pose_mask is not None and cheirality_ratio >= SIX_DOF_CHEIRALITY_INLIER_THRESHOLD:
                increment[:3, :3] = _project_to_rotation(rotation)
                direction = np.asarray(translation, dtype=np.float64).reshape(3)
                norm = np.linalg.norm(direction)
                if not np.isfinite(norm) or norm <= 1e-12:
                    raise _PairSolveFailure("pose recovery returned invalid translation")
                increment[:3, 3] = direction / norm
            elif homography is not None and homography_ratio >= ROTATION_HOMOGRAPHY_INLIER_THRESHOLD:
                assert homography_rotation is not None
                increment[:3, :3] = homography_rotation
                final_kind = CameraKind.ROTATION
                final_confidence = homography_ratio
            else:
                raise _PairSolveFailure("pose recovery had insufficient cheirality inliers")
        elif homography is not None and homography_ratio >= ROTATION_HOMOGRAPHY_INLIER_THRESHOLD:
            assert homography_rotation is not None
            increment[:3, :3] = homography_rotation
        else:
            raise _PairSolveFailure("neither essential nor homography model was credible")

        increment = validate_rigid_transform(increment, "world-to-camera increment")
        diagnostics: dict[str, Any] = {
            "detected_features": len(points0),
            "tracked_features": len(matched0),
            "tracking_support": tracking_support,
            "median_flow_px": median_flow,
            "homography_inliers": homography_ratio,
            "homography_evaluated": homography_evaluated,
            "homography_rotation_residual": homography_rotation_residual,
            "essential_inliers": essential_ratio,
            "essential_evaluated": essential is not None and essential_mask is not None,
            "cheirality_inliers": cheirality_ratio,
        }
        return _PairPose(increment, final_kind, final_confidence, diagnostics)

    @staticmethod
    def _sequence_kind(pairs: list[_PairPose | None]) -> CameraKind:
        kinds = {pair.kind for pair in pairs if pair is not None}
        if CameraKind.SIX_DOF in kinds:
            return CameraKind.SIX_DOF
        if CameraKind.ROTATION in kinds:
            return CameraKind.ROTATION
        return CameraKind.FIXED

    @staticmethod
    def _simplify_poses(poses: list[Float64Array], kind: CameraKind) -> list[Float64Array]:
        if kind is CameraKind.FIXED:
            return [poses[0].copy() for _ in poses]
        if kind is CameraKind.ROTATION:
            result = [pose.copy() for pose in poses]
            for pose in result:
                pose[:3, 3] = 0.0
            return result
        translations = smooth_translations([pose[:3, 3] for pose in poses])
        result = [pose.copy() for pose in poses]
        for pose, translation in zip(result, translations, strict=True):
            pose[:3, 3] = translation
        return result
