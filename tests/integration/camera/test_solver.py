from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gs_video.camera.classify import CameraKind
from gs_video.camera.opencv_solver import OpenCvCameraSolver, smooth_translations
from gs_video.domain.errors import CancelledError, UnsupportedMaterialError
from gs_video.pipeline.cancellation import CancellationToken


def _texture(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((240, 320), dtype=np.uint8)
    for x, y in rng.integers([10, 10], [310, 230], size=(500, 2)):
        cv2.circle(image, (int(x), int(y)), 1, 255, -1)
    return image


def _write_frames(tmp_path: Path, frames: list[np.ndarray]) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, frame in enumerate(frames):
        path = tmp_path / f"{index:03d}.png"
        assert cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def _warp_rotation(image: np.ndarray, degrees: float) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), degrees, 1)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))


def _perspective_yaw_frames() -> list[np.ndarray]:
    image = _texture(seed=17)
    height, width = image.shape
    focal = 0.5 * height / np.tan(np.deg2rad(30))
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    yaw = np.deg2rad(2.0)
    rotation = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )
    homography = intrinsics @ rotation @ np.linalg.inv(intrinsics)
    return [
        cv2.warpPerspective(image, np.linalg.matrix_power(homography, index), (width, height))
        for index in range(3)
    ]


def _parallax_frames() -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    xyz = np.column_stack(
        (
            rng.uniform(-2, 2, 1500),
            rng.uniform(-1.4, 1.4, 1500),
            rng.uniform(4, 10, 1500),
        )
    )
    focal = 0.5 * 480 / np.tan(np.deg2rad(30))
    intrinsics = np.array([[focal, 0, 320], [0, focal, 240], [0, 0, 1]])
    frames: list[np.ndarray] = []
    for camera_x in (0.0, 0.12, 0.24):
        camera_points = xyz - [camera_x, 0, 0]
        projected = (intrinsics @ camera_points.T).T
        projected = projected[:, :2] / projected[:, 2, None]
        image = np.zeros((480, 640), dtype=np.uint8)
        for index, (x, y) in enumerate(projected):
            if 5 < x < 635 and 5 < y < 475:
                cv2.circle(image, (round(x), round(y)), 1, 128 + (index % 2) * 127, -1)
        frames.append(image)
    return frames


def test_pure_fixed_sequence_is_solved_without_essential_translation(tmp_path: Path) -> None:
    image = _texture()

    solution = OpenCvCameraSolver().solve(
        _write_frames(tmp_path, [image, image.copy(), image.copy()]),
        lambda *_: None,
        CancellationToken(),
    )

    assert solution.kind is CameraKind.FIXED
    assert len(solution.camera_to_world) == 3
    for frame_pose in solution.camera_to_world:
        np.testing.assert_allclose(frame_pose, np.eye(4), atol=1e-8)
    assert "60" in solution.diagnostics["intrinsics_prior"]


def test_fixed_pair_uses_measured_homography_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _texture()

    def measured_homography(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(args[0])
        mask = np.zeros((len(points), 1), dtype=np.uint8)
        mask[: len(points) // 2] = 1
        return np.eye(3), mask

    def essential_must_be_unevaluated(*args: object, **kwargs: object) -> None:
        raise AssertionError("fixed pair should not require an essential matrix")

    monkeypatch.setattr(cv2, "findHomography", measured_homography)
    monkeypatch.setattr(cv2, "findEssentialMat", essential_must_be_unevaluated)

    solution = OpenCvCameraSolver().solve(
        _write_frames(tmp_path, [image, image.copy()]), lambda *_: None, CancellationToken()
    )

    pair = solution.diagnostics["pairs"][0]
    assert pair["homography_inliers"] == pytest.approx(0.5, abs=0.01)
    assert pair["homography_evaluated"] is True
    assert pair["essential_inliers"] is None
    assert pair["essential_evaluated"] is False
    assert 0.55 <= solution.confidence < 1.0


def test_fixed_pair_marks_failed_homography_as_unevaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _texture()

    def fail_homography(*args: object, **kwargs: object) -> None:
        raise cv2.error("homography failed")

    monkeypatch.setattr(cv2, "findHomography", fail_homography)

    solution = OpenCvCameraSolver().solve(
        _write_frames(tmp_path, [image, image.copy()]), lambda *_: None, CancellationToken()
    )

    pair = solution.diagnostics["pairs"][0]
    assert pair["homography_inliers"] is None
    assert pair["homography_evaluated"] is False
    assert pair["essential_inliers"] is None


def test_pure_rotation_sequence_has_zero_translation(tmp_path: Path) -> None:
    image = _texture()
    frames = [_warp_rotation(image, angle) for angle in (0.0, 2.0, 4.0)]

    solution = OpenCvCameraSolver().solve(
        _write_frames(tmp_path, frames), lambda *_: None, CancellationToken()
    )

    assert solution.kind is CameraKind.ROTATION
    assert any(not np.allclose(pose[:3, :3], np.eye(3), atol=1e-3) for pose in solution.camera_to_world[1:])
    for frame_pose in solution.camera_to_world:
        np.testing.assert_allclose(frame_pose[:3, 3], 0.0, atol=1e-10)


def test_calibrated_perspective_yaw_is_rotation_without_translation(tmp_path: Path) -> None:
    solution = OpenCvCameraSolver().solve(
        _write_frames(tmp_path, _perspective_yaw_frames()), lambda *_: None, CancellationToken()
    )

    assert solution.kind is CameraKind.ROTATION
    for frame_pose in solution.camera_to_world:
        np.testing.assert_array_equal(frame_pose[:3, 3], np.zeros(3))


def test_parallax_sequence_is_solved_as_six_dof(tmp_path: Path) -> None:
    solution = OpenCvCameraSolver().solve(
        _write_frames(tmp_path, _parallax_frames()), lambda *_: None, CancellationToken()
    )

    assert solution.kind is CameraKind.SIX_DOF
    assert solution.confidence >= 0.55
    assert np.linalg.norm(solution.camera_to_world[-1][:3, 3]) > 0.5


def test_solver_emits_each_pair_and_checks_cancellation(tmp_path: Path) -> None:
    image = _texture()
    events: list[tuple[int, int, str]] = []
    token = CancellationToken()

    def emit(current: int, total: int, message: str) -> None:
        events.append((current, total, message))
        if current == 1:
            token.cancel()

    with pytest.raises(CancelledError):
        OpenCvCameraSolver().solve(_write_frames(tmp_path, [image] * 3), emit, token)

    assert events == [(1, 2, "求解相机运动 1/2")]


def test_solver_rejects_corrupt_and_mismatched_frames(tmp_path: Path) -> None:
    image = _texture()
    valid = _write_frames(tmp_path, [image])[0]
    corrupt = tmp_path / "bad.png"
    corrupt.write_bytes(b"not an image")
    mismatched = _write_frames(tmp_path / "other", [np.zeros((200, 320), dtype=np.uint8)])[0]

    for paths in ([valid, corrupt], [valid, mismatched]):
        with pytest.raises(UnsupportedMaterialError, match="相机轨迹可信度过低"):
            OpenCvCameraSolver().solve(paths, lambda *_: None, CancellationToken())


def test_solver_rejects_too_few_features(tmp_path: Path) -> None:
    blank = np.zeros((240, 320), dtype=np.uint8)

    with pytest.raises(UnsupportedMaterialError, match="特征不足"):
        OpenCvCameraSolver().solve(
            _write_frames(tmp_path, [blank, blank]), lambda *_: None, CancellationToken()
        )


def test_five_consecutive_pair_failures_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _texture()
    frames = [_warp_rotation(image, float(angle)) for angle in range(6)]
    monkeypatch.setattr(cv2, "findHomography", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(cv2, "findEssentialMat", lambda *args, **kwargs: (None, None))

    with pytest.raises(UnsupportedMaterialError, match="相机轨迹可信度过低"):
        OpenCvCameraSolver().solve(
            _write_frames(tmp_path, frames),
            lambda *_: None,
            CancellationToken(),
        )


def test_brief_pair_failure_holds_pose_and_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _texture()
    frames = [_warp_rotation(image, float(angle * 2)) for angle in range(5)]
    original_homography = cv2.findHomography
    original_essential = cv2.findEssentialMat
    calls = {"homography": 0, "essential": 0}

    def homography_once(*args: object, **kwargs: object) -> tuple[None, None] | tuple[object, object]:
        calls["homography"] += 1
        if calls["homography"] == 1:
            return None, None
        return original_homography(*args, **kwargs)

    def essential_once(*args: object, **kwargs: object) -> tuple[None, None] | tuple[object, object]:
        calls["essential"] += 1
        if calls["essential"] == 1:
            return None, None
        return original_essential(*args, **kwargs)

    monkeypatch.setattr(cv2, "findHomography", homography_once)
    monkeypatch.setattr(cv2, "findEssentialMat", essential_once)

    solution = OpenCvCameraSolver().solve(
        _write_frames(tmp_path, frames), lambda *_: None, CancellationToken()
    )

    np.testing.assert_allclose(solution.camera_to_world[1], solution.camera_to_world[0])
    assert solution.diagnostics["pair_failures"] == 1
    assert 0.55 <= solution.confidence < 1.0


def test_local_quadratic_translation_smoothing_is_deterministic_and_no_alias() -> None:
    source = [np.array([float(i), float(i * i), 0.0]) for i in range(7)]
    originals = [point.copy() for point in source]

    first = smooth_translations(source)
    second = smooth_translations(source)

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first[3], [3.0, 9.0, 0.0])
    for actual, expected in zip(source, originals, strict=True):
        np.testing.assert_array_equal(actual, expected)
