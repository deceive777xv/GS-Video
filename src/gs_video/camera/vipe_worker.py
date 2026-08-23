from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from gs_video.camera.classify import CameraKind
from gs_video.camera.serialization import write_camera_solution
from gs_video.camera.solution import CameraSolution, SourceGroundEstimate
from gs_video.segmentation.paths import has_reparse_component


def _emit(event: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def _inventory(directory: Path, suffixes: set[str], count: int) -> tuple[Path, ...]:
    if has_reparse_component(directory) or not directory.is_dir():
        raise ValueError("ViPE input directory is unavailable")
    paths = tuple(
        sorted(
            (path for path in directory.iterdir() if path.suffix.lower() in suffixes),
            key=lambda path: int(path.stem),
        )
    )
    if (
        len(paths) != count
        or any(not path.stem.isdigit() or has_reparse_component(path) for path in paths)
        or [int(path.stem) for path in paths] != list(range(1, count + 1))
    ):
        raise ValueError("ViPE input inventory is not one complete numeric sequence")
    return paths


def _anchor_foreground_exclusion(
    frames: tuple[Path, ...], masks: tuple[Path, ...]
) -> np.ndarray:
    anchor: np.ndarray | None = None
    expected_size: tuple[int, int] | None = None
    for index, mask_path in enumerate(masks):
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
        if expected_size is None:
            expected_size = (mask.shape[1], mask.shape[0])
        elif expected_size != (mask.shape[1], mask.shape[0]):
            raise ValueError("ViPE masks have inconsistent dimensions")
        if index == 0:
            anchor = mask >= 128
    assert anchor is not None and expected_size is not None
    fraction = float(np.mean(anchor))
    if fraction <= 0 or fraction > 0.8:
        raise ValueError("foreground exclusion mask has implausible coverage")
    for frame_path in frames:
        with Image.open(frame_path) as image:
            frame_size = image.size
        if frame_size != expected_size:
            raise ValueError("ViPE frame and mask dimensions differ")
    return anchor


def _run_vipe(image_dir: Path, output: Path) -> None:
    from vipe import make_pipeline  # type: ignore[import-not-found]
    from vipe.config import parse_typed_config  # type: ignore[import-not-found]
    from vipe.streams.base import ProcessedVideoStream  # type: ignore[import-not-found]
    from vipe.streams.frame_dir_stream import FrameDirStream  # type: ignore[import-not-found]

    overrides = [
        "pipeline=default",
        f"pipeline.output.path={output}",
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_viz=false",
        "streams=frame_dir_stream",
        f"streams.base_path={image_dir}",
    ]
    args = parse_typed_config("default", hydra_args=overrides)
    pipeline = make_pipeline(args.pipeline)
    stream = ProcessedVideoStream(FrameDirStream(image_dir), []).cache(desc="Reading image frames")
    pipeline.run(stream)


def _one_artifact(root: Path, directory: str, suffix: str) -> Path:
    candidates = tuple((root / directory).glob(f"*{suffix}"))
    if len(candidates) != 1 or has_reparse_component(candidates[0]):
        raise ValueError(f"ViPE {directory} output inventory is invalid")
    return candidates[0]


def _read_depth(path: Path, count: int) -> tuple[np.ndarray, tuple[str, ...]]:
    import OpenEXR  # type: ignore[import-not-found]

    with zipfile.ZipFile(path) as archive:
        names = tuple(sorted(name for name in archive.namelist() if name.lower().endswith(".exr")))
        if len(names) != count:
            raise ValueError("ViPE depth frame count is invalid")
        with archive.open(names[0]) as stream:
            exr = OpenEXR.InputFile(stream)
            try:
                data_window = exr.header()["dataWindow"]
                width = int(data_window.max.x - data_window.min.x + 1)
                height = int(data_window.max.y - data_window.min.y + 1)
                channels = exr.channels(["Z"])
                if width <= 0 or height <= 0 or len(channels) != 1:
                    raise ValueError("ViPE anchor depth is invalid")
                depth = np.frombuffer(channels[0], dtype=np.float16)
                if depth.size != width * height:
                    raise ValueError("ViPE anchor depth is invalid")
                depth = depth.reshape((height, width))
            finally:
                exr.close()
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("ViPE anchor depth is invalid")
    return depth, names


def _source_ground(
    depth: np.ndarray,
    excluded: np.ndarray,
    calibration: np.ndarray,
    camera_to_world: np.ndarray,
) -> SourceGroundEstimate:
    height, width = depth.shape
    if excluded.shape != depth.shape:
        raise ValueError("ViPE depth and exclusion mask dimensions differ")
    yy, xx = np.mgrid[:height, :width]
    valid = (
        (yy >= int(height * 0.48))
        & ~excluded
        & np.isfinite(depth)
        & (depth > 0)
    )
    y, x = np.nonzero(valid)
    if x.size < 1000:
        raise ValueError("source ground has insufficient unmasked depth support")
    stride = max(1, x.size // 50000)
    x = x[::stride]
    y = y[::stride]
    z = depth[y, x]
    camera_points = np.column_stack(
        (
            (x + 0.5 - calibration[0, 2]) * z / calibration[0, 0],
            (y + 0.5 - calibration[1, 2]) * z / calibration[1, 1],
            z,
        )
    )
    points = (
        camera_to_world[:3, :3] @ camera_points.T
    ).T + camera_to_world[:3, 3]
    threshold = max(float(np.median(z)) * 0.012, 1e-4)
    rng = np.random.default_rng(0)
    best: np.ndarray | None = None
    best_count = 0
    for _ in range(1024):
        sample = rng.choice(points.shape[0], 3, replace=False)
        normal = np.cross(points[sample[1]] - points[sample[0]], points[sample[2]] - points[sample[0]])
        length = float(np.linalg.norm(normal))
        if length <= 1e-9:
            continue
        normal /= length
        normal_camera = camera_to_world[:3, :3].T @ normal
        if abs(float(normal_camera[1])) < 0.3:
            continue
        offset = -float(np.dot(normal, points[sample[0]]))
        inliers = np.abs(points @ normal + offset) <= threshold
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best = inliers
            best_count = count
    if best is None or best_count / points.shape[0] < 0.3:
        raise ValueError("source ground plane support is too low")
    support = points[best]
    centroid = np.mean(support, axis=0)
    _, _, vt = np.linalg.svd(support - centroid, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(np.dot(normal, centroid))
    if np.dot(normal, camera_to_world[:3, 3] - centroid) < 0:
        normal = -normal
        offset = -offset
    residual = np.abs(support @ normal + offset)
    rms = float(np.sqrt(np.mean(np.square(residual))))
    if rms > threshold:
        raise ValueError("source ground residual is too high")
    ratio = best_count / points.shape[0]
    return SourceGroundEstimate(
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        offset=offset,
        anchor_frame_index=0,
        confidence=float(np.clip(ratio * np.exp(-rms / threshold), 0, 1)),
        support_ratio=ratio,
        rms_residual=rms,
    )


def _collect_solution(vipe_root: Path, excluded: np.ndarray, count: int) -> tuple[CameraSolution, Path]:
    pose_path = _one_artifact(vipe_root, "pose", ".npz")
    intrinsics_path = _one_artifact(vipe_root, "intrinsics", ".npz")
    depth_path = _one_artifact(vipe_root, "depth", ".zip")
    with np.load(pose_path, allow_pickle=False) as archive:
        poses = np.asarray(archive["data"], dtype=np.float64)
        pose_indices = np.asarray(archive["inds"])
    with np.load(intrinsics_path, allow_pickle=False) as archive:
        calibration_values = np.asarray(archive["data"], dtype=np.float64)
        calibration_indices = np.asarray(archive["inds"])
    if (
        poses.shape != (count, 4, 4)
        or calibration_values.shape != (count, 4)
        or not np.array_equal(pose_indices, np.arange(count))
        or not np.array_equal(calibration_indices, np.arange(count))
    ):
        raise ValueError("ViPE pose or intrinsic frame authority is incomplete")
    calibrations = [
        np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        for fx, fy, cx, cy in calibration_values
    ]
    focal = np.asarray([[matrix[0, 0], matrix[1, 1]] for matrix in calibrations])
    focal_jump = float(np.max(np.abs(np.diff(focal, axis=0)) / np.maximum(focal[:-1], 1e-9)))
    if not np.isfinite(focal).all() or np.any(focal <= 0) or focal_jump > 0.25:
        raise ValueError("ViPE intrinsics failed temporal geometry audit")
    depth, _names = _read_depth(depth_path, count)
    ground = _source_ground(depth, excluded, calibrations[0], poses[0])
    confidence = float(min(1.0, ground.confidence * np.exp(-focal_jump)))
    if confidence < 0.25:
        raise ValueError("ViPE camera solve confidence is too low")
    return (
        CameraSolution(
            intrinsics=calibrations[0],
            frame_intrinsics=calibrations,
            camera_to_world=[pose for pose in poses],
            kind=CameraKind.SIX_DOF,
            confidence=confidence,
            diagnostics={
                "backend": "nvidia-vipe-1.2.0",
                "foreground_exclusion": "source_ground_only",
                "focal_max_relative_jump": focal_jump,
                "coordinate_convention": "camera-to-world; OpenCV x-right/y-down/z-forward",
            },
            source_ground=ground,
        ),
        depth_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.count < 2 or has_reparse_component(args.output) or not args.output.is_dir():
            raise ValueError("ViPE request paths or frame count are invalid")
        frames = _inventory(args.frames.absolute(), {".jpg", ".jpeg", ".png"}, args.count)
        masks = _inventory(args.masks.absolute(), {".png"}, args.count)
        if [path.stem for path in frames] != [path.stem for path in masks]:
            raise ValueError("ViPE frame and mask inventories differ")
        raw_output = args.output / f".vipe-raw-{uuid.uuid4().hex}"
        try:
            _emit({"type": "progress", "current": 1, "total": 3, "message": "读取人物遮罩用于源地面审计"})
            excluded = _anchor_foreground_exclusion(frames, masks)
            raw_output.mkdir()
            _emit({"type": "progress", "current": 2, "total": 3, "message": "ViPE 从完整视频自动解算相机与深度"})
            _run_vipe(args.frames.absolute(), raw_output)
            solution, depth_path = _collect_solution(raw_output, excluded, args.count)
            write_camera_solution(args.output / "solution.json", solution)
            shutil.copyfile(depth_path, args.output / "depth.zip")
            _emit({"type": "progress", "current": 3, "total": 3, "message": "审计源地面与相机几何"})
            _emit({"type": "complete"})
        finally:
            for owned in (raw_output,):
                if owned.parent == args.output and owned.name.startswith(".vipe-"):
                    shutil.rmtree(owned, ignore_errors=True)
        return 0
    except Exception as error:
        _emit({"type": "error", "message": str(error)[:500] or "ViPE camera solve failed"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
