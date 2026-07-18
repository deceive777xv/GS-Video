from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from gs_video.domain.errors import UnsupportedMaterialError
from gs_video.media.ffmpeg import proxy_command


_PROXY_FRAME_NAME = re.compile(r"^\d{6}\.jpg$")
_SOURCE_FRAME_NAME = re.compile(r"^\d{6}\.png$")


def source_frame_command(source: Path, output_dir: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        str(output_dir / "%06d.png"),
    ]


def _consecutive_frames(paths: list[Path], suffix: str) -> None:
    expected = [f"{index:06d}.{suffix}" for index in range(1, len(paths) + 1)]
    if [path.name for path in paths] != expected:
        raise UnsupportedMaterialError("ffmpeg 生成的帧序列必须从 000001 开始连续编号")


def extract_source_frames(source: Path, output_dir: Path) -> list[Path]:
    """Decode every source-video frame into an unscaled consecutive RGB PNG inventory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_file() and _SOURCE_FRAME_NAME.fullmatch(child.name):
            child.unlink()

    try:
        subprocess.run(
            source_frame_command(source, output_dir),
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise UnsupportedMaterialError("ffmpeg 无法生成全分辨率源帧") from exc

    frame_paths = sorted(
        child
        for child in output_dir.iterdir()
        if child.is_file() and _SOURCE_FRAME_NAME.fullmatch(child.name)
    )
    if not frame_paths:
        raise UnsupportedMaterialError("ffmpeg 未生成全分辨率源帧")
    _consecutive_frames(frame_paths, "png")
    expected_size: tuple[int, int] | None = None
    try:
        for path in frame_paths:
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGB":
                    raise UnsupportedMaterialError("全分辨率源帧必须是 RGB PNG")
                if expected_size is None:
                    expected_size = image.size
                elif image.size != expected_size:
                    raise UnsupportedMaterialError("全分辨率源帧尺寸不一致")
    except (OSError, UnidentifiedImageError) as exc:
        raise UnsupportedMaterialError("无法读取全分辨率源帧") from exc
    return frame_paths


def normalized_hsv_histogram(image: NDArray[np.uint8] | None) -> NDArray[np.float32]:
    if image is None or image.size == 0:
        raise UnsupportedMaterialError("无法读取代理帧")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cast(
        NDArray[np.float32],
        cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [30, 32, 32],
            [0, 180, 0, 256, 0, 256],
        ),
    )
    normalized = np.empty_like(histogram, dtype=np.float32)
    cv2.normalize(histogram, normalized, alpha=1.0, norm_type=cv2.NORM_L1)
    return normalized


def detect_shot_cuts(frame_paths: list[Path], threshold: float = 0.65) -> list[int]:
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    histograms = []
    for path in frame_paths:
        image = cast(NDArray[np.uint8] | None, cv2.imread(str(path), cv2.IMREAD_COLOR))
        histograms.append(normalized_hsv_histogram(image))
    distances = [
        cv2.compareHist(
            histograms[index - 1],
            histograms[index],
            cv2.HISTCMP_BHATTACHARYYA,
        )
        for index in range(1, len(histograms))
    ]

    return [
        index
        for index, distance in enumerate(distances, start=1)
        if distance >= threshold
        and index >= 2
        and index + 1 < len(histograms)
        and distances[index - 2] < threshold
        and distances[index] < threshold
    ]


def extract_proxy_frames(
    source: Path,
    output_dir: Path,
    max_height: int = 540,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_file() and _PROXY_FRAME_NAME.fullmatch(child.name):
            child.unlink()

    command = proxy_command(source, output_dir, max_height=max_height)
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise UnsupportedMaterialError("ffmpeg 无法生成代理帧") from exc

    frame_paths = sorted(
        child
        for child in output_dir.iterdir()
        if child.is_file() and _PROXY_FRAME_NAME.fullmatch(child.name)
    )
    if not frame_paths:
        raise UnsupportedMaterialError("ffmpeg 未生成代理帧")
    if detect_shot_cuts(frame_paths):
        raise UnsupportedMaterialError("检测到镜头切换")
    return frame_paths
