from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from gs_video.domain.errors import UnsupportedMaterialError


@dataclass(frozen=True, init=False)
class VideoMetadata:
    width: int
    height: int
    duration: float
    fps: Fraction
    has_audio: bool
    frame_count: int | None
    color_primaries: str | None
    color_transfer: str | None
    color_matrix: str | None
    color_range: str | None

    def __init__(
        self,
        width: int,
        height: int,
        duration: float,
        fps: Fraction | str,
        has_audio: bool = False,
        frame_count: int | None = None,
        color_primaries: str | None = None,
        color_transfer: str | None = None,
        color_matrix: str | None = None,
        color_range: str | None = None,
    ) -> None:
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "fps", Fraction(fps))
        object.__setattr__(self, "has_audio", has_audio)
        object.__setattr__(self, "frame_count", frame_count)
        object.__setattr__(self, "color_primaries", color_primaries)
        object.__setattr__(self, "color_transfer", color_transfer)
        object.__setattr__(self, "color_matrix", color_matrix)
        object.__setattr__(self, "color_range", color_range)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError
    parsed = int(value)
    if parsed <= 0:
        raise ValueError
    return parsed


def parse_probe(payload: Mapping[str, Any]) -> VideoMetadata:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise UnsupportedMaterialError("ffprobe 缺少 streams 字段")

    video_stream = next(
        (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise UnsupportedMaterialError("ffprobe 未找到视频流")

    try:
        width = _positive_int(video_stream["width"], "width")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedMaterialError("ffprobe 视频流的 width 无效") from exc
    try:
        height = _positive_int(video_stream["height"], "height")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedMaterialError("ffprobe 视频流的 height 无效") from exc

    try:
        fps = Fraction(str(video_stream["avg_frame_rate"]))
        if fps <= 0:
            raise ValueError
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise UnsupportedMaterialError("ffprobe 视频流的帧率无效") from exc

    format_data = payload.get("format")
    try:
        if not isinstance(format_data, dict):
            raise TypeError
        duration = float(format_data["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedMaterialError("ffprobe format 的 duration 无效") from exc

    raw_frame_count = video_stream.get("nb_frames")
    frame_count: int | None = None
    if raw_frame_count not in (None, "N/A"):
        try:
            frame_count = _positive_int(raw_frame_count, "nb_frames")
        except (TypeError, ValueError, OverflowError) as exc:
            raise UnsupportedMaterialError("ffprobe 视频流的 nb_frames 无效") from exc

    return VideoMetadata(
        width=width,
        height=height,
        duration=duration,
        fps=fps,
        has_audio=any(
            isinstance(stream, dict) and stream.get("codec_type") == "audio"
            for stream in streams
        ),
        frame_count=frame_count,
        color_primaries=video_stream.get("color_primaries"),
        color_transfer=video_stream.get("color_transfer"),
        color_matrix=video_stream.get("color_space"),
        color_range=video_stream.get("color_range"),
    )


def validate_source(metadata: VideoMetadata) -> None:
    if not math.isfinite(metadata.duration):
        raise UnsupportedMaterialError("视频时长必须是有限数值")
    if metadata.duration < 10:
        raise UnsupportedMaterialError("视频时长不能短于 10 秒")
    if metadata.duration > 120:
        raise UnsupportedMaterialError("视频时长不能超过 120 秒")
    if metadata.width <= 0 or metadata.height <= 0:
        raise UnsupportedMaterialError("视频分辨率无效")
    if metadata.width > 3840 or metadata.height > 2160:
        raise UnsupportedMaterialError("视频分辨率不能超过 3840×2160")
    if metadata.fps <= 0:
        raise UnsupportedMaterialError("视频帧率无效")
    values = {
        "primaries": metadata.color_primaries,
        "transfer": metadata.color_transfer,
        "matrix": metadata.color_matrix,
    }
    unsupported = {
        "primaries": {"bt2020"},
        "transfer": {"smpte2084", "arib-std-b67", "smpte428"},
        "matrix": {"bt2020nc", "bt2020c", "ictcp"},
    }
    if any(
        value is not None and value.lower() in unsupported[field]
        for field, value in values.items()
    ):
        raise UnsupportedMaterialError("首版只支持 SDR Rec.709，不能导入 HDR/BT.2020 视频")
    unspecified = {None, "", "unknown", "unspecified", "reserved"}
    for field, value in values.items():
        normalized = None if value is None else value.lower()
        if normalized not in unspecified and normalized != "bt709":
            raise UnsupportedMaterialError(
                f"首版不支持源视频的 {field} 色彩标记: {value}"
            )


def probe_video(path: Path) -> VideoMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise UnsupportedMaterialError("ffprobe 无法探测视频") from exc
    if not isinstance(payload, dict):
        raise UnsupportedMaterialError("ffprobe 返回的 JSON 无效")
    return parse_probe(payload)


def proxy_command(source: Path, output_dir: Path, max_height: int = 540) -> list[str]:
    if max_height <= 0:
        raise ValueError("max_height must be positive")
    if max_height > 540:
        raise ValueError("max_height must not exceed the MVP maximum of 540")
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-an",
        "-vf",
        f"scale=-2:min({max_height}\\,ih)",
        "-q:v",
        "2",
        str(output_dir / "%06d.jpg"),
    ]
