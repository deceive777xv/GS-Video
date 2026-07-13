from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, NoReturn

from PIL import Image, UnidentifiedImageError

from gs_video.domain.errors import GsVideoError, RepairableError
from gs_video.segmentation.paths import has_reparse_component


_PROBE_TIMEOUT_SECONDS = 30
_EXPORT_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class ExportResult:
    output: Path
    fps: Fraction
    frame_count: int
    duration: Fraction
    has_audio: bool


@dataclass(frozen=True)
class _OutputProbe:
    fps: Fraction
    frame_count: int
    duration: Fraction
    has_audio: bool


def _validate_numbers(fps: object, frame_count: object) -> tuple[Fraction, int]:
    if not isinstance(fps, Fraction) or fps <= 0:
        raise RepairableError("fps 必须是正的精确 Fraction")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise RepairableError("frame_count 必须是正整数")
    return fps, frame_count


def _require_ordinary(path: Path, *, kind: str, directory: bool = False) -> Path:
    absolute = path.absolute()
    if has_reparse_component(absolute):
        raise RepairableError(f"{kind}不能包含链接或重解析点")
    valid = absolute.is_dir() if directory else absolute.is_file()
    if not valid:
        expected = "目录" if directory else "普通文件"
        raise RepairableError(f"{kind}必须是存在的{expected}")
    return absolute


def _validate_inventory(frames_dir: Path, frame_count: int) -> tuple[Path, tuple[int, int]]:
    directory = _require_ordinary(frames_dir, kind="帧目录", directory=True)
    expected = {f"{index:06d}.png" for index in range(1, frame_count + 1)}
    try:
        children = list(directory.iterdir())
    except OSError as exc:
        raise RepairableError("无法读取帧清单") from exc
    if {child.name for child in children} != expected:
        raise RepairableError("帧清单必须从 000001.png 连续且不能包含额外项目")

    expected_size: tuple[int, int] | None = None
    for index in range(1, frame_count + 1):
        frame = directory / f"{index:06d}.png"
        if has_reparse_component(frame):
            raise RepairableError("帧文件不能是链接或重解析点")
        if not frame.is_file():
            raise RepairableError("帧清单只能包含普通 PNG 文件")
        try:
            with Image.open(frame) as image:
                if image.format != "PNG":
                    raise RepairableError(f"帧 {frame.name} 不是 PNG")
                if image.mode != "RGB":
                    raise RepairableError(f"帧 {frame.name} 必须使用 RGB 模式")
                image.load()
                size = image.size
        except RepairableError:
            raise
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise RepairableError(f"帧 {frame.name} 是损坏的 PNG") from exc
        if size[0] <= 0 or size[1] <= 0:
            raise RepairableError(f"帧 {frame.name} 尺寸无效")
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise RepairableError("所有 PNG 帧的尺寸必须完全一致")
    assert expected_size is not None
    return directory, expected_size


def _prepare_output(output: Path) -> tuple[Path, Path]:
    absolute = output.absolute()
    if not absolute.name or absolute == absolute.parent:
        raise RepairableError("输出路径无效")
    if has_reparse_component(absolute.parent) or has_reparse_component(absolute):
        raise RepairableError("输出路径不能包含链接或重解析点")
    try:
        absolute.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GsVideoError("无法创建导出目录") from exc
    if has_reparse_component(absolute.parent) or has_reparse_component(absolute):
        raise RepairableError("输出路径不能包含链接或重解析点")
    if not absolute.parent.is_dir() or (absolute.exists() and not absolute.is_file()):
        raise RepairableError("输出必须是普通文件路径")
    log_path = absolute.parent / f".{absolute.name}.export.log"
    if has_reparse_component(log_path) or (log_path.exists() and not log_path.is_file()):
        raise RepairableError("导出日志路径不能是链接或非普通文件")
    return absolute, log_path


def _append_log(log_path: Path, label: str, stderr: object) -> None:
    if has_reparse_component(log_path) or (log_path.exists() and not log_path.is_file()):
        raise GsVideoError("导出日志路径变成了链接或非普通文件")
    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    elif isinstance(stderr, str):
        text = stderr
    elif stderr is None:
        text = ""
    else:
        text = str(stderr)
    try:
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"[{label}]\n{text}\n")
    except OSError as exc:
        raise GsVideoError("无法写入导出诊断日志") from exc


def _run_command(
    command: list[str], *, timeout: int, label: str, log_path: Path
) -> subprocess.CompletedProcess[str]:
    try:
        if os.name == "nt":
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                shell=False,
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                shell=False,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired as exc:
        _raise_command_error(
            GsVideoError(f"{label} 超时"), exc, log_path=log_path, label=label, stderr=exc.stderr
        )
    except subprocess.CalledProcessError as exc:
        _raise_command_error(
            GsVideoError(f"{label} 执行失败"),
            exc,
            log_path=log_path,
            label=label,
            stderr=exc.stderr,
        )
    except OSError as exc:
        _raise_command_error(
            GsVideoError(f"{label} 无法启动"),
            exc,
            log_path=log_path,
            label=label,
            stderr=str(exc),
        )
    _append_log(log_path, label, completed.stderr)
    return completed


def _raise_command_error(
    error: GsVideoError,
    cause: BaseException,
    *,
    log_path: Path,
    label: str,
    stderr: object,
) -> NoReturn:
    try:
        _append_log(log_path, label, stderr)
    except GsVideoError as log_error:
        error.add_note(f"diagnostic log write also failed: {log_error}")
    raise error from cause


def _load_probe_json(completed: subprocess.CompletedProcess[str], *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise GsVideoError(f"{label} 返回的 ffprobe JSON 无效") from exc
    if not isinstance(payload, dict):
        raise GsVideoError(f"{label} 返回的 ffprobe JSON 无效")
    return payload


def _streams(payload: Mapping[str, Any], *, label: str) -> list[Mapping[str, Any]]:
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list) or not all(
        isinstance(stream, dict) for stream in raw_streams
    ):
        raise GsVideoError(f"{label} 的 ffprobe streams 无效")
    return raw_streams


def _probe_source_audio(source: Path, log_path: Path) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(source),
    ]
    completed = _run_command(
        command, timeout=_PROBE_TIMEOUT_SECONDS, label="ffprobe(source)", log_path=log_path
    )
    streams = _streams(_load_probe_json(completed, label="源视频"), label="源视频")
    if not any(stream.get("codec_type") == "video" for stream in streams):
        raise GsVideoError("源文件的 ffprobe 未找到视频流")
    return any(stream.get("codec_type") == "audio" for stream in streams)


def _positive_int_field(video: Mapping[str, Any]) -> int:
    for field in ("nb_read_frames", "nb_frames"):
        raw = video.get(field)
        if raw in (None, "N/A"):
            continue
        try:
            if isinstance(raw, bool):
                raise ValueError
            parsed = int(str(raw))
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed > 0:
            return parsed
    raise GsVideoError("ffprobe 未返回有效视频帧数")


def _positive_fraction(raw: object, field: str) -> Fraction:
    try:
        parsed = Fraction(str(raw))
    except (ValueError, ZeroDivisionError) as exc:
        raise GsVideoError(f"ffprobe {field} 无效") from exc
    if parsed <= 0:
        raise GsVideoError(f"ffprobe {field} 无效")
    return parsed


def _parse_output_probe(payload: Mapping[str, Any]) -> _OutputProbe:
    streams = _streams(payload, label="导出文件")
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise GsVideoError("ffprobe 未找到导出视频流")
    fps: Fraction | None = None
    for field in ("avg_frame_rate", "r_frame_rate"):
        try:
            fps = _positive_fraction(video.get(field), "视频帧率")
        except GsVideoError:
            continue
        break
    if fps is None:
        raise GsVideoError("ffprobe 视频帧率无效")

    format_data = payload.get("format")
    format_duration = format_data.get("duration") if isinstance(format_data, dict) else None
    duration: Fraction | None = None
    for duration_raw in (video.get("duration"), format_duration):
        try:
            duration = _positive_fraction(duration_raw, "视频时长")
        except GsVideoError:
            continue
        break
    if duration is None:
        raise GsVideoError("ffprobe 视频时长无效")
    return _OutputProbe(
        fps=fps,
        frame_count=_positive_int_field(video),
        duration=duration,
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def _probe_output(path: Path, log_path: Path) -> _OutputProbe:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=codec_type,avg_frame_rate,r_frame_rate,nb_read_frames,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = _run_command(
        command, timeout=_PROBE_TIMEOUT_SECONDS, label="ffprobe(output)", log_path=log_path
    )
    return _parse_output_probe(_load_probe_json(completed, label="导出文件"))


def _duration_argument(duration: Fraction) -> str:
    with localcontext() as context:
        context.prec = 50
        value = Decimal(duration.numerator) / Decimal(duration.denominator)
    return format(value, "f")


def _ffmpeg_command(
    frames_dir: Path,
    source_video: Path,
    fps: Fraction,
    frame_count: int,
    has_audio: bool,
    staging: Path,
) -> list[str]:
    rate = f"{fps.numerator}/{fps.denominator}"
    duration = _duration_argument(Fraction(frame_count, 1) / fps)
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-framerate",
        rate,
        "-start_number",
        "1",
        "-i",
        str(frames_dir / "%06d.png"),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-frames:v",
        str(frame_count),
        "-fps_mode",
        "cfr",
        "-r",
        rate,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
    ]
    if has_audio:
        command.extend(["-af", "apad"])
    command.extend(["-t", duration, str(staging)])
    return command


def _assert_output_safe(output: Path) -> None:
    if has_reparse_component(output.parent) or has_reparse_component(output):
        raise GsVideoError("输出路径在发布前变成了链接或重解析点")
    if not output.parent.is_dir() or (output.exists() and not output.is_file()):
        raise GsVideoError("输出路径在发布前不再安全")


def _validate_output_overlap(frames_dir: Path, source_video: Path, output: Path) -> None:
    destination = output.absolute()
    if destination == source_video:
        raise RepairableError("输出不能覆盖只读源视频")
    if destination == frames_dir or frames_dir in destination.parents:
        raise RepairableError("输出不能位于帧目录内部或等于帧目录")


def _cleanup_staging(staging: Path, output: Path) -> None:
    prefix = f".{output.name}.staging-"
    if (
        staging.parent != output.parent
        or re.fullmatch(rf"{re.escape(prefix)}[0-9a-f]{{32}}\.mp4", staging.name) is None
    ):
        return
    try:
        staging.unlink(missing_ok=True)
    except OSError:
        return


def export_mp4(
    frames_dir: Path,
    source_video: Path,
    fps: Fraction,
    frame_count: int,
    output: Path,
) -> ExportResult:
    """Encode and validate a frame-exact MP4 before atomically publishing it."""
    fps, frame_count = _validate_numbers(fps, frame_count)
    frames, size = _validate_inventory(Path(frames_dir), frame_count)
    if size[0] % 2 or size[1] % 2:
        raise RepairableError("libx264 yuv420p 要求 PNG 帧的宽度和高度都是偶数")
    source = _require_ordinary(Path(source_video), kind="源视频")
    _validate_output_overlap(frames, source, Path(output))
    destination, log_path = _prepare_output(Path(output))
    staging = destination.parent / (
        f".{destination.name}.staging-{uuid.uuid4().hex}.mp4"
    )
    expected_duration = Fraction(frame_count, 1) / fps
    try:
        has_audio = _probe_source_audio(source, log_path)
        # Recheck the inventory after probing and immediately before FFmpeg opens the inputs.
        _validate_inventory(frames, frame_count)
        _run_command(
            _ffmpeg_command(frames, source, fps, frame_count, has_audio, staging),
            timeout=_EXPORT_TIMEOUT_SECONDS,
            label="ffmpeg",
            log_path=log_path,
        )
        if has_reparse_component(staging) or not staging.is_file():
            raise GsVideoError("ffmpeg 未生成普通的暂存 MP4 文件")
        probe = _probe_output(staging, log_path)
        if probe.frame_count != frame_count:
            raise GsVideoError(
                f"导出视频帧数不匹配: expected {frame_count}, got {probe.frame_count}"
            )
        if probe.fps != fps:
            raise GsVideoError(f"导出视频帧率不匹配: expected {fps}, got {probe.fps}")
        if abs(probe.duration - expected_duration) > Fraction(1, 1) / fps:
            raise GsVideoError("导出视频时长误差超过一帧")
        if probe.has_audio != has_audio:
            raise GsVideoError("导出文件音轨存在性与源视频不一致")
        _assert_output_safe(destination)
        if has_reparse_component(staging) or not staging.is_file():
            raise GsVideoError("暂存 MP4 在发布前变得不安全")
        try:
            os.replace(staging, destination)
        except OSError as exc:
            raise GsVideoError("无法原子发布已验证的 MP4") from exc
    except BaseException:
        _cleanup_staging(staging, destination)
        raise
    return ExportResult(
        output=destination,
        fps=fps,
        frame_count=frame_count,
        duration=expected_duration,
        has_audio=has_audio,
    )
