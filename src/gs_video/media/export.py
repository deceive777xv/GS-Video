from __future__ import annotations

import json
import os
import stat
import subprocess
import uuid
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, NoReturn, TextIO

from PIL import Image, UnidentifiedImageError

from gs_video.domain.errors import GsVideoError, RepairableError
from gs_video.media.toolchain import MediaTools, resolve_media_tools
from gs_video.segmentation.paths import has_reparse_component


_PROBE_TIMEOUT_SECONDS = 30
_EXPORT_TIMEOUT_SECONDS = 300
_OWNERSHIP_ATTEMPTS = 8

FileIdentity = tuple[int, int]


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


def _identity_from_stat(stat: os.stat_result) -> FileIdentity:
    identity = (int(stat.st_dev), int(stat.st_ino))
    if identity == (0, 0):
        raise GsVideoError("文件系统未提供可验证的文件身份")
    return identity


def _path_identity(path: Path) -> FileIdentity:
    try:
        return _identity_from_stat(path.stat())
    except OSError as exc:
        raise GsVideoError(f"无法读取文件身份: {path}") from exc


@dataclass
class _LogSink:
    path: Path
    handle: TextIO
    directory: Path
    directory_identity: FileIdentity
    file_identity: FileIdentity
    closed: bool = False

    def write(self, label: str, stderr: object) -> None:
        if self.closed:
            raise GsVideoError("导出诊断日志已关闭")
        if isinstance(stderr, bytes):
            text = stderr.decode("utf-8", errors="replace")
        elif isinstance(stderr, str):
            text = stderr
        elif stderr is None:
            text = ""
        else:
            text = str(stderr)
        try:
            self.handle.write(f"[{label}]\n{text}\n")
            self.handle.flush()
        except (OSError, ValueError) as exc:
            raise GsVideoError("无法写入导出诊断日志") from exc

    def close(self) -> None:
        if self.closed:
            return
        failure: OSError | ValueError | None = None
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
        except (OSError, ValueError) as exc:
            failure = exc
        try:
            self.handle.close()
        except (OSError, ValueError) as exc:
            if failure is None:
                failure = exc
        self.closed = True
        if failure is not None:
            raise GsVideoError("无法持久化导出诊断日志") from failure


@dataclass
class _OwnedStaging:
    directory: Path
    directory_identity: FileIdentity
    file: Path
    file_identity: FileIdentity | None = None

    def verify_directory(self) -> None:
        if (
            has_reparse_component(self.directory)
            or not self.directory.is_dir()
            or _path_identity(self.directory) != self.directory_identity
        ):
            raise GsVideoError("暂存目录身份发生变化")

    def record_file(self, protected: set[FileIdentity]) -> None:
        self.verify_directory()
        if has_reparse_component(self.file) or not self.file.is_file():
            raise GsVideoError("ffmpeg 未生成普通的暂存 MP4 文件")
        identity = _path_identity(self.file)
        if identity in protected:
            raise GsVideoError("暂存 MP4 与受保护输入或输出身份重叠")
        self.file_identity = identity

    def verify_file(self) -> None:
        self.verify_directory()
        if (
            self.file_identity is None
            or has_reparse_component(self.file)
            or not self.file.is_file()
            or _path_identity(self.file) != self.file_identity
        ):
            raise GsVideoError("暂存 MP4 身份发生变化")

    def cleanup(self) -> None:
        try:
            self.verify_directory()
        except GsVideoError:
            return
        if self.file.exists() or self.file.is_symlink():
            if self.file_identity is None:
                return
            try:
                if (
                    has_reparse_component(self.file)
                    or not self.file.is_file()
                    or _path_identity(self.file) != self.file_identity
                ):
                    return
                self.file.unlink()
            except (GsVideoError, OSError):
                return
        try:
            self.verify_directory()
            self.directory.rmdir()
        except (GsVideoError, OSError):
            return


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
        try:
            frame_stat = frame.stat()
        except OSError as exc:
            raise RepairableError(f"无法读取帧 {frame.name} 的文件状态") from exc
        if not stat.S_ISREG(frame_stat.st_mode):
            raise RepairableError("帧清单只能包含普通 PNG 文件")
        if frame_stat.st_nlink != 1:
            raise RepairableError(f"帧 {frame.name} 不能是硬链接")
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


def _prepare_output(output: Path) -> Path:
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
    return absolute


def _protected_identities(
    frames_dir: Path, source: Path, output: Path, frame_count: int
) -> set[FileIdentity]:
    paths = [source, *(frames_dir / f"{index:06d}.png" for index in range(1, frame_count + 1))]
    if output.exists():
        paths.append(output)
    return {_path_identity(path) for path in paths}


def _verify_owned_directory(path: Path, identity: FileIdentity, label: str) -> None:
    if has_reparse_component(path) or not path.is_dir() or _path_identity(path) != identity:
        raise GsVideoError(f"{label}身份发生变化")


def _create_log_sink(output: Path, protected: set[FileIdentity]) -> _LogSink:
    logs_dir = output.parent / ".gs-video-logs"
    if has_reparse_component(output.parent) or has_reparse_component(logs_dir):
        raise RepairableError("导出日志目录不能包含链接或重解析点")
    try:
        logs_dir.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError:
        if has_reparse_component(logs_dir) or not logs_dir.is_dir():
            raise RepairableError("导出日志目录必须是普通目录")
    except OSError as exc:
        raise GsVideoError("无法创建导出日志目录") from exc
    directory_identity = _path_identity(logs_dir)
    _verify_owned_directory(logs_dir, directory_identity, "导出日志目录")

    for _attempt in range(_OWNERSHIP_ATTEMPTS):
        _verify_owned_directory(logs_dir, directory_identity, "导出日志目录")
        path = logs_dir / f"export-{uuid.uuid4().hex}.log"
        try:
            handle = path.open("x", encoding="utf-8", newline="\n")
        except FileExistsError:
            continue
        except OSError as exc:
            raise GsVideoError("无法独占创建导出日志") from exc
        try:
            file_identity: FileIdentity | None = None
            file_identity = _identity_from_stat(os.fstat(handle.fileno()))
            _verify_owned_directory(logs_dir, directory_identity, "导出日志目录")
            if (
                has_reparse_component(path)
                or not path.is_file()
                or _path_identity(path) != file_identity
                or file_identity in protected
            ):
                raise GsVideoError("新建导出日志与受保护文件身份重叠或已被替换")
            return _LogSink(
                path=path,
                handle=handle,
                directory=logs_dir,
                directory_identity=directory_identity,
                file_identity=file_identity,
            )
        except BaseException:
            handle.close()
            try:
                if (
                    not has_reparse_component(path)
                    and path.is_file()
                    and file_identity is not None
                    and _path_identity(path) == file_identity
                    and file_identity not in protected
                ):
                    path.unlink()
            except (GsVideoError, OSError):
                pass
            raise
    raise GsVideoError("导出日志 UUID 冲突，无法独占创建日志")


def _create_owned_staging(output: Path) -> _OwnedStaging:
    prefix = f".{output.name}.staging-"
    for _attempt in range(_OWNERSHIP_ATTEMPTS):
        if has_reparse_component(output.parent) or not output.parent.is_dir():
            raise GsVideoError("输出父目录在创建暂存目录前变得不安全")
        directory = output.parent / f"{prefix}{uuid.uuid4().hex}"
        try:
            directory.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError:
            continue
        except OSError as exc:
            raise GsVideoError("无法独占创建暂存目录") from exc
        identity = _path_identity(directory)
        try:
            _verify_owned_directory(directory, identity, "暂存目录")
        except BaseException:
            try:
                if directory.is_dir() and _path_identity(directory) == identity:
                    directory.rmdir()
            except (GsVideoError, OSError):
                pass
            raise
        return _OwnedStaging(
            directory=directory,
            directory_identity=identity,
            file=directory / "staging.mp4",
        )
    raise GsVideoError("暂存目录 UUID 冲突，无法独占创建 staging")


def _append_log(log_sink: _LogSink, label: str, stderr: object) -> None:
    log_sink.write(label, stderr)


def _run_command(
    command: list[str], *, timeout: int, label: str, log_sink: _LogSink
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
            GsVideoError(f"{label} 超时"),
            exc,
            log_sink=log_sink,
            label=label,
            stderr=exc.stderr,
        )
    except subprocess.CalledProcessError as exc:
        _raise_command_error(
            GsVideoError(f"{label} 执行失败"),
            exc,
            log_sink=log_sink,
            label=label,
            stderr=exc.stderr,
        )
    except OSError as exc:
        _raise_command_error(
            GsVideoError(f"{label} 无法启动"),
            exc,
            log_sink=log_sink,
            label=label,
            stderr=str(exc),
        )
    _append_log(log_sink, label, completed.stderr)
    return completed


def _raise_command_error(
    error: GsVideoError,
    cause: BaseException,
    *,
    log_sink: _LogSink,
    label: str,
    stderr: object,
) -> NoReturn:
    try:
        _append_log(log_sink, label, stderr)
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


def _probe_source_audio(source: Path, log_sink: _LogSink, tools: MediaTools) -> bool:
    command = [
        str(tools.ffprobe),
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(source),
    ]
    completed = _run_command(
        command, timeout=_PROBE_TIMEOUT_SECONDS, label="ffprobe(source)", log_sink=log_sink
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


def _probe_output(path: Path, log_sink: _LogSink, tools: MediaTools) -> _OutputProbe:
    command = [
        str(tools.ffprobe),
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
        command, timeout=_PROBE_TIMEOUT_SECONDS, label="ffprobe(output)", log_sink=log_sink
    )
    return _parse_output_probe(_load_probe_json(completed, label="导出文件"))


def _duration_argument(duration: Fraction) -> str:
    with localcontext() as context:
        context.prec = 50
        value = Decimal(duration.numerator) / Decimal(duration.denominator)
    return format(value, "f")


def _ffmpeg_command(
    tools: MediaTools,
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
        str(tools.ffmpeg),
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
    destination = _prepare_output(Path(output))
    tools = resolve_media_tools()
    protected = _protected_identities(frames, source, destination, frame_count)
    log_sink = _create_log_sink(destination, protected)
    staging: _OwnedStaging | None = None
    primary_error: BaseException | None = None
    expected_duration = Fraction(frame_count, 1) / fps
    try:
        staging = _create_owned_staging(destination)
        staging.verify_directory()
        has_audio = _probe_source_audio(source, log_sink, tools)
        staging.verify_directory()
        # Recheck the inventory after probing and immediately before FFmpeg opens the inputs.
        _validate_inventory(frames, frame_count)
        staging.verify_directory()
        try:
            _run_command(
                _ffmpeg_command(
                    tools, frames, source, fps, frame_count, has_audio, staging.file
                ),
                timeout=_EXPORT_TIMEOUT_SECONDS,
                label="ffmpeg",
                log_sink=log_sink,
            )
        except BaseException as exc:
            if staging.file.exists() or staging.file.is_symlink():
                try:
                    staging.record_file(protected | {log_sink.file_identity})
                except GsVideoError as ownership_error:
                    exc.add_note(f"staging ownership check also failed: {ownership_error}")
            raise
        staging.record_file(protected | {log_sink.file_identity})
        staging.verify_file()
        probe = _probe_output(staging.file, log_sink, tools)
        staging.verify_file()
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
        staging.verify_file()
        # Persist and close the sole log handle before publication. A log fsync/close
        # failure therefore cannot leave a newly published output reported as failed.
        log_sink.close()
        _assert_output_safe(destination)
        staging.verify_file()
        try:
            os.replace(staging.file, destination)
        except OSError as exc:
            raise GsVideoError("无法原子发布已验证的 MP4") from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if staging is not None:
            staging.cleanup()
        try:
            log_sink.close()
        except GsVideoError as log_error:
            if primary_error is not None:
                primary_error.add_note(f"diagnostic log close also failed: {log_error}")
            else:
                raise
    return ExportResult(
        output=destination,
        fps=fps,
        frame_count=frame_count,
        duration=expected_duration,
        has_audio=has_audio,
    )
