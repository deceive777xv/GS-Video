from __future__ import annotations

import json
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from gs_video.domain.errors import GsVideoError, RepairableError
from gs_video.media.export import ExportResult, _cleanup_staging, export_mp4


FIXTURE = Path(__file__).parents[2] / "fixtures" / "media" / "source.mp4"
HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def write_rgb_frames(directory: Path, count: int, *, size: tuple[int, int] = (16, 12)) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    width, height = size
    for index in range(1, count + 1):
        pixels = np.full((height, width, 3), index * 20 % 256, np.uint8)
        Image.fromarray(pixels).save(directory / f"{index:06d}.png")


def source_probe(*, has_audio: bool) -> dict[str, object]:
    streams: list[dict[str, str]] = [{"codec_type": "video"}]
    if has_audio:
        streams.append({"codec_type": "audio"})
    return {"streams": streams}


def output_probe(
    *,
    frame_count: int,
    fps: str,
    duration: str,
    has_audio: bool,
    count_field: str = "nb_read_frames",
) -> dict[str, object]:
    video = {
        "codec_type": "video",
        "avg_frame_rate": fps,
        count_field: str(frame_count),
        "duration": duration,
    }
    streams: list[dict[str, str]] = [video]
    if has_audio:
        streams.append({"codec_type": "audio"})
    return {"streams": streams, "format": {"duration": duration}}


class SuccessfulCommands:
    def __init__(self, *, has_audio: bool, frame_count: int, fps: Fraction) -> None:
        self.has_audio = has_audio
        self.frame_count = frame_count
        self.fps = fps
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, kwargs))
        if Path(command[0]).name.lower().startswith("ffmpeg"):
            Path(command[-1]).write_bytes(b"validated staging mp4")
            return subprocess.CompletedProcess(command, 0, "", "encoder diagnostics")
        target = Path(command[-1])
        if target.name.startswith("."):
            payload = output_probe(
                frame_count=self.frame_count,
                fps=f"{self.fps.numerator}/{self.fps.denominator}",
                duration=str(float(self.frame_count / self.fps)),
                has_audio=self.has_audio,
            )
        else:
            payload = source_probe(has_audio=self.has_audio)
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "probe diagnostics")


def test_export_constructs_exact_fractional_command_without_shortest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = tmp_path / "frames with spaces & symbols"
    source = tmp_path / "source (final).mp4"
    output = tmp_path / "published output.mp4"
    write_rgb_frames(frames, 3)
    source.write_bytes(b"source")
    commands = SuccessfulCommands(has_audio=True, frame_count=3, fps=Fraction(30000, 1001))
    monkeypatch.setattr(subprocess, "run", commands)

    result = export_mp4(frames, source, Fraction(30000, 1001), 3, output)

    assert result == ExportResult(
        output=output.absolute(),
        fps=Fraction(30000, 1001),
        frame_count=3,
        duration=Fraction(3003, 30000),
        has_audio=True,
    )
    ffmpeg_command, options = next(call for call in commands.calls if "ffmpeg" in call[0][0])
    assert ffmpeg_command[ffmpeg_command.index("-framerate") + 1] == "30000/1001"
    assert ffmpeg_command[ffmpeg_command.index("-start_number") + 1] == "1"
    assert ffmpeg_command[ffmpeg_command.index("-i") + 1] == str(frames / "%06d.png")
    assert str(source) in ffmpeg_command
    assert ffmpeg_command[ffmpeg_command.index("-frames:v") + 1] == "3"
    assert ffmpeg_command[ffmpeg_command.index("-map") + 1] == "0:v:0"
    assert "1:a:0?" in ffmpeg_command
    assert "apad" in ffmpeg_command
    assert "-shortest" not in ffmpeg_command
    assert Path(ffmpeg_command[-1]).parent == output.parent
    assert Path(ffmpeg_command[-1]) != output
    assert options["shell"] is False
    assert options["timeout"] == 300
    assert options["capture_output"] is True
    assert options["text"] is True
    if os.name == "nt":
        assert options["creationflags"] == subprocess.CREATE_NO_WINDOW
    else:
        assert "creationflags" not in options
    assert output.read_bytes() == b"validated staging mp4"
    assert "encoder diagnostics" in (tmp_path / ".published output.mp4.export.log").read_text()


def test_export_omits_audio_filter_when_source_has_no_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "silent.mp4"
    output = tmp_path / "out.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"source")
    commands = SuccessfulCommands(has_audio=False, frame_count=2, fps=Fraction(24, 1))
    monkeypatch.setattr(subprocess, "run", commands)

    result = export_mp4(frames, source, Fraction(24, 1), 2, output)

    ffmpeg_command = next(call[0] for call in commands.calls if "ffmpeg" in call[0][0])
    assert "1:a:0?" in ffmpeg_command
    assert "apad" not in ffmpeg_command
    assert result.has_audio is False


@pytest.mark.parametrize("fps", [0, -1, 29.97, True, "30000/1001"])
def test_export_rejects_non_positive_or_inexact_fps(tmp_path: Path, fps: object) -> None:
    with pytest.raises(RepairableError, match="fps"):
        export_mp4(tmp_path, tmp_path / "source.mp4", fps, 1, tmp_path / "out.mp4")  # type: ignore[arg-type]


@pytest.mark.parametrize("frame_count", [0, -1, True, 1.0])
def test_export_rejects_non_positive_or_inexact_frame_count(
    tmp_path: Path, frame_count: object
) -> None:
    with pytest.raises(RepairableError, match="frame_count"):
        export_mp4(
            tmp_path,
            tmp_path / "source.mp4",
            Fraction(24, 1),
            frame_count,  # type: ignore[arg-type]
            tmp_path / "out.mp4",
        )


@pytest.mark.parametrize("mutation", ["gap", "extra", "directory"])
def test_export_rejects_incomplete_or_nonordinary_inventory(
    tmp_path: Path, mutation: str
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"source")
    if mutation == "gap":
        (frames / "000002.png").unlink()
    elif mutation == "extra":
        (frames / "unexpected.txt").write_text("extra")
    else:
        (frames / "000002.png").unlink()
        (frames / "000002.png").mkdir()

    with pytest.raises(RepairableError, match="帧|清单"):
        export_mp4(frames, source, Fraction(24, 1), 2, tmp_path / "out.mp4")


@pytest.mark.parametrize("mutation", ["corrupt", "mode", "size"])
def test_export_rejects_corrupt_wrong_mode_or_wrong_size_png(
    tmp_path: Path, mutation: str
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"source")
    second = frames / "000002.png"
    if mutation == "corrupt":
        second.write_bytes(b"not png")
    elif mutation == "mode":
        Image.new("RGBA", (16, 12)).save(second)
    else:
        Image.new("RGB", (18, 12)).save(second)

    with pytest.raises(RepairableError, match="PNG|RGB|尺寸"):
        export_mp4(frames, source, Fraction(24, 1), 2, tmp_path / "out.mp4")


def test_export_rejects_odd_frame_dimensions_before_running_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    write_rgb_frames(frames, 2, size=(15, 11))
    source.write_bytes(b"source")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("must not run"))

    with pytest.raises(RepairableError, match="偶数|yuv420p"):
        export_mp4(frames, source, Fraction(24, 1), 2, tmp_path / "out.mp4")


def test_export_rejects_symlinked_frame_before_running_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"source")
    link = frames / "000002.png"
    link.unlink()
    try:
        link.symlink_to(frames / "000001.png")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("must not run"))

    with pytest.raises(RepairableError, match="链接|重解析"):
        export_mp4(frames, source, Fraction(24, 1), 2, tmp_path / "out.mp4")


def test_export_rejects_overwriting_the_read_only_source_video(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"original source")

    with pytest.raises(RepairableError, match="源视频|覆盖"):
        export_mp4(frames, source, Fraction(24, 1), 2, source)

    assert source.read_bytes() == b"original source"


@pytest.mark.parametrize("output_name", ["inside.mp4", "."])
def test_export_rejects_output_inside_or_at_the_frame_inventory_without_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output_name: str
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"source")
    before = {child.name: child.read_bytes() for child in frames.iterdir()}
    output = frames if output_name == "." else frames / output_name
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("must not probe"))

    with pytest.raises(RepairableError, match="帧目录|输出"):
        export_mp4(frames, source, Fraction(24, 1), 2, output)

    assert {child.name: child.read_bytes() for child in frames.iterdir()} == before


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, ["ffmpeg"], stderr="encode failed"),
        subprocess.TimeoutExpired(["ffmpeg"], 300, stderr="timed out diagnostics"),
        FileNotFoundError("ffmpeg missing"),
    ],
)
def test_export_maps_subprocess_failures_and_preserves_previous_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: Exception,
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"source")
    output.write_bytes(b"previous")
    calls = 0

    def fail_encoder(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 0, json.dumps(source_probe(has_audio=True)), "")
        raise failure

    monkeypatch.setattr(subprocess, "run", fail_encoder)

    with pytest.raises(GsVideoError, match="ffmpeg"):
        export_mp4(frames, source, Fraction(24, 1), 2, output)

    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".out.mp4.staging-*.mp4"))
    log = (tmp_path / ".out.mp4.export.log").read_text()
    if isinstance(failure, subprocess.TimeoutExpired):
        assert "timed out diagnostics" in log
    elif isinstance(failure, subprocess.CalledProcessError):
        assert "encode failed" in log


def test_log_write_failure_does_not_mask_the_primary_ffmpeg_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"source")
    calls = 0

    def commands(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 0, json.dumps(source_probe(has_audio=False)), "")
        raise subprocess.CalledProcessError(1, command, stderr="primary encoder failure")

    def fail_log(*args: object, **kwargs: object) -> None:
        if calls >= 2:
            raise GsVideoError("log write failed")

    monkeypatch.setattr(subprocess, "run", commands)
    monkeypatch.setattr("gs_video.media.export._append_log", fail_log)

    with pytest.raises(GsVideoError, match="ffmpeg 执行失败"):
        export_mp4(frames, source, Fraction(24, 1), 2, output)


def test_cleanup_refuses_staging_name_without_full_uuid(tmp_path: Path) -> None:
    output = tmp_path / "out.mp4"
    unowned = tmp_path / ".out.mp4.staging-not-a-full-uuid.mp4"
    unowned.write_bytes(b"do not remove")

    _cleanup_staging(unowned, output)

    assert unowned.read_bytes() == b"do not remove"


@pytest.mark.parametrize(
    "payload",
    [
        {"streams": "invalid"},
        {"streams": [{"codec_type": "audio"}], "format": {"duration": "1"}},
        output_probe(frame_count=1, fps="0/0", duration="1", has_audio=False),
        output_probe(frame_count=1, fps="24/1", duration="not-a-duration", has_audio=False),
    ],
)
def test_export_rejects_invalid_output_probe_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: dict[str, object]
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"source")
    output.write_bytes(b"previous")
    calls = 0

    def commands(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 0, json.dumps(source_probe(has_audio=False)), "")
        if calls == 2:
            Path(command[-1]).write_bytes(b"staging")
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", commands)

    with pytest.raises(GsVideoError, match="ffprobe|视频|帧率|时长"):
        export_mp4(frames, source, Fraction(24, 1), 2, output)

    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".out.mp4.staging-*.mp4"))


def test_export_accepts_nb_frames_and_r_frame_rate_probe_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"source")
    calls = 0

    def commands(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 0, json.dumps(source_probe(has_audio=False)), "")
        if calls == 2:
            Path(command[-1]).write_bytes(b"staging")
            return subprocess.CompletedProcess(command, 0, "", "")
        payload: dict[str, Any] = output_probe(
            frame_count=2,
            fps="0/0",
            duration="0.083333333",
            has_audio=False,
            count_field="nb_frames",
        )
        video = payload["streams"][0]
        video["r_frame_rate"] = "24/1"
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", commands)

    result = export_mp4(frames, source, Fraction(24, 1), 2, output)

    assert result.frame_count == 2


def test_export_falls_back_to_format_duration_when_stream_duration_is_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"source")
    calls = 0

    def commands(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 0, json.dumps(source_probe(has_audio=False)), "")
        if calls == 2:
            Path(command[-1]).write_bytes(b"staging")
            return subprocess.CompletedProcess(command, 0, "", "")
        payload = output_probe(
            frame_count=2,
            fps="24/1",
            duration="0.083333333",
            has_audio=False,
        )
        payload["streams"][0]["duration"] = "N/A-but-invalid"
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", commands)

    result = export_mp4(frames, source, Fraction(24, 1), 2, output)

    assert result.duration == Fraction(1, 12)


@pytest.mark.parametrize(
    ("frame_count", "duration", "audio", "message"),
    [
        (1, "0.083333334", False, "帧数"),
        (2, "0.2", False, "时长"),
        (2, "0.083333334", True, "音轨"),
    ],
)
def test_export_validates_frame_count_duration_and_audio_presence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frame_count: int,
    duration: str,
    audio: bool,
    message: str,
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"source")
    calls = 0

    def commands(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 0, json.dumps(source_probe(has_audio=False)), "")
        if calls == 2:
            Path(command[-1]).write_bytes(b"staging")
            return subprocess.CompletedProcess(command, 0, "", "")
        payload = output_probe(
            frame_count=frame_count,
            fps="24/1",
            duration=duration,
            has_audio=audio,
        )
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", commands)

    with pytest.raises(GsVideoError, match=message):
        export_mp4(frames, source, Fraction(24, 1), 2, output)


def test_export_rechecks_output_path_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"source")
    output.write_bytes(b"previous")
    commands = SuccessfulCommands(has_audio=False, frame_count=2, fps=Fraction(24, 1))
    monkeypatch.setattr(subprocess, "run", commands)
    checks = 0

    def becomes_unsafe(path: Path) -> bool:
        nonlocal checks
        if path == output.absolute():
            checks += 1
            return checks >= 2
        return False

    monkeypatch.setattr("gs_video.media.export.has_reparse_component", becomes_unsafe)

    with pytest.raises(GsVideoError, match="链接|重解析"):
        export_mp4(frames, source, Fraction(24, 1), 2, output)

    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".out.mp4.staging-*.mp4"))


def test_publication_failure_preserves_previous_output_and_removes_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    write_rgb_frames(frames, 2)
    source.write_bytes(b"source")
    output.write_bytes(b"previous")
    commands = SuccessfulCommands(has_audio=False, frame_count=2, fps=Fraction(24, 1))
    monkeypatch.setattr(subprocess, "run", commands)

    def fail_replace(source_path: Path, destination_path: Path) -> None:
        assert destination_path == output.absolute()
        assert source_path.parent == output.parent
        raise OSError("publication denied")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(GsVideoError, match="发布"):
        export_mp4(frames, source, Fraction(24, 1), 2, output)

    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".out.mp4.staging-*.mp4"))


def run_real_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    options: dict[str, object] = {
        "check": True,
        "capture_output": True,
        "text": True,
        "shell": False,
        "timeout": 60,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(command, **options)


def probe_real_streams(path: Path) -> dict[str, Any]:
    completed = run_real_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,avg_frame_rate,nb_read_frames,duration:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg tools are not installed")
def test_real_ffmpeg_exports_fractional_fps_with_fixture_audio(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    output = tmp_path / "fractional-with-audio.mp4"
    fps = Fraction(30000, 1001)
    write_rgb_frames(frames, 5)

    result = export_mp4(frames, FIXTURE, fps, 5, output)

    payload = probe_real_streams(output)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    assert video["nb_read_frames"] == "5"
    assert Fraction(video["avg_frame_rate"]) == fps
    assert any(stream["codec_type"] == "audio" for stream in payload["streams"])
    assert result.has_audio is True


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg tools are not installed")
def test_real_ffmpeg_succeeds_with_no_source_audio(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    silent_source = tmp_path / "silent-source.mp4"
    output = tmp_path / "silent-output.mp4"
    write_rgb_frames(frames, 4)
    run_real_command(
        ["ffmpeg", "-nostdin", "-y", "-i", str(FIXTURE), "-an", "-c:v", "copy", str(silent_source)]
    )

    result = export_mp4(frames, silent_source, Fraction(24, 1), 4, output)

    payload = probe_real_streams(output)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    assert video["nb_read_frames"] == "4"
    assert not any(stream["codec_type"] == "audio" for stream in payload["streams"])
    assert result.has_audio is False


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg tools are not installed")
def test_real_ffmpeg_short_audio_does_not_reduce_video_frame_count(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    short_audio_source = tmp_path / "short-audio-source.mp4"
    output = tmp_path / "padded-audio-output.mp4"
    write_rgb_frames(frames, 12)
    run_real_command(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x12:r=24:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.02",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-t",
            "1",
            str(short_audio_source),
        ]
    )
    source_payload = probe_real_streams(short_audio_source)
    source_audio = next(
        stream for stream in source_payload["streams"] if stream["codec_type"] == "audio"
    )
    assert Fraction(source_audio["duration"]) < Fraction(1, 2)

    export_mp4(frames, short_audio_source, Fraction(24, 1), 12, output)

    payload = probe_real_streams(output)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in payload["streams"] if stream["codec_type"] == "audio")
    assert video["nb_read_frames"] == "12"
    assert Fraction(audio["duration"]) >= Fraction(11, 24)
