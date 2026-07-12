from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from gs_video.domain.errors import UnsupportedMaterialError
from gs_video.media.ffmpeg import (
    VideoMetadata,
    parse_probe,
    probe_video,
    proxy_command,
    validate_source,
)


def test_parse_probe_preserves_fractional_frame_rate() -> None:
    metadata = parse_probe(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30000/1001",
                    "nb_frames": "300",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"duration": "10.01"},
        }
    )

    assert metadata == VideoMetadata(
        width=1920,
        height=1080,
        duration=10.01,
        fps=Fraction(30000, 1001),
        has_audio=True,
        frame_count=300,
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"streams": [], "format": {"duration": "10"}}, "视频流"),
        (
            {
                "streams": [
                    {"codec_type": "video", "height": 1080, "avg_frame_rate": "30/1"}
                ],
                "format": {"duration": "10"},
            },
            "width",
        ),
        (
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1920,
                        "height": 1080,
                        "avg_frame_rate": "0/0",
                    }
                ],
                "format": {"duration": "10"},
            },
            "帧率",
        ),
        (
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1920,
                        "height": 1080,
                        "avg_frame_rate": "not-a-rate",
                    }
                ],
                "format": {"duration": "10"},
            },
            "帧率",
        ),
        (
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1920,
                        "height": 1080,
                        "avg_frame_rate": "30/1",
                    }
                ],
                "format": {},
            },
            "duration",
        ),
    ],
)
def test_parse_probe_rejects_missing_or_invalid_stream_fields(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(UnsupportedMaterialError, match=message):
        parse_probe(payload)


@pytest.mark.parametrize("duration", [10, 30])
def test_validate_source_accepts_inclusive_duration_boundaries(duration: float) -> None:
    validate_source(VideoMetadata(width=1920, height=1080, duration=duration, fps="30/1"))


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (VideoMetadata(width=1920, height=1080, duration=9.99, fps="30/1"), "10 秒"),
        (VideoMetadata(width=1920, height=1080, duration=30.01, fps="30/1"), "30 秒"),
        (VideoMetadata(width=1921, height=1080, duration=10, fps="30/1"), "1920×1080"),
        (VideoMetadata(width=1920, height=1081, duration=10, fps="30/1"), "1920×1080"),
    ],
)
def test_validate_source_rejects_out_of_range_material(
    metadata: VideoMetadata, message: str
) -> None:
    with pytest.raises(UnsupportedMaterialError, match=message):
        validate_source(metadata)


def test_commands_keep_windows_paths_as_single_unquoted_arguments() -> None:
    source = Path(r"C:\clips & drafts\take (final); $raw.mp4")
    output = Path(r"C:\proxy frames & temp")

    command = proxy_command(source, output, max_height=360)

    assert command == [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-an",
        "-vf",
        r"scale=-2:min(360\,ih)",
        "-q:v",
        "2",
        str(output / "%06d.jpg"),
    ]
    assert '"' not in command[3]


def test_probe_video_runs_argument_list_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(r"C:\clips & drafts\source.mp4")
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 320,
                "height": 180,
                "avg_frame_rate": "10/1",
            }
        ],
        "format": {"duration": "10"},
    }
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    probe_video(source)

    assert captured == {
        "command": [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(source),
        ],
        "kwargs": {
            "check": True,
            "capture_output": True,
            "text": True,
            "shell": False,
        },
    }


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, ["ffprobe"], stderr="invalid data"),
        FileNotFoundError("ffprobe not found"),
    ],
)
def test_probe_video_maps_external_command_failures(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(UnsupportedMaterialError, match="ffprobe"):
        probe_video(Path("source.mp4"))


def test_probe_video_maps_invalid_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["ffprobe"], 0, "not json", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(UnsupportedMaterialError, match="ffprobe"):
        probe_video(Path("source.mp4"))
