from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from gs_video.domain.errors import UnsupportedMaterialError
from gs_video.media.ffmpeg import probe_video, validate_source
from gs_video.media.ingest import detect_shot_cuts, extract_proxy_frames


FIXTURE = Path(__file__).parents[2] / "fixtures" / "media" / "source.mp4"


def write_solid_frames(directory: Path, values: list[int]) -> list[Path]:
    paths: list[Path] = []
    for index, value in enumerate(values, start=1):
        path = directory / f"{index:06d}.jpg"
        image = np.full((32, 32, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


def fake_ffmpeg_writing(values: list[int]):  # type: ignore[no-untyped-def]
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output_dir = Path(command[-1]).parent
        write_solid_frames(output_dir, values)
        return subprocess.CompletedProcess(command, 0, "", "")

    return run


def test_detects_abrupt_shot_cut_with_two_stable_frames_on_each_side(tmp_path: Path) -> None:
    frames = write_solid_frames(tmp_path, [0, 0, 255, 255])

    assert detect_shot_cuts(frames, threshold=0.65) == [2]


def test_ignores_one_frame_flash(tmp_path: Path) -> None:
    frames = write_solid_frames(tmp_path, [0, 0, 255, 0, 0])

    assert detect_shot_cuts(frames, threshold=0.65) == []


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_detect_shot_cuts_rejects_missing_or_corrupt_images(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "000001.jpg"
    if kind == "corrupt":
        path.write_bytes(b"not an image")

    with pytest.raises(UnsupportedMaterialError, match="代理帧"):
        detect_shot_cuts([path])


def test_extract_proxy_frames_removes_stale_numbered_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source with spaces & symbols.mp4"
    output_dir = tmp_path / "proxy frames"
    output_dir.mkdir()
    stale = output_dir / "000003.jpg"
    stale.write_bytes(b"stale")
    unrelated = output_dir / "keep.jpg"
    unrelated.write_bytes(b"keep")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert not stale.exists()
        captured["command"] = command
        captured["kwargs"] = kwargs
        write_solid_frames(output_dir, [32, 32])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    frames = extract_proxy_frames(source, output_dir)

    assert frames == [output_dir / "000001.jpg", output_dir / "000002.jpg"]
    assert unrelated.read_bytes() == b"keep"
    assert captured["command"][3] == str(source)  # type: ignore[index]
    assert captured["kwargs"] == {
        "check": True,
        "capture_output": True,
        "text": True,
        "shell": False,
    }


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, ["ffmpeg"], stderr="decode failed"),
        FileNotFoundError("ffmpeg not found"),
    ],
)
def test_extract_proxy_frames_maps_external_command_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: Exception
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(UnsupportedMaterialError, match="ffmpeg"):
        extract_proxy_frames(tmp_path / "source.mp4", tmp_path / "frames")


def test_extract_proxy_frames_rejects_empty_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["ffmpeg"], 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(UnsupportedMaterialError, match="代理帧"):
        extract_proxy_frames(tmp_path / "source.mp4", tmp_path / "frames")


def test_extract_proxy_frames_rejects_detected_shot_cut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subprocess, "run", fake_ffmpeg_writing([0, 0, 255, 255]))

    with pytest.raises(UnsupportedMaterialError, match="检测到镜头切换"):
        extract_proxy_frames(tmp_path / "source.mp4", tmp_path / "frames")


def test_fixture_probes_and_extracts_one_hundred_proxy_frames(tmp_path: Path) -> None:
    metadata = probe_video(FIXTURE)
    validate_source(metadata)

    frames = extract_proxy_frames(FIXTURE, tmp_path / "frames")

    assert metadata.width == 320
    assert metadata.height == 180
    assert metadata.duration == pytest.approx(10.0, abs=0.01)
    assert metadata.has_audio is True
    assert len(frames) == 100
    image = cv2.imread(str(frames[0]))
    assert image is not None
    assert image.shape[:2] == (180, 320)
