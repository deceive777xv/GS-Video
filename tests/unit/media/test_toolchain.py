from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from gs_video.domain.errors import GsVideoError
from gs_video.media.toolchain import executable_name, resolve_media_tools


def write_tool(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / executable_name(name)
    path.write_bytes(b"locked local tool")
    return path


def clear_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GS_VIDEO_FFMPEG", raising=False)
    monkeypatch.delenv("GS_VIDEO_FFPROBE", raising=False)


def test_resolver_prefers_explicit_configured_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured_ffmpeg = write_tool(tmp_path / "configured", "ffmpeg")
    configured_ffprobe = write_tool(tmp_path / "configured", "ffprobe")
    write_tool(tmp_path / ".cache" / "ffmpeg" / "bin", "ffmpeg")
    write_tool(tmp_path / ".cache" / "ffmpeg" / "bin", "ffprobe")
    monkeypatch.setenv("GS_VIDEO_FFMPEG", str(configured_ffmpeg))
    monkeypatch.setenv("GS_VIDEO_FFPROBE", str(configured_ffprobe))

    tools = resolve_media_tools(project_root=tmp_path)

    assert tools.ffmpeg == configured_ffmpeg.resolve()
    assert tools.ffprobe == configured_ffprobe.resolve()
    assert tools.source == "configured"


def test_resolver_prefers_project_local_cache_over_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clear_overrides(monkeypatch)
    local_ffmpeg = write_tool(tmp_path / ".cache" / "ffmpeg" / "bin", "ffmpeg")
    local_ffprobe = write_tool(tmp_path / ".cache" / "ffmpeg" / "bin", "ffprobe")
    monkeypatch.setattr(shutil, "which", lambda name: str(tmp_path / f"system-{name}"))

    tools = resolve_media_tools(project_root=tmp_path)

    assert tools.ffmpeg == local_ffmpeg.resolve()
    assert tools.ffprobe == local_ffprobe.resolve()
    assert tools.source == "project-cache"


def test_resolver_uses_resolved_system_paths_when_no_local_tools_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clear_overrides(monkeypatch)
    system_ffmpeg = write_tool(tmp_path / "system", "ffmpeg")
    system_ffprobe = write_tool(tmp_path / "system", "ffprobe")
    paths = {"ffmpeg": system_ffmpeg, "ffprobe": system_ffprobe}
    monkeypatch.setattr(shutil, "which", lambda name: str(paths[name]))

    tools = resolve_media_tools(project_root=tmp_path)

    assert tools.ffmpeg == system_ffmpeg.resolve()
    assert tools.ffprobe == system_ffprobe.resolve()
    assert tools.source == "system-path"


def test_resolver_rejects_incomplete_explicit_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clear_overrides(monkeypatch)
    monkeypatch.setenv("GS_VIDEO_FFMPEG", str(write_tool(tmp_path, "ffmpeg")))

    with pytest.raises(GsVideoError, match="同时配置"):
        resolve_media_tools(project_root=tmp_path)


def test_resolver_fails_when_tools_are_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clear_overrides(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(GsVideoError, match="FFmpeg|ffmpeg"):
        resolve_media_tools(project_root=tmp_path)


def test_executable_name_matches_platform() -> None:
    suffix = ".exe" if os.name == "nt" else ""

    assert executable_name("ffmpeg") == f"ffmpeg{suffix}"
