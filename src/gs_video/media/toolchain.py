from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from gs_video.domain.errors import GsVideoError


@dataclass(frozen=True)
class MediaTools:
    ffmpeg: Path
    ffprobe: Path
    source: str


def executable_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _ordinary_tool(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise GsVideoError(f"{label} 工具路径不存在: {path}") from exc
    if not resolved.is_file():
        raise GsVideoError(f"{label} 工具路径不是普通文件: {resolved}")
    return resolved


def _project_cache_roots(project_root: Path) -> list[Path]:
    roots = [project_root]
    if project_root.parent.name == ".worktrees":
        roots.append(project_root.parent.parent)
    override = os.environ.get("GS_VIDEO_CACHE_DIR")
    cache_roots = [Path(override)] if override else []
    cache_roots.extend(root / ".cache" for root in roots)
    return cache_roots


def _local_tools(project_root: Path) -> MediaTools | None:
    for cache_root in _project_cache_roots(project_root):
        binary_dir = cache_root / "ffmpeg" / "bin"
        ffmpeg = binary_dir / executable_name("ffmpeg")
        ffprobe = binary_dir / executable_name("ffprobe")
        if ffmpeg.exists() and ffprobe.exists():
            return MediaTools(
                ffmpeg=_ordinary_tool(ffmpeg, "ffmpeg"),
                ffprobe=_ordinary_tool(ffprobe, "ffprobe"),
                source="project-cache",
            )
    return None


def resolve_media_tools(*, project_root: Path | None = None) -> MediaTools:
    """Resolve configured, project-cached, or system FFmpeg tools without downloading."""
    configured_ffmpeg = os.environ.get("GS_VIDEO_FFMPEG")
    configured_ffprobe = os.environ.get("GS_VIDEO_FFPROBE")
    if bool(configured_ffmpeg) != bool(configured_ffprobe):
        raise GsVideoError("GS_VIDEO_FFMPEG 与 GS_VIDEO_FFPROBE 必须同时配置")
    if configured_ffmpeg and configured_ffprobe:
        return MediaTools(
            ffmpeg=_ordinary_tool(Path(configured_ffmpeg), "ffmpeg"),
            ffprobe=_ordinary_tool(Path(configured_ffprobe), "ffprobe"),
            source="configured",
        )

    root = (project_root or Path(__file__).resolve().parents[3]).absolute()
    local = _local_tools(root)
    if local is not None:
        return local

    ffmpeg_found = shutil.which("ffmpeg")
    ffprobe_found = shutil.which("ffprobe")
    if ffmpeg_found is None or ffprobe_found is None:
        raise GsVideoError("找不到 FFmpeg/ffprobe；未配置且项目缓存和 PATH 均缺失")
    return MediaTools(
        ffmpeg=_ordinary_tool(Path(ffmpeg_found), "ffmpeg"),
        ffprobe=_ordinary_tool(Path(ffprobe_found), "ffprobe"),
        source="system-path",
    )
