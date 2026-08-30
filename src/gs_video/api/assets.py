from __future__ import annotations

from pathlib import Path
from typing import Protocol

from gs_video.api.schemas import AssetKind
from gs_video.domain.models import SceneSummary, VideoSummary
from gs_video.media.ffmpeg import VideoMetadata, probe_video, validate_source
from gs_video.scene.ply import estimate_scene_vram, load_gaussian_ply


AssetSummary = VideoSummary | SceneSummary


class AssetInspectorLike(Protocol):
    def inspect(
        self,
        kind: str,
        path: Path,
        *,
        size: int,
        sha256: str,
    ) -> AssetSummary: ...


class ExportInspectorLike(Protocol):
    def probe(self, path: Path) -> VideoMetadata: ...


class CompositePreviewInspectorLike(Protocol):
    """Read-only MP4 metadata probe used for published composite previews."""

    def probe(self, path: Path) -> VideoMetadata: ...


class ExportInspector:
    def probe(self, path: Path) -> VideoMetadata:
        return probe_video(path)


class AssetInspector:
    def inspect(
        self,
        kind: str,
        path: Path,
        *,
        size: int,
        sha256: str,
    ) -> AssetSummary:
        if kind == AssetKind.SOURCE_VIDEO.value:
            metadata = probe_video(path)
            validate_source(metadata)
            return VideoSummary(
                filename=path.name,
                size=size,
                sha256=sha256,
                width=metadata.width,
                height=metadata.height,
                duration_seconds=metadata.duration,
                fps=str(metadata.fps),
                has_audio=metadata.has_audio,
                frame_count=metadata.frame_count,
                color_primaries=metadata.color_primaries,
                color_transfer=metadata.color_transfer,
                color_matrix=metadata.color_matrix,
                color_range=metadata.color_range,
            )
        scene = load_gaussian_ply(path)
        estimated_bytes = estimate_scene_vram(scene, 1920, 1080)
        return SceneSummary(
            filename=path.name,
            size=size,
            sha256=sha256,
            gaussian_count=scene.count,
            estimated_vram_mb=max(1, (estimated_bytes + 1024**2 - 1) // 1024**2),
        )
