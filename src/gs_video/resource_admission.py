from __future__ import annotations

import math
import shutil
from fractions import Fraction
from pathlib import Path

from gs_video.domain.models import (
    Project,
    SceneSummary,
    StageName,
    StageStatus,
    VideoSummary,
)
from gs_video.pipeline.workflow import DEPENDENCIES


BASELINE_RENDER_PIXELS = 1920 * 1080
FRAMEBUFFER_BYTES_PER_PIXEL = 24
DISK_HEADROOM_NUMERATOR = 6
DISK_HEADROOM_DENOMINATOR = 5


def estimated_render_vram_mb(
    scene: SceneSummary, width: int, height: int
) -> int:
    extra_bytes = max(0, width * height - BASELINE_RENDER_PIXELS) * (
        FRAMEBUFFER_BYTES_PER_PIXEL
    )
    return scene.estimated_vram_mb + (extra_bytes + 1024**2 - 1) // 1024**2


def fits_vram_budget(
    scene: SceneSummary, width: int, height: int, budget_mb: int
) -> bool:
    return estimated_render_vram_mb(scene, width, height) * 5 <= budget_mb * 4


def _frame_count(video: VideoSummary) -> int:
    frame_count = video.frame_count
    if frame_count is None:
        frame_count = math.ceil(video.duration_seconds * float(Fraction(video.fps)))
    return frame_count


def _raw_stage_cache_bytes(video: VideoSummary, stage: StageName) -> int:
    frame_count = _frame_count(video)
    full_resolution_frame_bytes = video.width * video.height * frame_count
    proxy_height = min(540, video.height)
    proxy_width = max(1, math.ceil(video.width * proxy_height / video.height))
    proxies = proxy_width * proxy_height * frame_count * 3
    estimates = {
        StageName.INGEST: full_resolution_frame_bytes * 3 + proxies,
        StageName.SEGMENT: full_resolution_frame_bytes,
        StageName.SOLVE_CAMERA: 0,
        StageName.MAP_TRAJECTORY: 0,
        StageName.RENDER: full_resolution_frame_bytes * 3,
        StageName.COMPOSITE: full_resolution_frame_bytes * 3,
        StageName.EXPORT: video.size * 2,
    }
    return estimates[stage]


def _with_disk_headroom(raw_estimate: int) -> int:
    return (
        raw_estimate * DISK_HEADROOM_NUMERATOR
        + DISK_HEADROOM_DENOMINATOR
        - 1
    ) // DISK_HEADROOM_DENOMINATOR


def estimated_pipeline_cache_bytes(video: VideoSummary) -> int:
    raw_estimate = sum(_raw_stage_cache_bytes(video, stage) for stage in StageName)
    return _with_disk_headroom(raw_estimate)


def bytes_with_disk_headroom(size: int) -> int:
    if type(size) is not int or size < 0:
        raise ValueError("size must be a non-negative integer")
    return _with_disk_headroom(size)


def cache_has_capacity(root: Path, video: VideoSummary) -> tuple[bool, int, int]:
    required = estimated_pipeline_cache_bytes(video)
    available = shutil.disk_usage(root).free
    return available >= required, required, available


def estimated_remaining_pipeline_cache_bytes(
    project: Project, target: StageName
) -> int:
    video = project.workflow.source_summary
    if video is None:
        return 0

    pending = [target]
    required_stages: set[StageName] = set()
    while pending:
        stage = pending.pop()
        state = project.stages.get(stage)
        if state is not None and state.status is StageStatus.SUCCEEDED:
            continue
        if stage in required_stages:
            continue
        required_stages.add(stage)
        pending.extend(DEPENDENCIES[stage])
    raw_estimate = sum(
        _raw_stage_cache_bytes(video, stage) for stage in required_stages
    )
    return _with_disk_headroom(raw_estimate)


def project_cache_has_capacity(
    root: Path, project: Project, target: StageName
) -> tuple[bool, int, int]:
    required = estimated_remaining_pipeline_cache_bytes(project, target)
    available = shutil.disk_usage(root).free
    return available >= required, required, available


__all__ = [
    "bytes_with_disk_headroom",
    "cache_has_capacity",
    "estimated_remaining_pipeline_cache_bytes",
    "estimated_pipeline_cache_bytes",
    "estimated_render_vram_mb",
    "fits_vram_budget",
    "project_cache_has_capacity",
]
