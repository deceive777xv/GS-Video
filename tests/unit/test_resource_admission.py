from gs_video.domain.models import (
    Project,
    SceneSummary,
    StageName,
    StageState,
    StageStatus,
    VideoSummary,
)
from gs_video.resource_admission import (
    estimated_remaining_pipeline_cache_bytes,
    estimated_pipeline_cache_bytes,
    estimated_render_vram_mb,
    fits_vram_budget,
)


def video_summary() -> VideoSummary:
    return VideoSummary(
        filename="source.mp4",
        size=1024,
        sha256="a" * 64,
        width=3840,
        height=2160,
        duration_seconds=120,
        fps="30/1",
        has_audio=True,
        frame_count=3600,
    )


def test_vram_admission_combines_scene_and_4k_framebuffer_at_eighty_percent() -> None:
    scene = SceneSummary(
        filename="scene.ply",
        size=1,
        sha256="b" * 64,
        gaussian_count=1,
        estimated_vram_mb=6400,
    )

    estimate = estimated_render_vram_mb(scene, 3840, 2160)

    assert estimate > scene.estimated_vram_mb
    assert fits_vram_budget(scene, 3840, 2160, (estimate * 5 + 3) // 4)
    assert not fits_vram_budget(scene, 3840, 2160, estimate)


def test_cache_estimate_includes_twenty_percent_headroom() -> None:
    video = video_summary()
    minimum_raw_frames = video.width * video.height * 3600 * 10

    assert estimated_pipeline_cache_bytes(video) > minimum_raw_frames


def test_remaining_cache_estimate_skips_reusable_dependency_outputs() -> None:
    video = video_summary()
    project = Project(name="admission")
    project.workflow.source_summary = video
    full_estimate = estimated_pipeline_cache_bytes(video)

    assert (
        estimated_remaining_pipeline_cache_bytes(project, StageName.EXPORT)
        == full_estimate
    )

    project.stages[StageName.INGEST] = StageState(status=StageStatus.SUCCEEDED)
    after_ingest = estimated_remaining_pipeline_cache_bytes(
        project, StageName.EXPORT
    )
    assert 0 < after_ingest < full_estimate

    project.stages[StageName.COMPOSITE] = StageState(
        status=StageStatus.SUCCEEDED
    )
    project.stages[StageName.POST_PROCESS] = StageState(
        status=StageStatus.SUCCEEDED
    )
    export_only = estimated_remaining_pipeline_cache_bytes(
        project, StageName.EXPORT
    )
    assert export_only == (video.size * 2 * 6 + 4) // 5

    project.stages[StageName.EXPORT] = StageState(status=StageStatus.SUCCEEDED)
    assert estimated_remaining_pipeline_cache_bytes(project, StageName.EXPORT) == 0
