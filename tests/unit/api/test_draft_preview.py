from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gs_video.api.draft_preview import composite_draft, prepare_draft_composite
from gs_video.api.schemas import ApiError, DraftCompositePreviewRequest
from gs_video.camera.classify import CameraKind
from gs_video.camera.serialization import write_camera_solution
from gs_video.camera.solution import CameraSolution, SourceGroundEstimate
from gs_video.domain.models import (
    ArtifactCategory,
    ArtifactRef,
    ArtifactRole,
    Project,
    SceneSummary,
    StageName,
    StageState,
    StageStatus,
    SubjectPromptState,
    TargetGroundState,
    VideoSummary,
)


INGEST_KEY = "1" * 64
SEGMENT_KEY = "2" * 64
CAMERA_KEY = "3" * 64


def _reference(
    project: Project,
    category: ArtifactCategory,
    cache_key: str,
    member: str | None = None,
) -> ArtifactRef:
    return ArtifactRef(
        project_id=project.project_id,
        category=category,
        cache_key=cache_key,
        member=member,
    )


def _project(root: Path, *, current_solution: bool = True) -> Project:
    project = Project(name="draft")
    project.scene_ply = "source/scene.ply"
    project.workflow.source_summary = VideoSummary(
        filename="source.mp4",
        size=1,
        sha256="a" * 64,
        width=16,
        height=10,
        duration_seconds=1,
        fps="30",
        has_audio=False,
        frame_count=1,
    )
    project.workflow.scene_summary = SceneSummary(
        filename="scene.ply",
        size=1,
        sha256="b" * 64,
        gaussian_count=1,
        estimated_vram_mb=1,
    )
    project.workflow.subject_prompt = SubjectPromptState(frame_index=0, x=8, y=5)
    identity = tuple(tuple(float(value) for value in row) for row in np.eye(4))
    project.workflow.target_ground = TargetGroundState(
        scene_asset_id="source/scene.ply",
        hint_pixels=((1, 1), (2, 1), (1, 2)),
        p0_world=(0.0, 0.0, 0.0),
        p1_world=(0.0, 0.0, 1.0),
        p2_world=(1.0, 0.0, 0.0),
        plane_normal=(0.0, 1.0, 0.0),
        plane_offset=0.0,
        exploration_camera_to_world=identity,
        camera_fingerprint="c" * 64,
        preview_artifact_id="preview",
        camera_revision=1,
        pick_buffer_revision=1,
        support_counts=(20, 20, 20),
        weighted_inlier_ratio=0.9,
        rms_residual=0.01,
        confidence=0.9,
        revision=1,
        confirmed=True,
    )
    proxy_root = root / "proxies" / INGEST_KEY
    mask_root = root / "masks" / SEGMENT_KEY
    camera_root = root / "camera" / CAMERA_KEY
    proxy_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    camera_root.mkdir(parents=True)
    Image.new("RGB", (8, 5), (220, 20, 10)).save(proxy_root / "000001.jpg")
    Image.new("L", (8, 5), 255).save(mask_root / "000001.png")
    calibration = np.array(
        [[20.0, 0.0, 8.0], [0.0, 20.0, 5.0], [0.0, 0.0, 1.0]]
    )
    write_camera_solution(
        camera_root / "solution.json",
        CameraSolution(
            intrinsics=calibration,
            frame_intrinsics=[calibration],
            camera_to_world=[np.eye(4)],
            kind=CameraKind.FIXED,
            confidence=0.9,
            source_ground=(
                SourceGroundEstimate(
                    normal=(0.0, 1.0, 0.0),
                    offset=0.0,
                    anchor_frame_index=0,
                    confidence=0.9,
                    support_ratio=0.9,
                    rms_residual=0.01,
                )
                if current_solution
                else None
            ),
        ),
    )
    project.stages[StageName.INGEST] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=INGEST_KEY,
        artifacts={
            ArtifactRole.PROXY_FRAMES: _reference(
                project, ArtifactCategory.PROXIES, INGEST_KEY
            )
        },
    )
    project.stages[StageName.SEGMENT] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=SEGMENT_KEY,
        artifacts={
            ArtifactRole.SUBJECT_MASKS: _reference(
                project, ArtifactCategory.MASKS, SEGMENT_KEY
            )
        },
    )
    project.stages[StageName.SOLVE_CAMERA] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=CAMERA_KEY,
        artifacts={
            ArtifactRole.CAMERA_SOLUTION: _reference(
                project,
                ArtifactCategory.CAMERA,
                CAMERA_KEY,
                "solution.json",
            )
        },
    )
    return project


def _request(project: Project) -> DraftCompositePreviewRequest:
    return DraftCompositePreviewRequest(
        expected_project_id=project.project_id,
        request_id=1,
        maximum_width=12,
        maximum_height=7,
        gs_scale=1,
        scene_azimuth=0,
        output_crop={"x": -4, "y": -2, "width": 24, "height": 14},
    )


def test_draft_preview_uses_mapped_camera_intrinsics_and_real_crop(tmp_path: Path) -> None:
    project = _project(tmp_path)

    plan = prepare_draft_composite(project, tmp_path, None, _request(project))

    assert (plan.width, plan.height) == (12, 7)
    np.testing.assert_allclose(
        plan.camera.intrinsics(plan.width, plan.height),
        np.array([[10.0, 0.0, 6.0], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]]),
    )
    background = BytesIO()
    Image.new("RGB", (12, 7), (10, 40, 180)).save(background, format="JPEG")
    payload = composite_draft(plan, background.getvalue())
    with Image.open(BytesIO(payload)) as image:
        assert image.size == (12, 7)
        pixels = np.asarray(image.convert("RGB"))
    assert pixels[0, 0, 2] > pixels[0, 0, 0]
    assert pixels[3, 6, 0] > pixels[3, 6, 2]


def test_draft_preview_rejects_old_camera_solution_without_migrating_it(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path, current_solution=False)

    with pytest.raises(ApiError) as raised:
        prepare_draft_composite(project, tmp_path, None, _request(project))

    assert raised.value.envelope.code == "camera_solution_outdated"
