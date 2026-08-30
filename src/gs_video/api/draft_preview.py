from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import atan, degrees
from pathlib import Path
import numpy as np
from PIL import Image

from gs_video.api.schemas import (
    ApiError,
    DraftCompositePreviewRequest,
    SubjectMediaRole,
)
from gs_video.api.workflow import resolve_subject_media
from gs_video.camera.mapping import map_ground_aligned_trajectory
from gs_video.camera.serialization import read_camera_solution
from gs_video.composite.alpha import composite_frame_16bit
from gs_video.domain.models import (
    ArtifactRole,
    MatteRefinementSettings,
    Project,
    StageName,
    StageStatus,
)
from gs_video.scene.camera import MatrixCamera
from gs_video.storage.artifacts import ArtifactStore


@dataclass(frozen=True)
class DraftCompositePlan:
    camera: MatrixCamera
    width: int
    height: int
    frame_index: int
    source_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    foreground: bytes
    alpha: bytes
    matte_refinement: MatteRefinementSettings


def draft_authority(project: Project) -> tuple[object, ...]:
    return (
        project.project_id,
        project.scene_ply_asset_id or project.scene_ply,
        project.workflow.scene_summary,
        project.workflow.source_summary,
        project.workflow.subject_prompt,
        project.workflow.target_ground,
        project.stages.get(StageName.INGEST),
        project.stages.get(StageName.SEGMENT),
        project.stages.get(StageName.SOLVE_CAMERA),
        project.workflow.preview_epoch,
    )


def _solution_path(
    project: Project,
    project_root: Path,
    artifact_store: ArtifactStore | None,
) -> Path:
    stage = project.stages.get(StageName.SOLVE_CAMERA)
    if stage is None or stage.status is not StageStatus.SUCCEEDED:
        raise ApiError(
            409,
            code="camera_solution_required",
            category="project",
            message="Complete the current ViPE camera solve before previewing composition.",
        )
    reference = stage.artifacts.get(ArtifactRole.CAMERA_SOLUTION)
    if reference is None or reference.member != "solution.json":
        raise ApiError(
            409,
            code="camera_solution_outdated",
            category="project",
            message="This project does not contain a current ViPE camera solution. Create a new project and solve it again.",
        )
    try:
        if artifact_store is not None:
            return artifact_store.resolve(reference, directory=False)
        root = Path(project_root).resolve(strict=True)
        candidate = (root / reference.relative_path()).resolve(strict=True)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise OSError("camera solution escaped its project root")
        return candidate
    except OSError as error:
        raise ApiError(
            409,
            code="camera_solution_unavailable",
            category="project",
            message="The current ViPE camera solution is unavailable.",
            retryable=True,
        ) from error


def _preview_size(
    source: tuple[int, int], maximum: tuple[int, int]
) -> tuple[int, int]:
    width, height = source
    maximum_width, maximum_height = maximum
    scale = min(1.0, maximum_width / width, maximum_height / height)
    return max(2, int(width * scale)), max(2, int(height * scale))


def prepare_draft_composite(
    project: Project,
    project_root: Path,
    artifact_store: ArtifactStore | None,
    request: DraftCompositePreviewRequest,
) -> DraftCompositePlan:
    summary = project.workflow.source_summary
    ground = project.workflow.target_ground
    prompt = project.workflow.subject_prompt
    if summary is None or prompt is None:
        raise ApiError(
            409,
            code="subject_media_not_ready",
            category="project",
            message="Import the source video and select the subject before previewing composition.",
        )
    if ground is None or not ground.confirmed:
        raise ApiError(
            409,
            code="target_ground_required",
            category="project",
            message="Confirm the target Gaussian ground before previewing composition.",
        )
    try:
        solution = read_camera_solution(
            _solution_path(project, project_root, artifact_store)
        )
    except ApiError:
        raise
    except (OSError, ValueError) as error:
        raise ApiError(
            409,
            code="camera_solution_invalid",
            category="project",
            message="The current ViPE camera solution is invalid.",
        ) from error
    if solution.source_ground is None:
        raise ApiError(
            409,
            code="camera_solution_outdated",
            category="project",
            message="This project uses an older camera solution. Create a new project instead of migrating it.",
        )
    frame_index = prompt.frame_index if request.frame_index is None else request.frame_index
    if frame_index >= len(solution.camera_to_world):
        raise ApiError(
            409,
            code="draft_frame_unavailable",
            category="project",
            message="The requested representative frame is outside the solved trajectory.",
        )
    calibrations = solution.frame_intrinsics
    if calibrations is None:
        raise ApiError(
            409,
            code="camera_solution_outdated",
            category="project",
            message="This project uses an older camera solution. Create a new project instead of migrating it.",
        )
    try:
        mapped = map_ground_aligned_trajectory(
            solution,
            target_p0=ground.p0_world,
            target_p1=ground.p1_world,
            target_normal=ground.plane_normal,
            gs_scale=request.gs_scale,
            scene_azimuth_degrees=request.scene_azimuth,
        )
    except ValueError as error:
        raise ApiError(
            422,
            code="draft_alignment_invalid",
            category="validation",
            message="The draft trajectory alignment is invalid.",
        ) from error
    crop = request.output_crop
    width, height = _preview_size(
        (crop.width, crop.height),
        (request.maximum_width, request.maximum_height),
    )
    intrinsics = np.asarray(calibrations[frame_index], dtype=np.float64).copy()
    intrinsics[0, 2] -= crop.x
    intrinsics[1, 2] -= crop.y
    vertical_fov = degrees(
        2 * atan(summary.height / (2 * float(calibrations[frame_index][1, 1])))
    )
    foreground = resolve_subject_media(
        project, project_root, SubjectMediaRole.PROXY, frame_index
    )
    alpha = resolve_subject_media(
        project, project_root, SubjectMediaRole.ALPHA, frame_index
    )
    return DraftCompositePlan(
        camera=MatrixCamera(
            mapped[frame_index],
            vertical_fov,
            intrinsics_matrix=intrinsics,
            source_size=(crop.width, crop.height),
        ),
        width=width,
        height=height,
        frame_index=frame_index,
        source_size=(summary.width, summary.height),
        crop_box=(crop.x, crop.y, crop.x + crop.width, crop.y + crop.height),
        foreground=foreground.payload,
        alpha=alpha.payload,
        matte_refinement=request.matte_refinement.model_copy(deep=True),
    )


def composite_draft(plan: DraftCompositePlan, background_payload: bytes) -> bytes:
    try:
        with Image.open(BytesIO(background_payload)) as image:
            if image.size != (plan.width, plan.height):
                raise ValueError("draft background dimensions changed")
            background = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(BytesIO(plan.foreground)) as image:
            source = image.convert("RGB").resize(
                plan.source_size, Image.Resampling.LANCZOS
            )
            foreground_image = source.crop(plan.crop_box).resize(
                (plan.width, plan.height), Image.Resampling.LANCZOS
            )
            foreground = np.asarray(foreground_image, dtype=np.uint8)
        with Image.open(BytesIO(plan.alpha)) as image:
            source_alpha = image.convert("L").resize(
                plan.source_size, Image.Resampling.NEAREST
            )
            alpha_image = source_alpha.crop(plan.crop_box).resize(
                (plan.width, plan.height), Image.Resampling.NEAREST
            )
            alpha = np.asarray(alpha_image, dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise ApiError(
            409,
            code="draft_media_changed",
            category="project",
            message="The representative preview media changed while it was decoded.",
            retryable=True,
        ) from error
    spatial_scale = plan.width / max(plan.crop_box[2] - plan.crop_box[0], 1)
    pixels16 = composite_frame_16bit(
        foreground,
        background,
        alpha,
        plan.matte_refinement,
        spatial_scale=spatial_scale,
    )
    pixels8 = np.asarray(np.rint(pixels16.astype(np.float32) / 257.0), dtype=np.uint8)
    output = BytesIO()
    Image.fromarray(pixels8).save(
        output, format="PNG"
    )
    return output.getvalue()
