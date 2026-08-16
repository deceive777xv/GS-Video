from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite
from pathlib import PurePosixPath
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StageName(StrEnum):
    INGEST = "ingest"
    SEGMENT = "segment"
    SOLVE_CAMERA = "solve_camera"
    MAP_TRAJECTORY = "map_trajectory"
    RENDER = "render"
    COMPOSITE = "composite"
    EXPORT = "export"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALE = "stale"


class ArtifactRole(StrEnum):
    SOURCE_FRAMES = "source_frames"
    PROXY_FRAMES = "proxy_frames"
    SUBJECT_MASKS = "subject_masks"
    CAMERA_SOLUTION = "camera_solution"
    MAPPED_TRAJECTORY = "mapped_trajectory"
    RENDER_FRAMES = "render_frames"
    COMPOSITE_FRAMES = "composite_frames"
    COMPOSITE_PREVIEW = "composite_preview"
    EXPORT_VIDEO = "export_video"


class ArtifactCategory(StrEnum):
    FRAMES = "frames"
    PROXIES = "proxies"
    MASKS = "masks"
    CAMERA = "camera"
    TRAJECTORIES = "trajectories"
    RENDERS = "renders"
    COMPOSITES = "composites"
    PREVIEWS = "previews"
    EXPORTS = "exports"


class ArtifactRef(BaseModel):
    """Logical reference to one immutable project-scoped cache artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    category: ArtifactCategory
    cache_key: str
    member: str | None = None

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("artifact project_id must be a canonical UUID") from exc
        if str(parsed) != value:
            raise ValueError("artifact project_id must be a canonical UUID")
        return value

    @field_validator("cache_key")
    @classmethod
    def validate_cache_key(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("artifact cache_key must be 64 lowercase hexadecimal characters")
        return value

    @field_validator("member")
    @classmethod
    def validate_member(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or "\\" in value or ":" in value:
            raise ValueError("artifact member must be a safe relative POSIX path")
        candidate = PurePosixPath(value)
        if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
            raise ValueError("artifact member must be a safe relative POSIX path")
        normalized = candidate.as_posix()
        if normalized != value:
            raise ValueError("artifact member must be normalized")
        return value

    def relative_path(self) -> str:
        base = f"{self.category.value}/{self.cache_key}"
        return base if self.member is None else f"{base}/{self.member}"


class StageState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: StageStatus = StageStatus.PENDING
    cache_key: str | None = None
    output_paths: list[ArtifactRef] = Field(default_factory=list)
    error_code: str | None = None
    artifacts: dict[ArtifactRole, ArtifactRef] = Field(default_factory=dict)
    input_generation: int = Field(default=0, ge=0)
    run_id: str | None = None


@dataclass(frozen=True)
class StageWriteGuard:
    input_generation: int
    status: StageStatus
    run_id: str | None


@dataclass(frozen=True)
class StageWriteResult:
    project: "Project"
    applied: bool


@dataclass(frozen=True)
class StageClaimResult:
    project: "Project"
    claimed: bool


class VideoSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    size: int = Field(ge=0)
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    fps: str
    has_audio: bool
    frame_count: int | None = Field(default=None, gt=0)


class SceneSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    size: int = Field(ge=0)
    sha256: str
    gaussian_count: int = Field(gt=0)
    estimated_vram_mb: int = Field(gt=0)


class SubjectPromptState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class CameraPose(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: tuple[float, float, float]
    distance: float = Field(gt=0, allow_inf_nan=False)
    yaw: float = Field(allow_inf_nan=False)
    pitch: float = Field(gt=-90, lt=90, allow_inf_nan=False)
    fov_y_degrees: float = Field(gt=1, lt=179, allow_inf_nan=False)
    revision: int = Field(default=0, ge=0)

    @field_validator("target")
    @classmethod
    def validate_target(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if not all(isfinite(component) for component in value):
            raise ValueError("target must contain finite values")
        return value


class FootPointState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: tuple[int, int]
    world: tuple[float, float, float]
    preview_artifact_id: str
    camera_revision: int = Field(ge=1)
    pick_buffer_revision: int = Field(ge=1)


class VisibilityRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    review_frames: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end_frame < self.start_frame:
            raise ValueError("visibility range end must not precede its start")
        if any(
            frame < self.start_frame or frame > self.end_frame
            for frame in self.review_frames
        ):
            raise ValueError("visibility review frames must lie inside their range")
        return self


class SubjectVisibilityAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_asset_id: str
    segment_cache_key: str
    fully_visible_ranges: tuple[VisibilityRange, ...] = ()
    bottom_cropped_ranges: tuple[VisibilityRange, ...] = ()
    uncertain_ranges: tuple[VisibilityRange, ...] = ()
    recommended_anchor_frames: tuple[int, ...] = ()
    revision: int = Field(ge=1)


class SourcePerspectiveCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_asset_id: str
    ingest_cache_key: str
    segment_cache_key: str
    anchor_frame_index: int = Field(ge=0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    vertical_fov: float = Field(gt=1, lt=179, allow_inf_nan=False)
    horizon_line: tuple[float, float, float]
    gravity_direction_camera: tuple[float, float, float]
    revision: int = Field(ge=1)

    @field_validator("horizon_line", "gravity_direction_camera")
    @classmethod
    def validate_finite_vector(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if not all(isfinite(component) for component in value):
            raise ValueError("perspective vectors must contain finite values")
        if sum(component * component for component in value) <= 1e-18:
            raise ValueError("perspective vectors must have non-zero length")
        return value


class ExplorationCameraPose(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_to_world: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    fov_y_degrees: float = Field(gt=1, lt=179, allow_inf_nan=False)
    revision: int = Field(default=0, ge=0)

    @field_validator("camera_to_world")
    @classmethod
    def validate_camera_matrix(
        cls,
        value: tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ],
    ) -> tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]:
        import numpy as np

        matrix = np.asarray(value, dtype=np.float64)
        rotation = matrix[:3, :3]
        if (
            not np.all(np.isfinite(matrix))
            or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        ):
            raise ValueError("camera_to_world must be a finite rigid transform")
        return value


class LocalGroundAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_asset_id: str
    p0_world: tuple[float, float, float]
    p1_world: tuple[float, float, float]
    p2_world: tuple[float, float, float]
    plane_normal: tuple[float, float, float]
    plane_offset: float = Field(allow_inf_nan=False)
    frozen_camera_to_world: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    frozen_camera_fingerprint: str
    preview_artifact_id: str
    camera_revision: int = Field(ge=1)
    pick_buffer_revision: int = Field(ge=1)
    revision: int = Field(ge=1)

    @field_validator("p0_world", "p1_world", "p2_world", "plane_normal")
    @classmethod
    def validate_finite_point(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if not all(isfinite(component) for component in value):
            raise ValueError("local ground vectors must contain finite values")
        return value

    @field_validator("frozen_camera_to_world")
    @classmethod
    def validate_finite_matrix(
        cls,
        value: tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ],
    ) -> tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]:
        import numpy as np

        matrix = np.asarray(value, dtype=np.float64)
        rotation = matrix[:3, :3]
        if (
            not np.all(np.isfinite(matrix))
            or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        ):
            raise ValueError("frozen camera matrix must be a finite rigid transform")
        return value

    @model_validator(mode="after")
    def validate_plane(self) -> Self:
        import numpy as np

        p0 = np.asarray(self.p0_world, dtype=np.float64)
        p1 = np.asarray(self.p1_world, dtype=np.float64)
        p2 = np.asarray(self.p2_world, dtype=np.float64)
        normal = np.asarray(self.plane_normal, dtype=np.float64)
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6):
            raise ValueError("local ground normal must be unit length")
        triangle_normal = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(triangle_normal) <= 1e-9:
            raise ValueError("local ground points must not be collinear")
        if not np.isclose(float(np.dot(normal, p0)) + self.plane_offset, 0.0, atol=1e-6):
            raise ValueError("local ground plane offset must contain p0")
        if not np.isclose(abs(float(np.dot(normal, triangle_normal / np.linalg.norm(triangle_normal)))), 1.0, atol=1e-6):
            raise ValueError("local ground normal must match the selected triangle")
        return self


class SynthesisConstraintMode(StrEnum):
    CONTACT = "contact"
    PERSPECTIVE = "perspective"


class SubjectContactConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_asset_id: str
    ingest_cache_key: str
    segment_cache_key: str
    source_calibration_revision: int = Field(ge=1)
    anchor_frame_index: int = Field(ge=0)
    foot_pixel: tuple[int, int]
    revision: int = Field(ge=1)

    @field_validator("foot_pixel")
    @classmethod
    def validate_foot_pixel(cls, value: tuple[int, int]) -> tuple[int, int]:
        if any(component < 0 for component in value):
            raise ValueError("foot pixel must be non-negative")
        return value


class SynthesisPlacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_calibration_revision: int = Field(ge=1)
    ground_anchor_revision: int = Field(ge=1)
    source_contact_revision: int | None = Field(default=None, ge=1)
    mode: SynthesisConstraintMode
    scene_azimuth: float = Field(ge=-180, lt=180, allow_inf_nan=False)
    subject_to_scene_scale: float = Field(ge=0.25, le=4, allow_inf_nan=False)
    composition_offset_local: tuple[float, float] = (0.0, 0.0)
    anchor_camera_to_world: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    intrinsics: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    solver_cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision: int = Field(ge=1)

    @field_validator("composition_offset_local")
    @classmethod
    def validate_offset(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not all(isfinite(component) for component in value):
            raise ValueError("composition offset must contain finite values")
        return value

    @field_validator("anchor_camera_to_world", "intrinsics")
    @classmethod
    def validate_placement_matrix(
        cls, value: tuple[tuple[float, ...], ...]
    ) -> tuple[tuple[float, ...], ...]:
        if not all(isfinite(component) for row in value for component in row):
            raise ValueError("placement matrices must contain finite values")
        return value

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        import numpy as np

        camera = np.asarray(self.anchor_camera_to_world, dtype=np.float64)
        rotation = camera[:3, :3]
        if (
            camera.shape != (4, 4)
            or not np.allclose(camera[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        ):
            raise ValueError("placement camera must be a finite rigid transform")
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        if (
            intrinsics.shape != (3, 3)
            or intrinsics[0, 0] <= 0
            or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], (0.0, 0.0, 1.0), atol=1e-8)
        ):
            raise ValueError("placement intrinsics must be a valid pinhole matrix")
        if self.mode is SynthesisConstraintMode.CONTACT and self.composition_offset_local != (0.0, 0.0):
            raise ValueError("contact placement must not contain composition offset")
        if (self.mode is SynthesisConstraintMode.CONTACT) != (self.source_contact_revision is not None):
            raise ValueError("contact placement must bind exactly one confirmed source contact")
        return self


class PreviewState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_size: int = Field(ge=0)
    artifact_sha256: str
    generation: int = Field(ge=1)
    width: int = Field(gt=0, le=960)
    height: int = Field(gt=0, le=540)
    camera_revision: int = Field(ge=1)
    pick_buffer_revision: int = Field(ge=1)


class ExportResultState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    filename: str
    size: int = Field(ge=0)
    sha256: str
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    fps: str
    frame_count: int = Field(gt=0)
    has_audio: bool
    verified: bool = True


class WorkflowState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_summary: VideoSummary | None = None
    scene_summary: SceneSummary | None = None
    subject_prompt: SubjectPromptState | None = None
    target_camera: CameraPose | None = None
    exploration_camera: ExplorationCameraPose | None = None
    preview_epoch: int = Field(default=0, ge=0)
    confirmed_camera_revision: int | None = Field(default=None, ge=1)
    confirmed_preview_artifact_id: str | None = None
    foot_point: FootPointState | None = None
    subject_visibility_audit: SubjectVisibilityAudit | None = None
    source_calibration_generation: int = Field(default=0, ge=0)
    source_contact_generation: int = Field(default=0, ge=0)
    local_ground_generation: int = Field(default=0, ge=0)
    synthesis_placement_generation: int = Field(default=0, ge=0)
    source_perspective_calibration: SourcePerspectiveCalibration | None = None
    local_ground_anchor: LocalGroundAnchor | None = None
    subject_contact_constraint: SubjectContactConstraint | None = None
    synthesis_placement: SynthesisPlacement | None = None
    confirmed_synthesis_placement_revision: int | None = Field(default=None, ge=1)
    motion_scale: float = Field(default=1.0, ge=0.1, le=4.0)
    preview_height: int = Field(default=540, ge=180, le=540)
    active_task_id: str | None = None
    preview: PreviewState | None = None
    export_result: ExportResultState | None = None

    @model_validator(mode="after")
    def validate_synthesis_authority(self) -> Self:
        calibration = self.source_perspective_calibration
        contact = self.subject_contact_constraint
        ground = self.local_ground_anchor
        placement = self.synthesis_placement
        self.source_calibration_generation = max(
            self.source_calibration_generation,
            0 if calibration is None else calibration.revision,
        )
        self.source_contact_generation = max(
            self.source_contact_generation,
            0 if contact is None else contact.revision,
        )
        self.local_ground_generation = max(
            self.local_ground_generation,
            0 if ground is None else ground.revision,
        )
        self.synthesis_placement_generation = max(
            self.synthesis_placement_generation,
            0 if placement is None else placement.revision,
        )
        if contact is not None and (
            calibration is None
            or contact.source_asset_id != calibration.source_asset_id
            or contact.ingest_cache_key != calibration.ingest_cache_key
            or contact.segment_cache_key != calibration.segment_cache_key
            or contact.source_calibration_revision != calibration.revision
            or contact.anchor_frame_index != calibration.anchor_frame_index
        ):
            raise ValueError("source contact authority must match source calibration")
        if placement is not None and (
            calibration is None
            or ground is None
            or placement.source_calibration_revision != calibration.revision
            or placement.ground_anchor_revision != ground.revision
        ):
            raise ValueError("synthesis placement authority must match calibration and ground")
        if placement is not None and placement.mode is SynthesisConstraintMode.CONTACT:
            if contact is None or placement.source_contact_revision != contact.revision:
                raise ValueError("contact placement must match confirmed source contact")
        elif placement is not None and contact is not None:
            raise ValueError("perspective placement must not retain source contact")
        if self.confirmed_synthesis_placement_revision is not None and (
            placement is None
            or self.confirmed_synthesis_placement_revision != placement.revision
        ):
            raise ValueError("confirmed synthesis revision must match current placement")
        return self


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 6
    project_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_video_asset_id: str | None = None
    scene_ply_asset_id: str | None = None
    # Transitional read compatibility for schema <= 3. Catalog migration clears
    # these after the inputs have been committed to the shared asset library.
    source_video: str | None = None
    scene_ply: str | None = None
    stages: dict[StageName, StageState] = Field(default_factory=dict)
    workflow: WorkflowState = Field(default_factory=WorkflowState)

    def assert_artifact_authority(self) -> None:
        for state in self.stages.values():
            for reference in (*state.output_paths, *state.artifacts.values()):
                if reference.project_id != self.project_id:
                    raise ValueError("artifact reference belongs to another project")

    @model_validator(mode="after")
    def validate_artifact_authority(self) -> Self:
        self.assert_artifact_authority()
        return self
