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
    SOURCE_DEPTH = "source_depth"
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


class TargetGroundState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_asset_id: str
    hint_pixels: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    p0_world: tuple[float, float, float]
    p1_world: tuple[float, float, float]
    p2_world: tuple[float, float, float]
    plane_normal: tuple[float, float, float]
    plane_offset: float = Field(allow_inf_nan=False)
    exploration_camera_to_world: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    camera_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    preview_artifact_id: str
    camera_revision: int = Field(ge=1)
    pick_buffer_revision: int = Field(ge=1)
    support_counts: tuple[int, int, int]
    weighted_inlier_ratio: float = Field(ge=0, le=1, allow_inf_nan=False)
    rms_residual: float = Field(ge=0, allow_inf_nan=False)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    revision: int = Field(ge=1)
    confirmed: bool = False

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        import numpy as np

        if len(set(self.hint_pixels)) != 3 or any(
            component < 0 for point in self.hint_pixels for component in point
        ):
            raise ValueError("target ground requires three distinct nonnegative hints")
        if any(count < 3 for count in self.support_counts):
            raise ValueError("target ground requires support near every hint")
        points = tuple(
            np.asarray(point, dtype=np.float64)
            for point in (self.p0_world, self.p1_world, self.p2_world)
        )
        normal = np.asarray(self.plane_normal, dtype=np.float64)
        camera = np.asarray(self.exploration_camera_to_world, dtype=np.float64)
        if (
            any(point.shape != (3,) or not np.all(np.isfinite(point)) for point in points)
            or normal.shape != (3,)
            or not np.all(np.isfinite(normal))
            or not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6)
            or camera.shape != (4, 4)
            or not np.all(np.isfinite(camera))
            or not np.allclose(camera[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
            or not np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(camera[:3, :3]), 1.0, atol=1e-6)
        ):
            raise ValueError("target ground contains invalid plane or camera geometry")
        if np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) <= 1e-9:
            raise ValueError("target ground refined points must not be collinear")
        if any(
            not np.isclose(float(np.dot(normal, point)) + self.plane_offset, 0.0, atol=1e-5)
            for point in points
        ):
            raise ValueError("target ground refined points must lie on its fitted plane")
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


class OutputCropState(BaseModel):
    """Fixed output viewport expressed in observed-source pixel coordinates."""

    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=-32768, le=32768)
    y: int = Field(ge=-32768, le=32768)
    width: int = Field(ge=2, le=3840, multiple_of=2)
    height: int = Field(ge=2, le=2160, multiple_of=2)


class WorkflowState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_summary: VideoSummary | None = None
    scene_summary: SceneSummary | None = None
    subject_prompt: SubjectPromptState | None = None
    target_camera: CameraPose | None = None
    exploration_camera: ExplorationCameraPose | None = None
    preview_epoch: int = Field(default=0, ge=0)
    target_ground_generation: int = Field(default=0, ge=0)
    target_ground: TargetGroundState | None = None
    gs_scale: float = Field(default=1.0, ge=0.001, le=1000, allow_inf_nan=False)
    scene_azimuth: float = Field(default=0.0, ge=-180, lt=180, allow_inf_nan=False)
    output_crop: OutputCropState | None = None
    preview_height: int = Field(default=540, ge=180, le=540)
    active_task_id: str | None = None
    preview: PreviewState | None = None
    export_result: ExportResultState | None = None

    @model_validator(mode="after")
    def validate_target_ground_authority(self) -> Self:
        self.target_ground_generation = max(
            self.target_ground_generation,
            0 if self.target_ground is None else self.target_ground.revision,
        )
        return self


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 8
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
