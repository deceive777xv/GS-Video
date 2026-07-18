from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class StageState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: StageStatus = StageStatus.PENDING
    cache_key: str | None = None
    output_paths: list[str] = Field(default_factory=list)
    error_code: str | None = None
    artifacts: dict[ArtifactRole, str] = Field(default_factory=dict)
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
    preview_epoch: int = Field(default=0, ge=0)
    confirmed_camera_revision: int | None = Field(default=None, ge=1)
    confirmed_preview_artifact_id: str | None = None
    foot_point: FootPointState | None = None
    motion_scale: float = Field(default=1.0, ge=0.1, le=4.0)
    preview_height: int = Field(default=540, ge=180, le=540)
    active_task_id: str | None = None
    preview: PreviewState | None = None
    export_result: ExportResultState | None = None


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 3
    project_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_video: str | None = None
    scene_ply: str | None = None
    stages: dict[StageName, StageState] = Field(default_factory=dict)
    workflow: WorkflowState = Field(default_factory=WorkflowState)
