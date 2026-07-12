from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


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


class StageState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: StageStatus = StageStatus.PENDING
    cache_key: str | None = None
    output_paths: list[str] = Field(default_factory=list)
    error_code: str | None = None


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    project_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_video: str | None = None
    scene_ply: str | None = None
    stages: dict[StageName, StageState] = Field(default_factory=dict)
