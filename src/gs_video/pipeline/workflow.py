from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from gs_video.domain.contracts import Stage, StageResult
from gs_video.domain.models import Project, StageName, StageStatus
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter
from gs_video.pipeline.runner import PipelineRunner, SaveProject


DEPENDENCIES: dict[StageName, tuple[StageName, ...]] = {
    StageName.INGEST: (),
    StageName.SEGMENT: (StageName.INGEST,),
    StageName.SOLVE_CAMERA: (StageName.INGEST,),
    StageName.MAP_TRAJECTORY: (StageName.SOLVE_CAMERA,),
    StageName.RENDER: (StageName.MAP_TRAJECTORY,),
    StageName.COMPOSITE: (StageName.SEGMENT, StageName.RENDER),
    StageName.EXPORT: (StageName.COMPOSITE,),
}


class ChangeKind(StrEnum):
    SOURCE_VIDEO = "source_video"
    SUBJECT_PROMPT = "subject_prompt"
    TARGET_CAMERA = "target_camera"
    MOTION_SCALE = "motion_scale"
    EDGE_SETTINGS = "edge_settings"
    EXPORT_SETTINGS = "export_settings"


class RenderCacheNamespace(StrEnum):
    PREVIEW = "preview"
    FINAL = "final"


class WorkflowStageService(Protocol):
    def run(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult: ...


class WorkflowRenderService(Protocol):
    def run(
        self,
        project: Project,
        namespace: RenderCacheNamespace,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult: ...


@dataclass(frozen=True)
class _DelegatingStage:
    service: WorkflowStageService
    name: StageName = field(init=False)

    def execute(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        return self.service.run(project, token, emit)


class IngestStage(_DelegatingStage):
    name: StageName = StageName.INGEST


class SegmentStage(_DelegatingStage):
    name: StageName = StageName.SEGMENT


class SolveCameraStage(_DelegatingStage):
    name: StageName = StageName.SOLVE_CAMERA


class MapTrajectoryStage(_DelegatingStage):
    name: StageName = StageName.MAP_TRAJECTORY


class CompositeStage(_DelegatingStage):
    name: StageName = StageName.COMPOSITE


class ExportStage(_DelegatingStage):
    name: StageName = StageName.EXPORT


class RenderStage:
    name: StageName = StageName.RENDER

    def __init__(
        self,
        service: WorkflowRenderService,
        namespace: RenderCacheNamespace,
    ) -> None:
        self.service = service
        self.namespace = namespace

    def execute(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        return self.service.run(project, self.namespace, token, emit)


@dataclass(frozen=True)
class WorkflowServices:
    media_ingest: WorkflowStageService
    segmenter: WorkflowStageService
    camera_solver: WorkflowStageService
    trajectory_mapper: WorkflowStageService
    renderer: WorkflowRenderService
    compositor: WorkflowStageService
    exporter: WorkflowStageService


def discard_project(project: Project) -> None:
    pass


def build_mvp_workflow(
    services: WorkflowServices,
    project: Project,
    save: SaveProject = discard_project,
) -> PipelineRunner:
    stages: dict[StageName, Stage] = {
        StageName.INGEST: IngestStage(services.media_ingest),
        StageName.SEGMENT: SegmentStage(services.segmenter),
        StageName.SOLVE_CAMERA: SolveCameraStage(services.camera_solver),
        StageName.MAP_TRAJECTORY: MapTrajectoryStage(services.trajectory_mapper),
        StageName.RENDER: RenderStage(services.renderer, RenderCacheNamespace.FINAL),
        StageName.COMPOSITE: CompositeStage(services.compositor),
        StageName.EXPORT: ExportStage(services.exporter),
    }
    return PipelineRunner(
        project,
        stages,
        save=save,
        dependencies=DEPENDENCIES,
        reuse_succeeded=True,
    )


INVALIDATION_ROOT: dict[ChangeKind, tuple[StageName, ...]] = {
    ChangeKind.SOURCE_VIDEO: (StageName.INGEST,),
    ChangeKind.SUBJECT_PROMPT: (StageName.SEGMENT,),
    ChangeKind.TARGET_CAMERA: (StageName.MAP_TRAJECTORY,),
    ChangeKind.MOTION_SCALE: (StageName.MAP_TRAJECTORY,),
    ChangeKind.EDGE_SETTINGS: (StageName.COMPOSITE,),
    ChangeKind.EXPORT_SETTINGS: (StageName.EXPORT,),
}


def invalidate_from(project: Project, changed_stage: StageName) -> Project:
    invalidated = {changed_stage}
    pending = [changed_stage]

    while pending:
        dependency = pending.pop()
        for stage, dependencies in DEPENDENCIES.items():
            if dependency in dependencies and stage not in invalidated:
                invalidated.add(stage)
                pending.append(stage)

    for stage in invalidated:
        state = project.stages.get(stage)
        if state is not None:
            state.status = StageStatus.STALE
            state.cache_key = None
            state.error_code = None

    return project


def invalidate_for_change(project: Project, change: ChangeKind) -> Project:
    for root in INVALIDATION_ROOT[change]:
        invalidate_from(project, root)
    return project
