from __future__ import annotations

import asyncio
import errno
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, BinaryIO, Protocol, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, Request, Response, WebSocket, status

from gs_video.api.auth import require_session
from gs_video.api.assets import AssetInspectorLike, ExportInspectorLike
from gs_video.api.events import EventBus, TaskService, serve_events
from gs_video.api.export_routes import build_export_router
from gs_video.api.preview_routes import build_composite_preview_router
from gs_video.api.schemas import (
    API_VERSION,
    ApiError,
    ApiSettings,
    AssetImportRequest,
    AssetKind,
    AssetListItem,
    AssetResponse,
    BootstrapResponse,
    CameraConfirmationRequest,
    CacheCleanupRequest,
    CacheCleanupPlanRequest,
    CameraInput,
    EnvironmentRepairSnapshot,
    HealthResponse,
    LivePreviewRequest,
    PickRequest,
    PickResponse,
    PreviewFrameRequest,
    PreviewFrameResponse,
    ProjectPatch,
    ProjectAssetSelection,
    ProjectCreate,
    ProjectRename,
    TaskCreateRequest,
    TaskSnapshot,
    StorageLayoutUpdate,
    UploadComplete,
    UploadCreateRequest,
    UploadCreated,
    UploadStatus,
    VramBudgetUpdate,
)
from gs_video.api.subject_routes import build_subject_router
from gs_video.api.uploads import UploadManager, read_bounded_body
from gs_video.api.workflow import (
    PreviewArtifactStore,
    PreviewCoordinator,
    PreviewRequestFingerprint,
    PreviewServiceLike,
    validate_pick_buffer,
    validate_subject_prompt,
)
from gs_video.domain.contracts import PickBuffer
from gs_video.domain.models import (
    CameraPose,
    FootPointState,
    PreviewState,
    Project,
    SceneSummary,
    StageName,
    StageState,
    SubjectPromptState,
    VideoSummary,
)
from gs_video.environment.doctor import EnvironmentReport
from gs_video.environment.repair import EnvironmentRepairBusyError
from gs_video.environment.vram import (
    STANDARD_VRAM_MB,
    VramBudgetManager,
    VramBudgetMode,
    VramBudgetPersistenceError,
    VramBudgetSnapshot,
    VramBudgetUnavailableError,
)
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter, discard_progress
from gs_video.pipeline.workflow import ChangeKind, invalidate_for_change
from gs_video.project.assets import (
    AssetKind as LibraryAssetKind,
    AssetLibrary,
    AssetRecord,
)
from gs_video.project.catalog import ProjectCatalog, ProjectSummary
from gs_video.project.manager import (
    ActiveProjectManager,
    ActiveProjectRequiredError,
)
from gs_video.scene.camera import OrbitCamera
from gs_video.storage.artifacts import ArtifactStore
from gs_video.storage.layout import (
    CacheCleanupPlan,
    CacheCleanupResult,
    StorageLayoutManager,
    StorageLayoutSnapshot,
)
from gs_video.resource_admission import (
    fits_vram_budget,
    project_cache_has_capacity,
)


_DESKTOP_ORIGINS = {
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
}


def _require_desktop_path_import(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin not in _DESKTOP_ORIGINS:
        raise ApiError(
            403,
            code="desktop_import_required",
            category="authorization",
            message="Local path import is available only in the desktop application.",
        )


class ProjectRepositoryLike(Protocol):
    @property
    def root(self) -> Path: ...

    def load(self) -> Project: ...

    def save(self, project: Project) -> None: ...

    def update(self, mutation: Callable[[Project], None]) -> Project: ...


class EnvironmentDoctorLike(Protocol):
    def check(self) -> EnvironmentReport: ...


class EnvironmentRepairLike(Protocol):
    def snapshot(self) -> EnvironmentRepairSnapshot: ...

    def start(self) -> EnvironmentRepairSnapshot: ...

    def cancel(self) -> EnvironmentRepairSnapshot: ...

    def is_busy(self) -> bool: ...

    async def shutdown(self) -> None: ...


class PipelineRunnerLike(Protocol):
    def run(
        self,
        name: StageName,
        token: CancellationToken,
        emit: ProgressEmitter = discard_progress,
    ) -> StageState: ...

    def supports(self, name: StageName) -> bool: ...


class WorkerRegistryLike(Protocol):
    async def terminate_all(self) -> None: ...


@dataclass(frozen=True)
class ApiServices:
    project_repository: ProjectRepositoryLike
    environment_doctor: EnvironmentDoctorLike
    pipeline_runner: PipelineRunnerLike
    worker_registry: WorkerRegistryLike
    preview_service: PreviewServiceLike | None = None
    asset_inspector: AssetInspectorLike | None = None
    export_inspector: ExportInspectorLike | None = None
    environment_repair: EnvironmentRepairLike | None = None
    vram_budget: VramBudgetManager | None = None
    project_catalog: ProjectCatalog | None = None
    asset_library: AssetLibrary | None = None
    project_manager: ActiveProjectManager | None = None
    preview_artifacts: Any | None = None
    upload_root: Path | None = None
    artifact_store: ArtifactStore | None = None
    storage_layout: StorageLayoutManager | None = None


def _services(request: Request) -> ApiServices:
    return cast(ApiServices, request.app.state.services)


def _settings(request: Request) -> ApiSettings:
    return cast(ApiSettings, request.app.state.settings)


def _task_service(request: Request) -> TaskService:
    return cast(TaskService, request.app.state.task_service)


def _upload_manager(request: Request) -> UploadManager:
    return cast(UploadManager, request.app.state.upload_manager)


def _environment_repair(request: Request) -> EnvironmentRepairLike:
    repair = _services(request).environment_repair
    if repair is None:
        raise ApiError(
            503,
            code="environment_repair_unavailable",
            category="environment",
            message="Environment repair is not available in this local service.",
            retryable=False,
        )
    return repair


_VRAM_BLOCKING_STAGES = {
    StageName.SEGMENT,
    StageName.RENDER,
    StageName.COMPOSITE,
    StageName.EXPORT,
}


def _runtime_change_lock(request: Request) -> asyncio.Lock:
    return cast(asyncio.Lock, request.app.state.runtime_change_lock)


def _vram_budget(request: Request) -> VramBudgetManager:
    budget = _services(request).vram_budget
    if budget is None:
        raise ApiError(
            503,
            code="vram_budget_unavailable",
            category="runtime",
            message="VRAM budget configuration is unavailable in this local service.",
            retryable=True,
        )
    return budget


def _storage_layout(request: Request) -> StorageLayoutManager:
    layout = _services(request).storage_layout
    if layout is None:
        raise ApiError(
            503,
            code="storage_layout_unavailable",
            category="runtime",
            message="Storage layout configuration is unavailable.",
            retryable=True,
        )
    return layout


def _storage_blocked_reason(
    request: Request, *, include_update: bool = True
) -> str | None:
    if _task_service(request).is_busy():
        return "task_active"
    repair = _services(request).environment_repair
    if repair is not None and repair.is_busy():
        return "environment_repair_active"
    if include_update and _runtime_change_lock(request).locked():
        return "update_in_progress"
    return None


def _vram_blocked_reason(
    request: Request, *, include_update: bool = True
) -> str | None:
    if _task_service(request).is_busy(_VRAM_BLOCKING_STAGES):
        return "gpu_task_active"
    repair = _services(request).environment_repair
    if repair is not None and repair.is_busy():
        return "environment_repair_active"
    if include_update and _runtime_change_lock(request).locked():
        return "update_in_progress"
    return None


def _fallback_vram_snapshot(environment: EnvironmentReport) -> VramBudgetSnapshot:
    total = max(0, environment.vram_mb)
    selected = max(1024, environment.vram_limit_mb)
    mode = (
        VramBudgetMode.STANDARD
        if total < 1024 or selected == min(STANDARD_VRAM_MB, total)
        else VramBudgetMode.CUSTOM
    )
    return VramBudgetSnapshot(
        mode=mode,
        total_vram_mb=total,
        selected_vram_mb=selected,
        editable=False,
        blocked_reason="vram_budget_unavailable",
    )


def _preview_service(request: Request) -> PreviewServiceLike:
    return cast(PreviewServiceLike, request.app.state.preview_service)


def _close_live_preview_if_supported(service: object | None) -> None:
    if service is None:
        return
    close_live = getattr(service, "close_live", None)
    if callable(close_live):
        close_live()


def _suspend_live_preview_if_supported(service: object | None) -> object | None:
    if service is None:
        return None
    suspend_live = getattr(service, "suspend_live", None)
    if callable(suspend_live):
        return cast(object, suspend_live())
    _close_live_preview_if_supported(service)
    return None


def _resume_live_preview_if_supported(
    service: object | None, token: object | None
) -> None:
    if service is None or token is None:
        return
    resume_live = getattr(service, "resume_live", None)
    if callable(resume_live):
        resume_live(token)


def _preview_artifacts(request: Request) -> PreviewArtifactStore:
    return cast(PreviewArtifactStore, request.app.state.preview_artifacts)


def _preview_coordinator(request: Request) -> PreviewCoordinator:
    return cast(PreviewCoordinator, request.app.state.preview_coordinator)


def _camera(value: CameraInput) -> OrbitCamera:
    return OrbitCamera(
        target=(value.target[0], value.target[1], value.target[2]),
        distance=value.distance,
        yaw=value.yaw,
        pitch=value.pitch,
        fov_y_degrees=value.fov_y_degrees,
    )


def _unproject(
    camera: OrbitCamera, x: int, y: int, depth: float, width: int, height: int
) -> tuple[float, float, float]:
    import numpy as np

    ray = np.linalg.inv(camera.intrinsics(width, height)) @ np.array(
        [x + 0.5, y + 0.5, 1.0], dtype=np.float64
    )
    camera_point = ray * depth
    world = camera.camera_to_world() @ np.array(
        [camera_point[0], camera_point[1], camera_point[2], 1.0],
        dtype=np.float64,
    )
    return (float(world[0]), float(world[1]), float(world[2]))


def _load_project(repository: ProjectRepositoryLike) -> Project:
    try:
        return repository.load()
    except ActiveProjectRequiredError as error:
        raise ApiError(
            409,
            code="active_project_required",
            category="project",
            message="Open or create a project before using the workflow.",
        ) from error
    except (OSError, ValueError) as error:
        raise ApiError(
            404,
            code="project_unavailable",
            category="project",
            message="The current project is unavailable.",
        ) from error


def _scene_input(
    services: ApiServices, project: Project
) -> tuple[str | Path, str]:
    if project.scene_ply_asset_id is not None:
        if services.asset_library is None:
            raise ApiError(
                503,
                code="asset_library_unavailable",
                category="asset",
                message="The shared asset library is unavailable.",
            )
        try:
            return (
                services.asset_library.resolve(
                    project.scene_ply_asset_id, LibraryAssetKind.PLY
                ),
                project.scene_ply_asset_id,
            )
        except (KeyError, ValueError, OSError) as error:
            raise ApiError(
                409,
                code="scene_unavailable",
                category="project",
                message="The selected Gaussian scene is unavailable.",
            ) from error
    if project.scene_ply is not None:
        return project.scene_ply, project.scene_ply
    raise ApiError(
        409,
        code="scene_required",
        category="project",
        message="Import a Gaussian scene before rendering a preview.",
    )
def _confined_destination(root: Path, directory: str, filename: str) -> Path:
    canonical_root = root.resolve()
    destination_dir = (canonical_root / directory).resolve()
    if not destination_dir.is_relative_to(canonical_root):
        raise ApiError(
            400,
            code="invalid_project_root",
            category="filesystem",
            message="The project input directory is invalid.",
        )
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = (destination_dir / f"{uuid4().hex}-{filename}").resolve()
    if not destination.is_relative_to(destination_dir):
        raise ApiError(
            400,
            code="invalid_asset_path",
            category="validation",
            message="The selected asset path is invalid.",
        )
    return destination


def _copy_bounded(source: Path, destination: Path, limit: int) -> tuple[int, str]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while block := reader.read(1024 * 1024):
                size += len(block)
                if size > limit:
                    raise ApiError(
                        413,
                        code="asset_too_large",
                        category="validation",
                        message="The selected asset exceeds the configured size limit.",
                    )
                writer.write(block)
                digest.update(block)
        os.replace(temporary, destination)
    except OSError as error:
        if error.errno == errno.ENOSPC:
            raise ApiError(
                507,
                code="storage_full",
                category="storage",
                message="Insufficient storage for the selected asset.",
            ) from error
        raise ApiError(
            400,
            code="asset_import_failed",
            category="filesystem",
            message="The selected asset could not be imported.",
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return size, digest.hexdigest()


def _import_asset_sync(
    services: ApiServices,
    settings: ApiSettings,
    asset: AssetImportRequest,
) -> AssetResponse:
    try:
        source = Path(asset.path).expanduser().resolve(strict=True)
    except OSError as error:
        raise ApiError(
            400,
            code="asset_unavailable",
            category="filesystem",
            message="The selected asset is unavailable.",
        ) from error
    if not source.is_file():
        raise ApiError(
            400,
            code="asset_unavailable",
            category="filesystem",
            message="The selected asset is unavailable.",
        )
    if services.asset_library is not None:
        library_kind = (
            LibraryAssetKind.VIDEO
            if asset.kind == AssetKind.SOURCE_VIDEO.value
            else LibraryAssetKind.PLY
        )

        def inspect(path: Path, size: int, sha256: str) -> Any:
            if services.asset_inspector is None:
                raise ValueError("asset inspection is unavailable")
            return services.asset_inspector.inspect(
                asset.kind, path, size=size, sha256=sha256
            )

        try:
            record, _ = services.asset_library.import_file(
                library_kind, source, inspect
            )
        except ValueError as error:
            raise ApiError(
                422,
                code="unsupported_asset",
                category="validation",
                message="The selected asset is outside the supported MVP limits.",
            ) from error
        except OSError as error:
            raise ApiError(
                400,
                code="asset_import_failed",
                category="filesystem",
                message="The selected asset could not be imported.",
                retryable=True,
            ) from error
        if asset.assign_to_current:
            current = services.project_repository.load()
            _require_combined_vram_admission(
                services,
                record.video_summary
                if record.kind is LibraryAssetKind.VIDEO
                else current.workflow.source_summary,
                record.scene_summary
                if record.kind is LibraryAssetKind.PLY
                else current.workflow.scene_summary,
            )
            suspension = (
                _suspend_live_preview_if_supported(services.preview_service)
                if library_kind is LibraryAssetKind.PLY
                else None
            )
            try:
                services.project_repository.update(
                    lambda project: _select_library_asset(project, record)
                )
            finally:
                _resume_live_preview_if_supported(
                    services.preview_service, suspension
                )
        return AssetResponse(
            kind=asset.kind,
            path=record.asset_id,
            size=record.size,
            sha256=record.sha256,
            asset_id=record.asset_id,
        )
    suspension = (
        _suspend_live_preview_if_supported(services.preview_service)
        if asset.kind == AssetKind.SCENE_PLY.value
        else None
    )
    try:
        destination = _confined_destination(
            services.project_repository.root, "source", source.name
        )
        size, sha256 = _copy_bounded(
            source, destination, settings.max_upload_size
        )
        relative = destination.relative_to(
            services.project_repository.root.resolve()
        ).as_posix()
        summary = None
        if services.asset_inspector is not None:
            try:
                summary = services.asset_inspector.inspect(
                    asset.kind, destination, size=size, sha256=sha256
                )
            except Exception as error:
                destination.unlink(missing_ok=True)
                raise ApiError(
                    422,
                    code="unsupported_asset",
                    category="validation",
                    message="The selected asset is outside the supported MVP limits.",
                ) from error

        def persist(project: Project) -> None:
            if asset.kind == AssetKind.SOURCE_VIDEO.value:
                _replace_source_video(project, relative, cast(Any, summary))
            else:
                project.scene_ply = relative
                project.workflow.scene_summary = cast(Any, summary)
                invalidate_for_change(project, ChangeKind.TARGET_CAMERA)
                project.workflow.target_camera = None
                _clear_preview_authority(project)
            project.workflow.export_result = None

        services.project_repository.update(persist)
        return AssetResponse(
            kind=asset.kind, path=relative, size=size, sha256=sha256
        )
    finally:
        _resume_live_preview_if_supported(services.preview_service, suspension)


def _replace_source_video(project: Project, relative: str, summary: Any) -> None:
    project.source_video = relative
    project.workflow.source_summary = summary
    project.workflow.subject_prompt = None
    _clear_preview_authority(project)
    project.workflow.export_result = None
    project.workflow.active_task_id = None
    invalidate_for_change(project, ChangeKind.SOURCE_VIDEO)


def _select_library_asset(project: Project, record: AssetRecord) -> None:
    if record.kind is LibraryAssetKind.VIDEO:
        project.source_video_asset_id = record.asset_id
        project.source_video = None
        project.workflow.source_summary = record.video_summary
        project.workflow.subject_prompt = None
        project.workflow.active_task_id = None
        invalidate_for_change(project, ChangeKind.SOURCE_VIDEO)
    else:
        project.scene_ply_asset_id = record.asset_id
        project.scene_ply = None
        project.workflow.scene_summary = record.scene_summary
        project.workflow.target_camera = None
        invalidate_for_change(project, ChangeKind.TARGET_CAMERA)
    _clear_preview_authority(project)
    project.workflow.export_result = None


def _require_combined_vram_admission(
    services: ApiServices,
    source: VideoSummary | None,
    scene: SceneSummary | None,
) -> None:
    if source is None or scene is None or services.vram_budget is None:
        return
    budget_mb = services.vram_budget.current_limit_mb()
    if not fits_vram_budget(scene, source.width, source.height, budget_mb):
        raise ApiError(
            422,
            code="combined_vram_limit_exceeded",
            category="resource",
            message=(
                "The selected video resolution and Gaussian scene exceed 80% "
                "of the configured VRAM budget."
            ),
        )


def _require_task_cache_admission(
    services: ApiServices, stage: StageName
) -> None:
    store = services.artifact_store
    if store is None:
        return
    project = services.project_repository.load()
    cache_ok, required_bytes, available_bytes = project_cache_has_capacity(
        store.root, project, stage
    )
    if not cache_ok:
        raise ApiError(
            507,
            code="storage_full",
            category="storage",
            message=(
                "Insufficient cache storage for the requested workflow: requires "
                f"{required_bytes} bytes with 20% headroom, but only "
                f"{available_bytes} bytes are available."
            ),
            retryable=True,
        )


def _clear_preview_authority(project: Project) -> None:
    project.workflow.preview_epoch += 1
    project.workflow.confirmed_camera_revision = None
    project.workflow.confirmed_preview_artifact_id = None
    project.workflow.foot_point = None
    project.workflow.preview = None


def build_router() -> APIRouter:
    router = APIRouter()
    protected = APIRouter(dependencies=[Depends(require_session)])

    @protected.get("/healthz", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @protected.post("/api/v1/shutdown", status_code=status.HTTP_202_ACCEPTED)
    async def shutdown(request: Request) -> Response:
        callback = getattr(request.app.state, "request_shutdown", None)
        if not callable(callback):
            raise ApiError(
                503,
                code="shutdown_unavailable",
                category="runtime",
                message="The local service cannot be stopped by this host.",
            )
        callback()
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @protected.get("/api/v1/bootstrap", response_model=BootstrapResponse)
    async def bootstrap(request: Request) -> BootstrapResponse:
        services = _services(request)
        suspension = await asyncio.to_thread(
            _suspend_live_preview_if_supported,
            services.preview_service,
        )
        try:
            environment = await asyncio.to_thread(
                services.environment_doctor.check
            )
        finally:
            await asyncio.to_thread(
                _resume_live_preview_if_supported,
                services.preview_service,
                suspension,
            )
        capabilities: tuple[str, ...] = (
            "projects",
            "assets",
            "uploads",
            "tasks",
            "events",
        )
        if services.environment_repair is not None:
            capabilities = (*capabilities, "environment_repair")
        if services.vram_budget is not None:
            capabilities = (*capabilities, "vram_budget")
        if services.storage_layout is not None:
            capabilities = (*capabilities, "storage_layout")
        blocked_reason = _vram_blocked_reason(request)
        budget = (
            services.vram_budget.snapshot(
                editable=blocked_reason is None,
                blocked_reason=blocked_reason,
            )
            if services.vram_budget is not None
            else _fallback_vram_snapshot(environment)
        )
        projects = (
            services.project_manager.list()
            if services.project_manager is not None
            else ()
        )
        active_project = (
            services.project_manager.active_project()
            if services.project_manager is not None
            else _load_project(services.project_repository)
        )
        asset_counts = (
            {
                kind: len(services.asset_library.list(kind))
                for kind in LibraryAssetKind
            }
            if services.asset_library is not None
            else {}
        )
        return BootstrapResponse(
            api_version=API_VERSION,
            capabilities=capabilities,
            project=active_project,
            projects=projects,
            asset_counts=asset_counts,
            environment=environment,
            vram_budget=budget,
            storage_layout=(
                services.storage_layout.snapshot(
                    editable=_storage_blocked_reason(request) is None,
                    blocked_reason=_storage_blocked_reason(request),
                )
                if services.storage_layout is not None
                else None
            ),
        )

    @protected.get(
        "/api/v1/runtime/storage-layout",
        response_model=StorageLayoutSnapshot,
    )
    async def get_storage_layout(request: Request) -> StorageLayoutSnapshot:
        blocked_reason = _storage_blocked_reason(request)
        return await asyncio.to_thread(
            _storage_layout(request).snapshot,
            editable=blocked_reason is None,
            blocked_reason=blocked_reason,
        )

    @protected.patch(
        "/api/v1/runtime/storage-layout",
        response_model=StorageLayoutSnapshot,
    )
    async def patch_storage_layout(
        request: Request,
        update: StorageLayoutUpdate,
    ) -> StorageLayoutSnapshot:
        async with _runtime_change_lock(request):
            blocked_reason = _storage_blocked_reason(
                request, include_update=False
            )
            if blocked_reason is not None:
                raise ApiError(
                    409,
                    code="storage_layout_busy",
                    category="conflict",
                    message="Wait for active work to finish before changing storage.",
                    retryable=True,
                )
            services = _services(request)
            suspension = await asyncio.to_thread(
                _suspend_live_preview_if_supported,
                services.preview_service,
            )
            try:
                return await asyncio.to_thread(
                    _storage_layout(request).switch,
                    Path(update.project_library_root),
                    Path(update.cache_root),
                    project_action=update.project_action,
                    cache_action=update.cache_action,
                )
            except (OSError, RuntimeError, ValueError) as error:
                await asyncio.to_thread(
                    _resume_live_preview_if_supported,
                    services.preview_service,
                    suspension,
                )
                raise ApiError(
                    422,
                    code="storage_layout_change_failed",
                    category="filesystem",
                    message=str(error),
                    retryable=False,
                ) from error

    @protected.post(
        "/api/v1/runtime/storage-layout/cache-cleanup/plan",
        response_model=CacheCleanupPlan,
    )
    async def plan_storage_cache_cleanup(
        request: Request,
        cleanup: CacheCleanupPlanRequest,
    ) -> CacheCleanupPlan:
        blocked_reason = _storage_blocked_reason(request, include_update=False)
        if blocked_reason is not None:
            raise ApiError(
                409,
                code="storage_cleanup_busy",
                category="conflict",
                message="Wait for active work to finish before scanning cache.",
                retryable=True,
            )
        try:
            return await asyncio.to_thread(
                _storage_layout(request).plan_cache_cleanup,
                cleanup.mode,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise ApiError(
                422,
                code="storage_cleanup_plan_failed",
                category="filesystem",
                message=str(error),
                retryable=False,
            ) from error

    @protected.post(
        "/api/v1/runtime/storage-layout/cache-cleanup",
        response_model=CacheCleanupResult,
    )
    async def cleanup_storage_cache(
        request: Request,
        cleanup: CacheCleanupRequest,
    ) -> CacheCleanupResult:
        async with _runtime_change_lock(request):
            blocked_reason = _storage_blocked_reason(
                request, include_update=False
            )
            if blocked_reason is not None:
                raise ApiError(
                    409,
                    code="storage_cleanup_busy",
                    category="conflict",
                    message="Wait for active work to finish before cleaning cache.",
                    retryable=True,
                )
            services = _services(request)
            suspension = await asyncio.to_thread(
                _suspend_live_preview_if_supported,
                services.preview_service,
            )
            try:
                return await asyncio.to_thread(
                    _storage_layout(request).cleanup_cache,
                    cleanup.mode,
                    cleanup.plan_token,
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise ApiError(
                    422,
                    code="storage_cleanup_failed",
                    category="filesystem",
                    message=str(error),
                    retryable=False,
                ) from error
            finally:
                await asyncio.to_thread(
                    _resume_live_preview_if_supported,
                    services.preview_service,
                    suspension,
                )

    @protected.get(
        "/api/v1/runtime/vram-budget",
        response_model=VramBudgetSnapshot,
    )
    async def get_vram_budget(request: Request) -> VramBudgetSnapshot:
        blocked_reason = _vram_blocked_reason(request)
        return await asyncio.to_thread(
            _vram_budget(request).snapshot,
            editable=blocked_reason is None,
            blocked_reason=blocked_reason,
        )

    @protected.patch(
        "/api/v1/runtime/vram-budget",
        response_model=VramBudgetSnapshot,
    )
    async def patch_vram_budget(
        request: Request,
        update: VramBudgetUpdate,
    ) -> VramBudgetSnapshot:
        async with _runtime_change_lock(request):
            blocked_reason = _vram_blocked_reason(request, include_update=False)
            if blocked_reason is not None:
                raise ApiError(
                    409,
                    code="vram_budget_busy",
                    category="conflict",
                    message="Wait for the active GPU task or environment repair to finish.",
                    retryable=True,
                )
            services = _services(request)
            suspension = await asyncio.to_thread(
                _suspend_live_preview_if_supported,
                services.preview_service,
            )
            try:
                try:
                    return await asyncio.to_thread(
                        _vram_budget(request).update,
                        update.mode,
                        update.selected_vram_mb,
                    )
                except ValueError as error:
                    raise ApiError(
                        422,
                        code="invalid_vram_budget",
                        category="validation",
                        message=str(error),
                        retryable=False,
                    ) from error
                except VramBudgetUnavailableError as error:
                    raise ApiError(
                        503,
                        code="vram_budget_gpu_unavailable",
                        category="runtime",
                        message="Physical GPU memory could not be detected.",
                        retryable=True,
                    ) from error
                except VramBudgetPersistenceError as error:
                    raise ApiError(
                        500,
                        code="vram_budget_save_failed",
                        category="runtime",
                        message="The VRAM preference could not be saved.",
                        retryable=True,
                    ) from error
            finally:
                await asyncio.to_thread(
                    _resume_live_preview_if_supported,
                    services.preview_service,
                    suspension,
                )

    @protected.get(
        "/api/v1/environment/repair",
        response_model=EnvironmentRepairSnapshot,
    )
    async def get_environment_repair(request: Request) -> EnvironmentRepairSnapshot:
        return await asyncio.to_thread(_environment_repair(request).snapshot)

    @protected.post(
        "/api/v1/environment/repair",
        response_model=EnvironmentRepairSnapshot,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def start_environment_repair(request: Request) -> EnvironmentRepairSnapshot:
        repair = _environment_repair(request)
        async with _runtime_change_lock(request):
            try:
                return await asyncio.to_thread(repair.start)
            except EnvironmentRepairBusyError as error:
                raise ApiError(
                    409,
                    code="environment_repair_busy",
                    category="environment",
                    message="Another environment repair is already active.",
                    retryable=True,
                ) from error

    @protected.delete(
        "/api/v1/environment/repair",
        response_model=EnvironmentRepairSnapshot,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def cancel_environment_repair(request: Request) -> EnvironmentRepairSnapshot:
        repair = _environment_repair(request)
        return await asyncio.to_thread(repair.cancel)

    @protected.get("/api/v1/projects/current", response_model=Project)
    async def get_current_project(request: Request) -> Project:
        return _load_project(_services(request).project_repository)

    @protected.patch("/api/v1/projects/current", response_model=Project)
    async def patch_current_project(request: Request, patch: ProjectPatch) -> Project:
        services = _services(request)
        repository = services.project_repository

        async with _runtime_change_lock(request):
            if patch.name is not None and services.project_manager is not None:
                current = _load_project(repository)
                try:
                    await asyncio.to_thread(
                        services.project_manager.rename,
                        current.project_id,
                        patch.name,
                    )
                except ValueError as error:
                    raise ApiError(
                        409,
                        code="project_name_conflict",
                        category="project",
                        message=str(error),
                    ) from error

            def mutate(project: Project) -> None:
                if patch.name is not None and services.project_manager is None:
                    project.name = patch.name
                if "subject_prompt" in patch.model_fields_set:
                    prompt = (
                        None
                        if patch.subject_prompt is None
                        else SubjectPromptState.model_validate(
                            patch.subject_prompt.model_dump()
                        )
                    )
                    if prompt is not None:
                        artifact_root = (
                            repository.root
                            if services.artifact_store is None
                            else services.artifact_store.project_root(
                                project.project_id
                            )
                        )
                        validate_subject_prompt(project, artifact_root, prompt)
                    project.workflow.subject_prompt = prompt
                    invalidate_for_change(project, ChangeKind.SUBJECT_PROMPT)
                    project.workflow.export_result = None
                if patch.motion_scale is not None:
                    project.workflow.motion_scale = patch.motion_scale
                    invalidate_for_change(project, ChangeKind.MOTION_SCALE)
                    project.workflow.export_result = None
                if patch.preview_height is not None:
                    project.workflow.preview_height = patch.preview_height
                    invalidate_for_change(project, ChangeKind.TARGET_CAMERA)
                    _clear_preview_authority(project)
                    project.workflow.export_result = None

            if patch.model_fields_set == {"name"} and services.project_manager is not None:
                return _load_project(repository)
            return await asyncio.to_thread(repository.update, mutate)

    def require_project_manager(request: Request) -> ActiveProjectManager:
        manager = _services(request).project_manager
        if manager is None:
            raise ApiError(
                503,
                code="project_catalog_unavailable",
                category="project",
                message="Project management is unavailable in this local service.",
            )
        return manager

    def reject_project_change_while_busy(request: Request) -> None:
        if _task_service(request).is_busy():
            raise ApiError(
                409,
                code="project_task_active",
                category="conflict",
                message="Wait for the active task to finish or cancel it first.",
                retryable=True,
            )

    @protected.get("/api/v1/projects", response_model=tuple[ProjectSummary, ...])
    async def list_projects(request: Request) -> tuple[ProjectSummary, ...]:
        return await asyncio.to_thread(require_project_manager(request).list)

    @protected.post(
        "/api/v1/projects",
        response_model=Project,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_project(request: Request, body: ProjectCreate) -> Project:
        async with _runtime_change_lock(request):
            reject_project_change_while_busy(request)
            try:
                return await asyncio.to_thread(
                    require_project_manager(request).create,
                    body.name,
                    lambda: _close_live_preview_if_supported(
                        _services(request).preview_service
                    ),
                )
            except ValueError as error:
                raise ApiError(
                    409,
                    code="project_name_conflict",
                    category="project",
                    message=str(error),
                ) from error

    @protected.post(
        "/api/v1/projects/{project_id}/activate",
        response_model=Project,
    )
    async def activate_project(request: Request, project_id: str) -> Project:
        async with _runtime_change_lock(request):
            reject_project_change_while_busy(request)
            services = _services(request)
            try:
                return await asyncio.to_thread(
                    require_project_manager(request).activate,
                    project_id,
                    lambda: _close_live_preview_if_supported(
                        services.preview_service
                    ),
                )
            except (KeyError, ValueError) as error:
                raise ApiError(
                    404,
                    code="project_not_found",
                    category="project",
                    message="The requested project is unavailable.",
                ) from error
            except RuntimeError as error:
                raise ApiError(
                    409,
                    code="project_locked",
                    category="project",
                    message="The requested project is open in another application instance.",
                    retryable=True,
                ) from error

    @protected.patch(
        "/api/v1/projects/{project_id}",
        response_model=ProjectSummary,
    )
    async def rename_project(
        request: Request, project_id: str, body: ProjectRename
    ) -> ProjectSummary:
        try:
            return await asyncio.to_thread(
                require_project_manager(request).rename, project_id, body.name
            )
        except KeyError as error:
            raise ApiError(
                404,
                code="project_not_found",
                category="project",
                message="The requested project is unavailable.",
            ) from error
        except ValueError as error:
            raise ApiError(
                409,
                code="project_name_conflict",
                category="project",
                message=str(error),
            ) from error

    @protected.delete(
        "/api/v1/projects/{project_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_project(request: Request, project_id: str) -> Response:
        async with _runtime_change_lock(request):
            reject_project_change_while_busy(request)
            services = _services(request)
            active = require_project_manager(request).active_project()
            try:
                await asyncio.to_thread(
                    require_project_manager(request).delete,
                    project_id,
                    (
                        lambda: _close_live_preview_if_supported(
                            services.preview_service
                        )
                        if active is not None and active.project_id == project_id
                        else None
                    ),
                )
            except KeyError as error:
                raise ApiError(
                    404,
                    code="project_not_found",
                    category="project",
                    message="The requested project is unavailable.",
                ) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @protected.post(
        "/api/v1/projects/current/preview/live",
        response_class=Response,
    )
    async def render_live_preview(
        request: Request, preview: LivePreviewRequest
    ) -> Response:
        services = _services(request)
        repository = services.project_repository
        project = _load_project(repository)
        scene_summary = project.workflow.scene_summary
        if scene_summary is None:
            raise ApiError(
                409,
                code="scene_required",
                category="project",
                message="Import a Gaussian scene before rendering a preview.",
            )
        scene_input, _scene_reference = _scene_input(services, project)
        payload = await asyncio.to_thread(
            _preview_service(request).render_live,
            repository.root,
            scene_input,
            scene_summary.model_copy(deep=True),
            preview.request_id,
            _camera(preview.camera),
            preview.width,
            preview.height,
        )
        return Response(
            content=payload,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "X-Preview-Request-Id": str(preview.request_id),
            },
        )

    @protected.delete(
        "/api/v1/projects/current/preview/live",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def close_live_preview(request: Request) -> Response:
        await asyncio.to_thread(_preview_service(request).close_live)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @protected.post(
        "/api/v1/projects/current/preview",
        response_model=PreviewFrameResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def render_preview(
        request: Request, preview: PreviewFrameRequest
    ) -> PreviewFrameResponse:
        services = _services(request)
        repository = services.project_repository
        project = _load_project(repository)
        scene_summary = project.workflow.scene_summary
        if scene_summary is None:
            raise ApiError(
                409,
                code="scene_required",
                category="project",
                message="Import a Gaussian scene before rendering a preview.",
            )
        current = project.workflow.preview
        if current is not None and preview.generation <= current.generation:
            raise ApiError(
                409,
                code="stale_preview_generation",
                category="conflict",
                message="A newer preview generation is already authoritative.",
            )
        camera = _camera(preview.camera)
        scene_path, scene_reference = _scene_input(services, project)
        scene_authority = scene_summary.model_copy(deep=True)
        preview_epoch = project.workflow.preview_epoch
        request_fingerprint: PreviewRequestFingerprint = (
            (
                preview.camera.target[0],
                preview.camera.target[1],
                preview.camera.target[2],
            ),
            preview.camera.distance,
            preview.camera.yaw,
            preview.camera.pitch,
            preview.camera.fov_y_degrees,
            preview.width,
            preview.height,
        )
        buffer: PickBuffer = await _preview_coordinator(request).render(
            (
                project.project_id,
                scene_reference,
                scene_authority.sha256,
                scene_authority.size,
                preview_epoch,
            ),
            preview.generation,
            request_fingerprint,
            _preview_service(request).render_pick,
            repository.root,
            scene_path,
            scene_authority,
            camera,
            preview.width,
            preview.height,
        )
        validate_pick_buffer(buffer, width=preview.width, height=preview.height)
        response: PreviewFrameResponse | None = None

        def persist(latest: Project) -> None:
            nonlocal response
            if (
                (
                    latest.scene_ply_asset_id or latest.scene_ply
                ) != scene_reference
                or latest.workflow.scene_summary != scene_authority
                or latest.workflow.preview_epoch != preview_epoch
            ):
                raise ApiError(
                    409,
                    code="scene_changed",
                    category="conflict",
                    message="The Gaussian scene changed while the preview was rendered.",
                    retryable=True,
                )
            latest_preview = latest.workflow.preview
            if (
                latest_preview is not None
                and preview.generation <= latest_preview.generation
            ):
                raise ApiError(
                    409,
                    code="stale_preview_generation",
                    category="conflict",
                    message="A newer preview generation is already authoritative.",
                )
            artifact_id, artifact_size, artifact_sha256 = (
                _preview_artifacts(request).publish(
                    buffer,
                    preserve_artifact_ids=(
                        set()
                        if latest_preview is None
                        else {latest_preview.artifact_id}
                    ),
                )
            )
            revision = (
                1
                if latest.workflow.target_camera is None
                else latest.workflow.target_camera.revision + 1
            )
            latest.workflow.target_camera = CameraPose(
                **preview.camera.model_dump(), revision=revision
            )
            latest.workflow.foot_point = None
            latest.workflow.confirmed_camera_revision = None
            latest.workflow.confirmed_preview_artifact_id = None
            latest.workflow.preview = PreviewState(
                artifact_id=artifact_id,
                artifact_size=artifact_size,
                artifact_sha256=artifact_sha256,
                generation=preview.generation,
                width=preview.width,
                height=preview.height,
                camera_revision=revision,
                pick_buffer_revision=revision,
            )
            latest.workflow.export_result = None
            invalidate_for_change(latest, ChangeKind.TARGET_CAMERA)
            response = PreviewFrameResponse(
                artifact_id=artifact_id,
                generation=preview.generation,
                width=preview.width,
                height=preview.height,
                camera_revision=revision,
                pick_buffer_revision=revision,
            )

        await asyncio.to_thread(repository.update, persist)
        assert response is not None
        return response

    @protected.get("/api/v1/projects/current/previews/{artifact_id}")
    async def get_preview_artifact(request: Request, artifact_id: str) -> Response:
        project = _load_project(_services(request).project_repository)
        preview = project.workflow.preview
        if preview is None or preview.artifact_id != artifact_id:
            raise ApiError(
                404,
                code="preview_unavailable",
                category="project",
                message="The requested preview is unavailable.",
            )
        payload = await asyncio.to_thread(
            _preview_artifacts(request).read,
            artifact_id=artifact_id,
            expected_size=preview.artifact_size,
            expected_sha256=preview.artifact_sha256,
        )
        return Response(
            content=payload,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @protected.post(
        "/api/v1/projects/current/camera/confirm", response_model=Project
    )
    async def confirm_camera(
        request: Request, confirmation: CameraConfirmationRequest
    ) -> Project:
        repository = _services(request).project_repository

        def persist(project: Project) -> None:
            camera = project.workflow.target_camera
            preview = project.workflow.preview
            if (
                camera is None
                or preview is None
                or camera.revision != confirmation.camera_revision
                or preview.camera_revision != confirmation.camera_revision
            ):
                raise ApiError(
                    409,
                    code="stale_camera_revision",
                    category="conflict",
                    message="Render the current camera before confirming it.",
                )
            project.workflow.confirmed_camera_revision = (
                confirmation.camera_revision
            )
            project.workflow.confirmed_preview_artifact_id = preview.artifact_id
            project.workflow.foot_point = None

        return await asyncio.to_thread(repository.update, persist)

    @protected.post(
        "/api/v1/projects/current/pick", response_model=PickResponse
    )
    async def pick_foot_point(
        request: Request, pick: PickRequest
    ) -> PickResponse:
        repository = _services(request).project_repository
        project = _load_project(repository)
        preview = project.workflow.preview
        camera_state = project.workflow.target_camera
        scene_path = project.scene_ply_asset_id or project.scene_ply
        scene_authority = project.workflow.scene_summary
        preview_epoch = project.workflow.preview_epoch
        if (
            preview is None
            or camera_state is None
            or scene_path is None
            or scene_authority is None
            or preview.artifact_id != pick.preview_artifact_id
            or preview.camera_revision != pick.camera_revision
            or preview.pick_buffer_revision != pick.pick_buffer_revision
        ):
            raise ApiError(
                409,
                code="stale_pick_buffer",
                category="conflict",
                message="Regenerate the preview before choosing a foot point.",
            )
        if (
            project.workflow.confirmed_camera_revision != pick.camera_revision
            or project.workflow.confirmed_preview_artifact_id
            != pick.preview_artifact_id
        ):
            raise ApiError(
                409,
                code="camera_not_confirmed",
                category="conflict",
                message="Confirm the current camera before choosing a foot point.",
            )
        if pick.x >= preview.width or pick.y >= preview.height:
            raise ApiError(
                422,
                code="pick_outside_image",
                category="validation",
                message="The selected point is outside the preview image.",
            )
        buffer = _preview_artifacts(request).pick_buffer(preview.artifact_id)
        depth = float(buffer.expected_depth[pick.y, pick.x])
        if not (depth > 0.0 and depth < float("inf")):
            raise ApiError(
                422,
                code="invalid_pick_depth",
                category="render",
                message="Choose a point with valid scene depth.",
            )
        camera = OrbitCamera(
            target=camera_state.target,
            distance=camera_state.distance,
            yaw=camera_state.yaw,
            pitch=camera_state.pitch,
            fov_y_degrees=camera_state.fov_y_degrees,
        )
        world = _unproject(
            camera, pick.x, pick.y, depth, preview.width, preview.height
        )
        foot = FootPointState(
            image=(pick.x, pick.y),
            world=world,
            preview_artifact_id=pick.preview_artifact_id,
            camera_revision=pick.camera_revision,
            pick_buffer_revision=pick.pick_buffer_revision,
        )

        def persist(latest: Project) -> None:
            latest_preview = latest.workflow.preview
            if (
                latest_preview is None
                or (latest.scene_ply_asset_id or latest.scene_ply) != scene_path
                or latest.workflow.scene_summary != scene_authority
                or latest.workflow.preview_epoch != preview_epoch
                or latest_preview.artifact_id != pick.preview_artifact_id
                or latest_preview.camera_revision != pick.camera_revision
                or latest_preview.pick_buffer_revision
                != pick.pick_buffer_revision
            ):
                raise ApiError(
                    409,
                    code="stale_pick_buffer",
                    category="conflict",
                    message="Regenerate the preview before choosing a foot point.",
                )
            if (
                latest.workflow.confirmed_camera_revision != pick.camera_revision
                or latest.workflow.confirmed_preview_artifact_id
                != pick.preview_artifact_id
            ):
                raise ApiError(
                    409,
                    code="camera_not_confirmed",
                    category="conflict",
                    message="Confirm the current camera before choosing a foot point.",
                )
            latest.workflow.foot_point = foot
            latest.workflow.export_result = None
            invalidate_for_change(latest, ChangeKind.TARGET_CAMERA)

        await asyncio.to_thread(repository.update, persist)
        return PickResponse.model_validate(foot.model_dump())

    @protected.post(
        "/api/v1/assets/import",
        response_model=AssetResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def import_asset(request: Request, asset: AssetImportRequest) -> AssetResponse:
        _require_desktop_path_import(request)
        services = _services(request)
        settings = _settings(request)
        if asset.assign_to_current:
            async with _runtime_change_lock(request):
                reject_project_change_while_busy(request)
                return await asyncio.to_thread(
                    _import_asset_sync, services, settings, asset
                )
        return await asyncio.to_thread(
            _import_asset_sync, services, settings, asset
        )

    @protected.get("/api/v1/assets", response_model=tuple[AssetListItem, ...])
    async def list_assets(
        request: Request, kind: LibraryAssetKind
    ) -> tuple[AssetListItem, ...]:
        services = _services(request)
        if services.asset_library is None or services.project_catalog is None:
            raise ApiError(
                503,
                code="asset_library_unavailable",
                category="asset",
                message="The shared asset library is unavailable.",
            )
        records = await asyncio.to_thread(services.asset_library.list, kind)
        return tuple(
            AssetListItem(
                asset=record,
                references=services.project_catalog.references(record.asset_id),
            )
            for record in records
        )

    @protected.delete(
        "/api/v1/assets/{asset_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_asset(request: Request, asset_id: str) -> Response:
        services = _services(request)
        if services.asset_library is None or services.project_catalog is None:
            raise ApiError(
                503,
                code="asset_library_unavailable",
                category="asset",
                message="The shared asset library is unavailable.",
            )
        async with _runtime_change_lock(request):
            references = await asyncio.to_thread(
                services.project_catalog.references, asset_id
            )
            if references:
                names = ", ".join(reference.name for reference in references[:8])
                raise ApiError(
                    409,
                    code="asset_in_use",
                    category="conflict",
                    message=f"Remove this asset from its projects first: {names}",
                )
            try:
                await asyncio.to_thread(services.asset_library.delete, asset_id)
            except KeyError as error:
                raise ApiError(
                    404,
                    code="asset_not_found",
                    category="asset",
                    message="The requested asset is unavailable.",
                ) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @protected.patch(
        "/api/v1/projects/current/assets",
        response_model=Project,
    )
    async def select_project_assets(
        request: Request, selection: ProjectAssetSelection
    ) -> Project:
        services = _services(request)
        if services.asset_library is None:
            raise ApiError(
                503,
                code="asset_library_unavailable",
                category="asset",
                message="The shared asset library is unavailable.",
            )
        async with _runtime_change_lock(request):
            reject_project_change_while_busy(request)
            current = await asyncio.to_thread(services.project_repository.load)
            if current.project_id != selection.expected_project_id:
                raise ApiError(
                    409,
                    code="project_context_changed",
                    category="conflict",
                    message="The active project changed before the asset selection was applied.",
                    retryable=True,
                )
            records: dict[str, AssetRecord | None] = {}
            for field, kind in (
                ("source_video_asset_id", LibraryAssetKind.VIDEO),
                ("scene_ply_asset_id", LibraryAssetKind.PLY),
            ):
                if field not in selection.model_fields_set:
                    continue
                asset_id = getattr(selection, field)
                if asset_id is None:
                    records[field] = None
                    continue
                try:
                    record = services.asset_library.get(asset_id)
                except KeyError as error:
                    raise ApiError(
                        404,
                        code="asset_not_found",
                        category="asset",
                        message="The selected asset is unavailable.",
                    ) from error
                if record.kind is not kind:
                    raise ApiError(
                        422,
                        code="asset_kind_mismatch",
                        category="validation",
                        message="The selected asset has the wrong kind.",
                    )
                records[field] = record
            if not records:
                raise ApiError(
                    422,
                    code="asset_selection_required",
                    category="validation",
                    message="Select at least one project asset field.",
                )
            prospective_source = current.workflow.source_summary
            prospective_scene = current.workflow.scene_summary
            if "source_video_asset_id" in records:
                source_record = records["source_video_asset_id"]
                prospective_source = (
                    None if source_record is None else source_record.video_summary
                )
            if "scene_ply_asset_id" in records:
                scene_record = records["scene_ply_asset_id"]
                prospective_scene = (
                    None if scene_record is None else scene_record.scene_summary
                )
            _require_combined_vram_admission(
                services, prospective_source, prospective_scene
            )

            def persist(project: Project) -> None:
                for field, record in records.items():
                    if record is not None:
                        _select_library_asset(project, record)
                    elif field == "source_video_asset_id":
                        project.source_video_asset_id = None
                        project.workflow.source_summary = None
                        project.workflow.subject_prompt = None
                        invalidate_for_change(project, ChangeKind.SOURCE_VIDEO)
                        _clear_preview_authority(project)
                        project.workflow.export_result = None
                    else:
                        project.scene_ply_asset_id = None
                        project.workflow.scene_summary = None
                        project.workflow.target_camera = None
                        invalidate_for_change(project, ChangeKind.TARGET_CAMERA)
                        _clear_preview_authority(project)
                        project.workflow.export_result = None

            return await asyncio.to_thread(
                services.project_repository.update, persist
            )

    @protected.post(
        "/api/v1/tasks",
        response_model=TaskSnapshot,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_task(request: Request, task: TaskCreateRequest) -> TaskSnapshot:
        stage = StageName(task.target_stage)
        services = _services(request)
        runner = services.pipeline_runner
        supports = getattr(runner, "supports", None)
        if callable(supports) and not supports(stage):
            raise ApiError(
                503,
                code="workflow_unavailable",
                category="capability",
                message="The requested workflow stage is not assembled in this build.",
                retryable=False,
            )
        runtime_lock = _runtime_change_lock(request)
        await runtime_lock.acquire()
        suspension: object | None = None

        def release_suspension() -> None:
            if suspension is not None:
                _resume_live_preview_if_supported(
                    services.preview_service, suspension
                )
        try:
            if (
                services.project_manager is not None
                and task.expected_project_id is None
            ):
                raise ApiError(
                    422,
                    code="project_context_required",
                    category="validation",
                    message="Task creation requires the active project id.",
                    retryable=False,
                )
            current = await asyncio.to_thread(services.project_repository.load)
            if (
                task.expected_project_id is not None
                and current.project_id != task.expected_project_id
            ):
                raise ApiError(
                    409,
                    code="project_context_changed",
                    category="conflict",
                    message="The active project changed before the task was created.",
                    retryable=True,
                )
            suspension = await asyncio.to_thread(
                _suspend_live_preview_if_supported,
                services.preview_service,
            )
            if stage in {
                StageName.SEGMENT,
                StageName.RENDER,
                StageName.COMPOSITE,
                StageName.EXPORT,
            }:
                repair = services.environment_repair
                if repair is not None and repair.is_busy():
                    raise ApiError(
                        503,
                        code="environment_repair_in_progress",
                        category="environment",
                        message="Wait for environment repair to finish before starting this stage.",
                        retryable=True,
                    )
                report = await asyncio.to_thread(services.environment_doctor.check)
                if not report.ready:
                    issue_codes = ", ".join(
                        report_issue.code for report_issue in report.issues[:8]
                    )
                    raise ApiError(
                        503,
                        code="environment_not_ready",
                        category="environment",
                        message=(
                            "Repair the local runtime before starting this stage"
                            + (f": {issue_codes}" if issue_codes else ".")
                        ),
                        retryable=True,
                    )
            await asyncio.to_thread(
                _require_task_cache_admission, services, stage
            )
            snapshot = await _task_service(request).create(
                stage,
                on_terminal=release_suspension,
            )
            await asyncio.to_thread(
                services.project_repository.update,
                lambda project: setattr(
                    project.workflow, "active_task_id", snapshot.id
                ),
            )
        except BaseException:
            release_suspension()
            raise
        finally:
            runtime_lock.release()
        return snapshot

    @protected.get("/api/v1/tasks/{task_id}", response_model=TaskSnapshot)
    async def get_task(request: Request, task_id: str) -> TaskSnapshot:
        return _task_service(request).get(task_id)

    @protected.delete(
        "/api/v1/tasks/{task_id}",
        response_model=TaskSnapshot,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def cancel_task(request: Request, task_id: str) -> TaskSnapshot:
        return await _task_service(request).cancel(task_id)

    @protected.post(
        "/api/v1/uploads",
        response_model=UploadCreated,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_upload(request: Request, upload: UploadCreateRequest) -> UploadCreated:
        return await asyncio.to_thread(_upload_manager(request).create, upload)

    @protected.get("/api/v1/uploads/{upload_id}", response_model=UploadStatus)
    async def get_upload(request: Request, upload_id: str) -> UploadStatus:
        return await asyncio.to_thread(_upload_manager(request).status, upload_id)

    @protected.put("/api/v1/uploads/{upload_id}")
    async def reject_normalized_chunk_traversal(upload_id: str) -> None:
        del upload_id
        raise ApiError(
            400,
            code="invalid_chunk_index",
            category="validation",
            message="The upload chunk index is invalid.",
        )

    @protected.put(
        "/api/v1/uploads/{upload_id}/chunks/{index:path}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def put_upload_chunk(
        request: Request, upload_id: str, index: str
    ) -> Response:
        if not index.isascii() or not index.isdecimal():
            raise ApiError(
                400,
                code="invalid_chunk_index",
                category="validation",
                message="The upload chunk index is invalid.",
            )
        content = await read_bounded_body(request)
        await asyncio.to_thread(
            _upload_manager(request).put_chunk, upload_id, int(index), content
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @protected.post(
        "/api/v1/uploads/{upload_id}/complete",
        response_model=UploadComplete,
        status_code=status.HTTP_201_CREATED,
    )
    async def complete_upload(request: Request, upload_id: str) -> UploadComplete:
        services = _services(request)
        library_record: AssetRecord | None = None

        def persist(completed: UploadComplete, source_stream: BinaryIO) -> None:
            nonlocal library_record
            if services.asset_library is not None:
                assert services.upload_root is not None
                library_kind = (
                    LibraryAssetKind.VIDEO
                    if completed.kind == AssetKind.SOURCE_VIDEO.value
                    else LibraryAssetKind.PLY
                )

                def inspect(path: Path, size: int, sha256: str) -> Any:
                    if services.asset_inspector is None:
                        raise ValueError("asset inspection is unavailable")
                    return services.asset_inspector.inspect(
                        completed.kind, path, size=size, sha256=sha256
                    )

                library_record, _ = services.asset_library.import_stream(
                    library_kind,
                    completed.filename,
                    source_stream,
                    inspect,
                )
                if completed.assign_to_current:
                    current = services.project_repository.load()
                    _require_combined_vram_admission(
                        services,
                        library_record.video_summary
                        if library_record.kind is LibraryAssetKind.VIDEO
                        else current.workflow.source_summary,
                        library_record.scene_summary
                        if library_record.kind is LibraryAssetKind.PLY
                        else current.workflow.scene_summary,
                    )
                    suspension = (
                        _suspend_live_preview_if_supported(services.preview_service)
                        if library_kind is LibraryAssetKind.PLY
                        else None
                    )
                    try:
                        services.project_repository.update(
                            lambda project: _select_library_asset(
                                project, library_record
                            )
                        )
                    finally:
                        _resume_live_preview_if_supported(
                            services.preview_service, suspension
                        )
                return
            summary = None
            if services.asset_inspector is not None:
                path = services.project_repository.root / completed.path
                try:
                    summary = services.asset_inspector.inspect(
                        completed.kind,
                        path,
                        size=completed.size,
                        sha256=completed.sha256,
                    )
                except Exception as error:
                    raise ApiError(
                        422,
                        code="unsupported_asset",
                        category="validation",
                        message="The uploaded asset is outside the supported MVP limits.",
                    ) from error

            def update_project(project: Project) -> None:
                if completed.kind == AssetKind.SOURCE_VIDEO.value:
                    _replace_source_video(
                        project, completed.path, cast(Any, summary)
                    )
                else:
                    project.scene_ply = completed.path
                    project.workflow.scene_summary = cast(Any, summary)
                    invalidate_for_change(project, ChangeKind.TARGET_CAMERA)
                    project.workflow.target_camera = None
                    _clear_preview_authority(project)
                project.workflow.export_result = None

            suspension = (
                _suspend_live_preview_if_supported(services.preview_service)
                if completed.kind == AssetKind.SCENE_PLY.value
                else None
            )
            try:
                services.project_repository.update(update_project)
            finally:
                _resume_live_preview_if_supported(
                    services.preview_service, suspension
                )

        upload_manager = _upload_manager(request)
        if upload_manager.assigns_to_current(upload_id):
            async with _runtime_change_lock(request):
                reject_project_change_while_busy(request)
                completed = await asyncio.to_thread(
                    upload_manager.complete,
                    upload_id,
                    persist_stream=persist,
                )
        else:
            completed = await asyncio.to_thread(
                upload_manager.complete,
                upload_id,
                persist_stream=persist,
            )
        if library_record is not None:
            assert services.upload_root is not None
            (services.upload_root / completed.path).unlink(missing_ok=True)
            return completed.model_copy(update={"path": library_record.asset_id})
        return completed

    @protected.delete(
        "/api/v1/uploads/{upload_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def cancel_upload(request: Request, upload_id: str) -> Response:
        await asyncio.to_thread(_upload_manager(request).cancel, upload_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.websocket("/api/v1/events")
    async def event_websocket(websocket: WebSocket) -> None:
        settings = cast(ApiSettings, websocket.app.state.settings)
        events = cast(EventBus, websocket.app.state.event_bus)
        await serve_events(websocket, settings, events)

    protected.include_router(build_subject_router())
    protected.include_router(build_export_router())
    protected.include_router(build_composite_preview_router())
    router.include_router(protected)
    return router
