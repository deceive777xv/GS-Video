from __future__ import annotations

import asyncio
import errno
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, Request, Response, WebSocket, status

from gs_video.api.auth import require_session
from gs_video.api.events import EventBus, TaskService, serve_events
from gs_video.api.schemas import (
    API_VERSION,
    ApiError,
    ApiSettings,
    AssetImportRequest,
    AssetKind,
    AssetResponse,
    BootstrapResponse,
    HealthResponse,
    ProjectPatch,
    TaskCreateRequest,
    TaskSnapshot,
    UploadComplete,
    UploadCreateRequest,
    UploadCreated,
    UploadStatus,
)
from gs_video.api.uploads import UploadManager, read_bounded_body
from gs_video.domain.models import Project, StageName, StageState
from gs_video.environment.doctor import EnvironmentReport
from gs_video.pipeline.cancellation import CancellationToken


class ProjectRepositoryLike(Protocol):
    root: Path

    def load(self) -> Project: ...

    def save(self, project: Project) -> None: ...


class EnvironmentDoctorLike(Protocol):
    def check(self) -> EnvironmentReport: ...


class PipelineRunnerLike(Protocol):
    def run(self, name: StageName, token: CancellationToken) -> StageState: ...


class WorkerRegistryLike(Protocol):
    async def terminate_all(self) -> None: ...


@dataclass(frozen=True)
class ApiServices:
    project_repository: ProjectRepositoryLike
    environment_doctor: EnvironmentDoctorLike
    pipeline_runner: PipelineRunnerLike
    worker_registry: WorkerRegistryLike


def _services(request: Request) -> ApiServices:
    return cast(ApiServices, request.app.state.services)


def _settings(request: Request) -> ApiSettings:
    return cast(ApiSettings, request.app.state.settings)


def _task_service(request: Request) -> TaskService:
    return cast(TaskService, request.app.state.task_service)


def _upload_manager(request: Request) -> UploadManager:
    return cast(UploadManager, request.app.state.upload_manager)


def _load_project(repository: ProjectRepositoryLike) -> Project:
    try:
        return repository.load()
    except (OSError, ValueError) as error:
        raise ApiError(
            404,
            code="project_unavailable",
            category="project",
            message="The current project is unavailable.",
        ) from error


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
    destination = _confined_destination(
        services.project_repository.root, "source", source.name
    )
    size, sha256 = _copy_bounded(source, destination, settings.max_upload_size)
    relative = destination.relative_to(
        services.project_repository.root.resolve()
    ).as_posix()
    project = _load_project(services.project_repository)
    if asset.kind == AssetKind.SOURCE_VIDEO.value:
        project.source_video = relative
    else:
        project.scene_ply = relative
    services.project_repository.save(project)
    return AssetResponse(
        kind=asset.kind, path=relative, size=size, sha256=sha256
    )


def build_router() -> APIRouter:
    router = APIRouter()
    protected = APIRouter(dependencies=[Depends(require_session)])

    @protected.get("/healthz", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @protected.get("/api/v1/bootstrap", response_model=BootstrapResponse)
    async def bootstrap(request: Request) -> BootstrapResponse:
        services = _services(request)
        environment = await asyncio.to_thread(services.environment_doctor.check)
        return BootstrapResponse(
            api_version=API_VERSION,
            capabilities=("projects", "assets", "uploads", "tasks", "events"),
            project=_load_project(services.project_repository),
            environment=environment,
        )

    @protected.get("/api/v1/projects/current", response_model=Project)
    async def get_current_project(request: Request) -> Project:
        return _load_project(_services(request).project_repository)

    @protected.patch("/api/v1/projects/current", response_model=Project)
    async def patch_current_project(request: Request, patch: ProjectPatch) -> Project:
        repository = _services(request).project_repository
        project = _load_project(repository)
        if patch.name is not None:
            project.name = patch.name
        repository.save(project)
        return project

    @protected.post(
        "/api/v1/assets/import",
        response_model=AssetResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def import_asset(request: Request, asset: AssetImportRequest) -> AssetResponse:
        services = _services(request)
        settings = _settings(request)
        return await asyncio.to_thread(
            _import_asset_sync, services, settings, asset
        )

    @protected.post(
        "/api/v1/tasks",
        response_model=TaskSnapshot,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_task(request: Request, task: TaskCreateRequest) -> TaskSnapshot:
        return await _task_service(request).create(StageName(task.target_stage))

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

        def persist(completed: UploadComplete) -> None:
            project = _load_project(services.project_repository)
            project.source_video = completed.path
            services.project_repository.save(project)

        return await asyncio.to_thread(
            _upload_manager(request).complete, upload_id, persist
        )

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

    router.include_router(protected)
    return router
