from __future__ import annotations

import asyncio
import errno
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, Request, Response, WebSocket, status

from gs_video.api.auth import require_session
from gs_video.api.assets import AssetInspectorLike, ExportInspectorLike
from gs_video.api.events import EventBus, TaskService, serve_events
from gs_video.api.export_routes import build_export_router
from gs_video.api.schemas import (
    API_VERSION,
    ApiError,
    ApiSettings,
    AssetImportRequest,
    AssetKind,
    AssetResponse,
    BootstrapResponse,
    CameraConfirmationRequest,
    CameraInput,
    HealthResponse,
    PickRequest,
    PickResponse,
    PreviewFrameRequest,
    PreviewFrameResponse,
    ProjectPatch,
    TaskCreateRequest,
    TaskSnapshot,
    UploadComplete,
    UploadCreateRequest,
    UploadCreated,
    UploadStatus,
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
    StageName,
    StageState,
    SubjectPromptState,
)
from gs_video.environment.doctor import EnvironmentReport
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.workflow import ChangeKind, invalidate_for_change
from gs_video.scene.camera import OrbitCamera


class ProjectRepositoryLike(Protocol):
    root: Path

    def load(self) -> Project: ...

    def save(self, project: Project) -> None: ...

    def update(self, mutation: Callable[[Project], None]) -> Project: ...


class EnvironmentDoctorLike(Protocol):
    def check(self) -> EnvironmentReport: ...


class PipelineRunnerLike(Protocol):
    def run(self, name: StageName, token: CancellationToken) -> StageState: ...

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


def _services(request: Request) -> ApiServices:
    return cast(ApiServices, request.app.state.services)


def _settings(request: Request) -> ApiSettings:
    return cast(ApiSettings, request.app.state.settings)


def _task_service(request: Request) -> TaskService:
    return cast(TaskService, request.app.state.task_service)


def _upload_manager(request: Request) -> UploadManager:
    return cast(UploadManager, request.app.state.upload_manager)


def _preview_service(request: Request) -> PreviewServiceLike:
    return cast(PreviewServiceLike, request.app.state.preview_service)


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


def _replace_source_video(project: Project, relative: str, summary: Any) -> None:
    project.source_video = relative
    project.workflow.source_summary = summary
    project.workflow.subject_prompt = None
    _clear_preview_authority(project)
    project.workflow.export_result = None
    project.workflow.active_task_id = None
    invalidate_for_change(project, ChangeKind.SOURCE_VIDEO)


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

        def mutate(project: Project) -> None:
            if patch.name is not None:
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
                    validate_subject_prompt(project, repository.root, prompt)
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

        return await asyncio.to_thread(repository.update, mutate)

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
        if project.scene_ply is None or scene_summary is None:
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
        scene_path = project.scene_ply
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
                scene_path,
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
                latest.scene_ply != scene_path
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
        scene_path = project.scene_ply
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
                or latest.scene_ply != scene_path
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
        stage = StageName(task.target_stage)
        runner = _services(request).pipeline_runner
        supports = getattr(runner, "supports", None)
        if callable(supports) and not supports(stage):
            raise ApiError(
                503,
                code="workflow_unavailable",
                category="capability",
                message="The requested workflow stage is not assembled in this build.",
                retryable=False,
            )
        snapshot = await _task_service(request).create(stage)
        repository = _services(request).project_repository
        await asyncio.to_thread(
            repository.update,
            lambda project: setattr(
                project.workflow, "active_task_id", snapshot.id
            ),
        )
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

        def persist(completed: UploadComplete) -> None:
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

            services.project_repository.update(update_project)

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

    protected.include_router(build_subject_router())
    protected.include_router(build_export_router())
    router.include_router(protected)
    return router
