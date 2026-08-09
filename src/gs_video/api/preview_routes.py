from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Protocol, cast

from fastapi import APIRouter, Request, Response

from gs_video.api.assets import CompositePreviewInspectorLike
from gs_video.api.export_routes import (
    _artifact_fingerprint,
    _read_stable_artifact,
    _verified_artifact_state,
)
from gs_video.api.schemas import ApiError, CompositePreviewResponse
from gs_video.domain.models import (
    ArtifactCategory,
    ArtifactRef,
    ArtifactRole,
    Project,
    StageName,
    StageStatus,
)
from gs_video.storage.artifacts import ArtifactStore


_COMPOSITE_PREVIEW_LIMIT = 256 * 1024 * 1024


class ProjectRepositoryLike(Protocol):
    root: Path

    def load(self) -> Project: ...


class ServicesLike(Protocol):
    project_repository: ProjectRepositoryLike
    artifact_store: ArtifactStore


def _services(request: Request) -> ServicesLike:
    return cast(ServicesLike, request.app.state.services)


def _preview_inspector(request: Request) -> CompositePreviewInspectorLike:
    return cast(CompositePreviewInspectorLike, request.app.state.export_inspector)


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


def _preview_changed() -> ApiError:
    return ApiError(
        409,
        code="composite_preview_changed",
        category="conflict",
        message="The composite preview is no longer authoritative.",
    )


def _authoritative_preview_relative(project: Project) -> tuple[ArtifactRef, str]:
    stage = project.stages.get(StageName.COMPOSITE)
    if stage is None or stage.cache_key is None:
        raise _preview_changed()
    expected = ArtifactRef(
        project_id=project.project_id,
        category=ArtifactCategory.PREVIEWS,
        cache_key=stage.cache_key,
        member="composite-preview.mp4",
    )
    registered = stage.artifacts.get(ArtifactRole.COMPOSITE_PREVIEW)
    if (
        stage.status is not StageStatus.SUCCEEDED
        or registered != expected
        or not stage.output_paths
        or stage.output_paths[-1] != expected
    ):
        raise _preview_changed()
    return expected, stage.cache_key


def _revalidate_preview_authority(
    repository: ProjectRepositoryLike,
    project: Project,
    relative: ArtifactRef,
    cache_key: str,
) -> None:
    latest = _load_project(repository)
    latest_relative, latest_cache_key = _authoritative_preview_relative(latest)
    if (
        latest.project_id != project.project_id
        or latest_relative != relative
        or latest_cache_key != cache_key
    ):
        raise _preview_changed()


def _opaque_preview_id(
    project_id: str,
    cache_key: str,
    identity: tuple[int, int, int, int, int, int],
    size: int,
    sha256: str,
) -> str:
    material = "\0".join(
        (
            "composite-preview",
            project_id,
            cache_key,
            *(str(part) for part in identity),
            str(size),
            sha256,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _read_current_preview(
    root: Path, relative: str
) -> tuple[Path, bytes, str, tuple[int, int, int, int, int, int]]:
    try:
        path, path_stat = _verified_artifact_state(
            root,
            relative,
            directory="previews",
            expected_size=None,
            limit=_COMPOSITE_PREVIEW_LIMIT,
        )
        identity = _artifact_fingerprint(path_stat)
        payload, digest = _read_stable_artifact(
            path, path_stat, limit=_COMPOSITE_PREVIEW_LIMIT
        )
    except ApiError as error:
        raise _preview_changed() from error
    return path, payload, digest, identity


def _revalidate_preview_identity(
    root: Path,
    relative: str,
    *,
    expected_identity: tuple[int, int, int, int, int, int],
    expected_size: int,
    expected_sha256: str,
) -> None:
    try:
        path, path_stat = _verified_artifact_state(
            root,
            relative,
            directory="previews",
            expected_size=expected_size,
            limit=_COMPOSITE_PREVIEW_LIMIT,
        )
        if _artifact_fingerprint(path_stat) != expected_identity:
            raise _preview_changed()
        _payload, digest = _read_stable_artifact(
            path, path_stat, limit=_COMPOSITE_PREVIEW_LIMIT
        )
    except ApiError as error:
        if error.envelope.code == "composite_preview_changed":
            raise
        raise _preview_changed() from error
    if digest != expected_sha256:
        raise _preview_changed()


def _resolve_composite_preview(
    repository: ProjectRepositoryLike,
    artifact_store: ArtifactStore,
    inspector: CompositePreviewInspectorLike,
) -> CompositePreviewResponse:
    project = _load_project(repository)
    relative, cache_key = _authoritative_preview_relative(project)
    root = artifact_store.lookup_project_root(project.project_id)
    path, payload, digest, identity = _read_current_preview(
        root, relative.relative_path()
    )
    try:
        metadata = inspector.probe(path)
    except Exception as error:
        raise ApiError(
            409,
            code="composite_preview_invalid",
            category="media",
            message="The composite preview could not be verified.",
        ) from error
    _revalidate_preview_identity(
        root,
        relative.relative_path(),
        expected_identity=identity,
        expected_size=len(payload),
        expected_sha256=digest,
    )
    _revalidate_preview_authority(repository, project, relative, cache_key)
    if metadata.frame_count is None:
        raise ApiError(
            409,
            code="composite_preview_verification_incomplete",
            category="media",
            message="The composite preview did not report a frame count.",
        )
    return CompositePreviewResponse(
        artifact_id=_opaque_preview_id(
            project.project_id, cache_key, identity, len(payload), digest
        ),
        filename="composite-preview.mp4",
        size=len(payload),
        sha256=digest,
        duration_seconds=metadata.duration,
        fps=str(metadata.fps),
        frame_count=metadata.frame_count,
    )


def _read_composite_preview_blob(
    repository: ProjectRepositoryLike,
    artifact_store: ArtifactStore,
    artifact_id: str,
) -> bytes:
    project = _load_project(repository)
    relative, cache_key = _authoritative_preview_relative(project)
    root = artifact_store.lookup_project_root(project.project_id)
    _path, payload, digest, identity = _read_current_preview(
        root, relative.relative_path()
    )
    current_id = _opaque_preview_id(
        project.project_id, cache_key, identity, len(payload), digest
    )
    if current_id != artifact_id:
        raise ApiError(
            404,
            code="composite_preview_unavailable",
            category="project",
            message="The requested composite preview is unavailable.",
        )
    _revalidate_preview_authority(repository, project, relative, cache_key)
    _revalidate_preview_identity(
        root,
        relative.relative_path(),
        expected_identity=identity,
        expected_size=len(payload),
        expected_sha256=digest,
    )
    return payload


def build_composite_preview_router() -> APIRouter:
    router = APIRouter()

    @router.get(
        "/api/v1/projects/current/composite-preview",
        response_model=CompositePreviewResponse,
    )
    async def get_composite_preview(request: Request) -> CompositePreviewResponse:
        services = _services(request)
        repository = services.project_repository
        return await asyncio.to_thread(
            _resolve_composite_preview,
            repository,
            services.artifact_store,
            _preview_inspector(request),
        )

    @router.get("/api/v1/artifacts/composite-previews/{artifact_id}")
    async def get_composite_preview_artifact(
        request: Request, artifact_id: str
    ) -> Response:
        payload = await asyncio.to_thread(
            _read_composite_preview_blob,
            _services(request).project_repository,
            _services(request).artifact_store,
            artifact_id,
        )
        return Response(
            content=payload,
            media_type="video/mp4",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return router
