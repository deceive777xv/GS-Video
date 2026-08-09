from __future__ import annotations

import asyncio
import hashlib
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from fastapi import APIRouter, Request, Response, status

from gs_video.api.assets import ExportInspectorLike
from gs_video.api.schemas import (
    ApiError,
    ApiSettings,
    ExportCopyRequest,
    VerifiedExportResponse,
)
from gs_video.domain.errors import GsVideoError, RepairableError
from gs_video.domain.models import (
    ArtifactRef,
    ArtifactRole,
    ExportResultState,
    Project,
    StageName,
    StageStatus,
)
from gs_video.media.export import copy_verified_export
from gs_video.segmentation.paths import has_reparse_component
from gs_video.storage.artifacts import ArtifactStore


class ProjectRepositoryLike(Protocol):
    root: Path

    def load(self) -> Project: ...

    def update(self, mutation: Callable[[Project], None]) -> Project: ...


class ServicesLike(Protocol):
    project_repository: ProjectRepositoryLike
    artifact_store: ArtifactStore


def _services(request: Request) -> ServicesLike:
    return cast(ServicesLike, request.app.state.services)


def _settings(request: Request) -> ApiSettings:
    return cast(ApiSettings, request.app.state.settings)


def _export_inspector(request: Request) -> ExportInspectorLike:
    return cast(ExportInspectorLike, request.app.state.export_inspector)


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


def _artifact_fingerprint(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_nlink),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_stable_artifact(
    path: Path, expected_stat: os.stat_result, *, limit: int
) -> tuple[bytes, str]:
    captured_size = int(expected_stat.st_size)
    if captured_size <= 0 or captured_size > limit:
        raise ApiError(
            409,
            code="artifact_changed",
            category="filesystem",
            message="The project artifact identity changed.",
        )
    digest = hashlib.sha256()
    payload = bytearray()
    try:
        with path.open("rb") as stream:
            opened_stat = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(opened_stat.st_mode)
                or opened_stat.st_nlink != 1
                or _artifact_fingerprint(opened_stat)
                != _artifact_fingerprint(expected_stat)
            ):
                raise ApiError(
                    409,
                    code="artifact_changed",
                    category="filesystem",
                    message="The project artifact identity changed.",
                )
            while True:
                remaining_with_sentinel = captured_size - len(payload) + 1
                block = stream.read(min(1024 * 1024, remaining_with_sentinel))
                if not block:
                    break
                payload.extend(block)
                if len(payload) > captured_size:
                    raise ApiError(
                        409,
                        code="artifact_changed",
                        category="filesystem",
                        message="The project artifact exceeded its captured size.",
                    )
                digest.update(block)
            final_handle_stat = os.fstat(stream.fileno())
        final_path_stat = path.stat()
    except ApiError:
        raise
    except OSError as error:
        raise ApiError(
            409,
            code="artifact_changed",
            category="filesystem",
            message="The project artifact could not be read safely.",
        ) from error
    expected_fingerprint = _artifact_fingerprint(expected_stat)
    if (
        len(payload) != captured_size
        or _artifact_fingerprint(final_handle_stat) != expected_fingerprint
        or _artifact_fingerprint(final_path_stat) != expected_fingerprint
        or has_reparse_component(path)
    ):
        raise ApiError(
            409,
            code="artifact_changed",
            category="filesystem",
            message="The project artifact identity changed while it was read.",
        )
    return bytes(payload), digest.hexdigest()


def _verified_artifact_state(
    root: Path,
    relative: str,
    *,
    directory: str,
    expected_size: int | None,
    limit: int,
) -> tuple[Path, os.stat_result]:
    canonical_root = root.resolve()
    allowed_root = (canonical_root / directory).resolve()
    try:
        candidate = canonical_root / relative
        path = candidate.resolve(strict=True)
        path_stat = path.stat()
    except OSError as error:
        raise ApiError(
            404,
            code="artifact_unavailable",
            category="project",
            message="The requested project artifact is unavailable.",
        ) from error
    if (
        not path.is_relative_to(allowed_root)
        or has_reparse_component(candidate)
        or has_reparse_component(path)
        or not stat.S_ISREG(path_stat.st_mode)
        or (int(path_stat.st_dev), int(path_stat.st_ino)) == (0, 0)
        or path_stat.st_nlink != 1
        or path_stat.st_size <= 0
        or path_stat.st_size > limit
        or (expected_size is not None and path_stat.st_size != expected_size)
    ):
        raise ApiError(
            409,
            code="artifact_changed",
            category="filesystem",
            message="The project artifact identity changed.",
        )
    return path, path_stat


def _read_verified_artifact(
    root: Path,
    relative: str,
    *,
    directory: str,
    expected_size: int | None,
    expected_sha256: str | None,
    limit: int,
) -> tuple[Path, bytes, str]:
    path, path_stat = _verified_artifact_state(
        root,
        relative,
        directory=directory,
        expected_size=expected_size,
        limit=limit,
    )
    payload, digest = _read_stable_artifact(path, path_stat, limit=limit)
    if expected_size is not None and len(payload) != expected_size:
        raise ApiError(
            409,
            code="artifact_changed",
            category="filesystem",
            message="The project artifact identity changed.",
        )
    if expected_sha256 is not None and digest != expected_sha256:
        raise ApiError(
            409,
            code="artifact_changed",
            category="filesystem",
            message="The project artifact identity changed.",
        )
    return path, payload, digest


def _verified_artifact_path(
    root: Path,
    relative: str,
    *,
    directory: str,
    expected_size: int | None,
    limit: int,
) -> Path:
    path, _path_stat = _verified_artifact_state(
        root,
        relative,
        directory=directory,
        expected_size=expected_size,
        limit=limit,
    )
    return path


def _authoritative_export_relative(
    project: Project, *, not_ready: bool = False
) -> tuple[ArtifactRef, str]:
    stage = project.stages.get(StageName.EXPORT)
    registered = (
        None if stage is None else stage.artifacts.get(ArtifactRole.EXPORT_VIDEO)
    )
    if (
        stage is None
        or stage.status is not StageStatus.SUCCEEDED
        or stage.cache_key is None
        or registered is None
        or not stage.output_paths
        or stage.output_paths[-1] != registered
    ):
        raise ApiError(
            409,
            code="export_not_ready" if not_ready else "export_changed",
            category="project" if not_ready else "conflict",
            message=(
                "A verified export is not ready."
                if not_ready
                else "The verified export is no longer authoritative."
            ),
        )
    return registered, stage.cache_key


def _opaque_export_id(cache_key: str, sha256: str) -> str:
    return hashlib.sha256(f"export\0{cache_key}\0{sha256}".encode()).hexdigest()[:32]


def _validate_export_descriptor_authority(
    export: ExportResultState, cache_key: str
) -> None:
    if export.artifact_id != _opaque_export_id(cache_key, export.sha256):
        raise ApiError(
            409,
            code="export_changed",
            category="conflict",
            message="The verified export descriptor is stale.",
        )


def _external_export_destination(project_root: Path, raw_destination: str) -> Path:
    try:
        canonical_root = project_root.resolve(strict=True)
        requested = Path(raw_destination).expanduser()
        if not requested.is_absolute():
            raise ValueError("destination must be absolute")
        destination = requested.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ApiError(
            400,
            code="invalid_export_destination",
            category="filesystem",
            message="The selected export destination is unsafe or invalid.",
        ) from error
    if destination == canonical_root or destination.is_relative_to(canonical_root):
        raise ApiError(
            400,
            code="invalid_export_destination",
            category="filesystem",
            message="The export destination must be outside the project directory.",
        )
    return destination


def build_export_router() -> APIRouter:
    router = APIRouter()

    @router.get(
        "/api/v1/projects/current/export",
        response_model=VerifiedExportResponse,
    )
    async def get_verified_export(request: Request) -> VerifiedExportResponse:
        repository = _services(request).project_repository
        artifact_store = _services(request).artifact_store
        project = _load_project(repository)
        relative, export_cache_key = _authoritative_export_relative(
            project, not_ready=True
        )
        artifact_root = artifact_store.lookup_project_root(project.project_id)
        artifact_limit = _settings(request).max_artifact_response_size
        path, payload, digest = await asyncio.to_thread(
            _read_verified_artifact,
            artifact_root,
            relative.relative_path(),
            directory="exports",
            expected_size=None,
            expected_sha256=None,
            limit=artifact_limit,
        )
        metadata = await asyncio.to_thread(_export_inspector(request).probe, path)
        await asyncio.to_thread(
            _read_verified_artifact,
            artifact_root,
            relative.relative_path(),
            directory="exports",
            expected_size=len(payload),
            expected_sha256=digest,
            limit=artifact_limit,
        )
        if metadata.frame_count is None:
            raise ApiError(
                409,
                code="export_verification_incomplete",
                category="export",
                message="The exported video did not report a verified frame count.",
            )
        result = ExportResultState(
            artifact_id=_opaque_export_id(export_cache_key, digest),
            filename=path.name,
            size=len(payload),
            sha256=digest,
            duration_seconds=metadata.duration,
            fps=str(metadata.fps),
            frame_count=metadata.frame_count,
            has_audio=metadata.has_audio,
            verified=True,
        )

        def persist(latest: Project) -> None:
            try:
                latest_relative, latest_cache_key = _authoritative_export_relative(
                    latest
                )
            except ApiError as error:
                raise ApiError(
                    409,
                    code="export_changed",
                    category="conflict",
                    message="The export changed while it was being verified.",
                    retryable=True,
                ) from error
            if latest_relative != relative or latest_cache_key != export_cache_key:
                raise ApiError(
                    409,
                    code="export_changed",
                    category="conflict",
                    message="The export changed while it was being verified.",
                    retryable=True,
                )
            latest.workflow.export_result = result

        await asyncio.to_thread(repository.update, persist)
        return VerifiedExportResponse.model_validate(
            result.model_dump(exclude={"sha256"})
        )

    @router.get("/api/v1/projects/current/exports/{artifact_id}")
    async def get_export_artifact(request: Request, artifact_id: str) -> Response:
        services = _services(request)
        project = _load_project(services.project_repository)
        export = project.workflow.export_result
        if export is None or export.artifact_id != artifact_id or not export.verified:
            raise ApiError(
                404,
                code="export_unavailable",
                category="project",
                message="The requested export is unavailable.",
            )
        artifact_limit = _settings(request).max_artifact_response_size
        if export.size <= 0 or export.size > artifact_limit:
            raise ApiError(
                409,
                code="export_changed",
                category="conflict",
                message="The verified export is outside the configured size bound.",
            )
        relative, export_cache_key = _authoritative_export_relative(project)
        artifact_root = services.artifact_store.lookup_project_root(project.project_id)
        _validate_export_descriptor_authority(export, export_cache_key)
        _path, payload, _digest = await asyncio.to_thread(
            _read_verified_artifact,
            artifact_root,
            relative.relative_path(),
            directory="exports",
            expected_size=export.size,
            expected_sha256=export.sha256,
            limit=artifact_limit,
        )
        return Response(
            content=payload,
            media_type="video/mp4",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @router.post(
        "/api/v1/projects/current/exports/{artifact_id}/copy",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def copy_export_artifact(
        request: Request, artifact_id: str, body: ExportCopyRequest
    ) -> Response:
        services = _services(request)
        project = _load_project(services.project_repository)
        export = project.workflow.export_result
        if export is None or export.artifact_id != artifact_id or not export.verified:
            raise ApiError(
                404,
                code="export_unavailable",
                category="project",
                message="The requested export is unavailable.",
            )
        artifact_limit = _settings(request).max_artifact_response_size
        if export.size <= 0 or export.size > artifact_limit:
            raise ApiError(
                409,
                code="export_changed",
                category="conflict",
                message="The verified export is outside the configured size bound.",
            )
        relative, export_cache_key = _authoritative_export_relative(project)
        artifact_root = services.artifact_store.lookup_project_root(project.project_id)
        _validate_export_descriptor_authority(export, export_cache_key)
        source = await asyncio.to_thread(
            _verified_artifact_path,
            artifact_root,
            relative.relative_path(),
            directory="exports",
            expected_size=export.size,
            limit=artifact_limit,
        )
        destination = _external_export_destination(
            services.project_repository.root, body.destination
        )
        export_snapshot = export.model_copy(deep=True)

        def revalidate_authority() -> None:
            latest = _load_project(services.project_repository)
            latest_relative, latest_cache_key = _authoritative_export_relative(latest)
            latest_export = latest.workflow.export_result
            if (
                latest_relative != relative
                or latest_cache_key != export_cache_key
                or latest_export is None
                or latest_export != export_snapshot
            ):
                raise ApiError(
                    409,
                    code="export_changed",
                    category="conflict",
                    message="The verified export changed before it was copied.",
                    retryable=True,
                )
            _validate_export_descriptor_authority(latest_export, latest_cache_key)

        try:
            await asyncio.to_thread(
                copy_verified_export,
                source,
                destination,
                expected_size=export.size,
                expected_sha256=export.sha256,
                before_publish=revalidate_authority,
            )
        except RepairableError as error:
            raise ApiError(
                400,
                code="invalid_export_destination",
                category="filesystem",
                message="The selected export destination is unsafe or invalid.",
            ) from error
        except GsVideoError as error:
            raise ApiError(
                409,
                code="export_copy_failed",
                category="filesystem",
                message="The verified export could not be copied safely.",
                retryable=True,
            ) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
