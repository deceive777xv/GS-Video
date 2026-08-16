from __future__ import annotations

import asyncio
from typing import Any, cast

from fastapi import APIRouter, Request, Response

from gs_video.api.schemas import ApiError, SubjectMediaResponse, SubjectMediaRole
from gs_video.api.workflow import resolve_subject_media
from gs_video.domain.models import Project


def _repository(request: Request) -> Any:
    return request.app.state.services.project_repository


def _artifact_root(request: Request, project: Project, repository: Any) -> Any:
    store = request.app.state.services.artifact_store
    return (
        repository.root
        if store is None
        else store.lookup_project_root(project.project_id)
    )


def _load_project(repository: Any) -> Project:
    try:
        return cast(Project, repository.load())
    except (OSError, ValueError) as error:
        raise ApiError(
            404,
            code="project_unavailable",
            category="project",
            message="The current project is unavailable.",
        ) from error


def build_subject_router() -> APIRouter:
    router = APIRouter()

    @router.get(
        "/api/v1/projects/current/subject-media/{role}",
        response_model=SubjectMediaResponse,
    )
    async def get_subject_media(
        request: Request, role: SubjectMediaRole, frame_index: int | None = None
    ) -> SubjectMediaResponse:
        repository = _repository(request)
        project = _load_project(repository)
        resolved = await asyncio.to_thread(
            resolve_subject_media,
            project,
            _artifact_root(request, project, repository),
            role,
            frame_index,
        )
        return SubjectMediaResponse(
            role=resolved.role,
            artifact_id=resolved.artifact_id,
            frame_index=resolved.frame_index,
            width=resolved.width,
            height=resolved.height,
            size=resolved.size,
            mime_type=resolved.mime_type,
        )

    @router.get(
        "/api/v1/projects/current/subject-media/{role}/{artifact_id}"
    )
    async def get_subject_media_artifact(
        request: Request,
        role: SubjectMediaRole,
        artifact_id: str,
        frame_index: int | None = None,
    ) -> Response:
        repository = _repository(request)
        project = _load_project(repository)
        resolved = await asyncio.to_thread(
            resolve_subject_media,
            project,
            _artifact_root(request, project, repository),
            role,
            frame_index,
        )
        if resolved.artifact_id != artifact_id:
            raise ApiError(
                404,
                code="subject_media_unavailable",
                category="project",
                message="The requested subject media is unavailable.",
            )
        return Response(
            content=resolved.payload,
            media_type=resolved.mime_type,
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return router
