from __future__ import annotations

import asyncio
import secrets
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import SecretStr
from starlette.exceptions import HTTPException as StarletteHTTPException

from gs_video.api.events import EventBus, TaskService
from gs_video.api.middleware import LocalSecurityBoundary
from gs_video.api.routes import ApiServices, build_router
from gs_video.api.schemas import ApiError, ApiSettings, ErrorEnvelope
from gs_video.api.uploads import UploadManager
from gs_video.environment.doctor import EnvironmentDoctor
from gs_video.pipeline.runner import PipelineRunner
from gs_video.project.repository import ProjectRepository


def _error_response(status_code: int, envelope: ErrorEnvelope) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=envelope.model_dump(mode="json"))


def create_app(settings: ApiSettings, services: ApiServices) -> FastAPI:
    event_bus = EventBus(settings.event_window)
    task_service = TaskService(
        services.pipeline_runner,
        event_bus,
        max_tasks=settings.max_tasks,
        workers=settings.task_workers,
        shutdown_timeout=settings.shutdown_timeout,
    )
    upload_manager = UploadManager(
        services.project_repository.root,
        settings.max_upload_size,
        settings.max_active_uploads,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            await app.state.task_service.start()
            yield
        finally:
            try:
                await app.state.task_service.cancel_all()
            finally:
                try:
                    await app.state.services.worker_registry.terminate_all()
                finally:
                    await asyncio.to_thread(app.state.upload_manager.close)

    app = FastAPI(
        title="GS Video local API",
        version="1",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.services = services
    app.state.event_bus = event_bus
    app.state.task_service = task_service
    app.state.upload_manager = upload_manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.add_middleware(LocalSecurityBoundary, settings=settings)

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, error: ApiError) -> JSONResponse:
        del request
        return _error_response(error.status_code, error.envelope)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        del request, error
        return _error_response(
            422,
            ErrorEnvelope(
                code="invalid_request",
                category="validation",
                message="The request did not match the API contract.",
                retryable=False,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        del request
        code = "not_found" if error.status_code == 404 else "http_error"
        return _error_response(
            error.status_code,
            ErrorEnvelope(
                code=code,
                category="routing",
                message="The requested API resource was not found."
                if error.status_code == 404
                else "The request could not be completed.",
                retryable=False,
            ),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
        del request, error
        return _error_response(
            500,
            ErrorEnvelope(
                code="internal_error",
                category="internal",
                message="The local service encountered an unexpected error.",
                retryable=False,
            ),
        )

    app.include_router(build_router())
    return app


def run_api(host: str, port: int) -> int:
    settings = ApiSettings(
        bind_host=host,
        port=port,
        session_token=SecretStr(secrets.token_urlsafe(32)),
        allowed_origins=(),
    )
    with tempfile.TemporaryDirectory(prefix="gs-video-api-") as temporary:
        repository = ProjectRepository(Path(temporary) / "project")
        project = repository.create("GS Video session")
        repository.save(project)
        services = ApiServices(
            project_repository=repository,
            environment_doctor=EnvironmentDoctor(),
            pipeline_runner=PipelineRunner(project, {}, save=repository.save),
            worker_registry=_NoopWorkerRegistry(),
        )
        app = create_app(settings, services)
        try:
            uvicorn.run(
                app,
                host=settings.bind_host,
                port=settings.port,
                access_log=False,
                log_config=None,
            )
        finally:
            app.state.upload_manager.close()
    return 0


class _NoopWorkerRegistry:
    async def terminate_all(self) -> None:
        return None
