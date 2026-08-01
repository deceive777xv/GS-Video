from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
import json
from ipaddress import ip_address
import os
import socket
import sys
from typing import Any, Protocol, TextIO

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import SecretStr
from starlette.exceptions import HTTPException as StarletteHTTPException

from gs_video.api.events import EventBus, TaskService
from gs_video.api.assets import ExportInspector
from gs_video.api.middleware import LocalSecurityBoundary
from gs_video.api.routes import ApiServices, build_router
from gs_video.api.schemas import API_VERSION, ApiError, ApiSettings, ErrorEnvelope
from gs_video.api.uploads import UploadManager
from gs_video.api.workflow import PreviewArtifactStore, PreviewCoordinator
from gs_video.runtime import WorkflowRuntimeConfig, assemble_api_services


def _error_response(status_code: int, envelope: ErrorEnvelope) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=envelope.model_dump(mode="json"))


def create_app(settings: ApiSettings, services: ApiServices) -> FastAPI:
    if services.preview_service is None:
        raise ValueError("production API services require an explicit preview service")
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
                await app.state.task_service.request_cancel_all()
            finally:
                try:
                    await app.state.services.worker_registry.terminate_all()
                finally:
                    try:
                        repair = app.state.services.environment_repair
                        if repair is not None:
                            await repair.shutdown()
                    finally:
                        try:
                            await app.state.task_service.finish_shutdown()
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
    app.state.preview_service = services.preview_service
    app.state.preview_artifacts = PreviewArtifactStore(
        services.project_repository.root
    )
    app.state.preview_coordinator = PreviewCoordinator()
    app.state.export_inspector = services.export_inspector or ExportInspector()
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


class ServerLike(Protocol):
    should_exit: bool

    def run(self, *, sockets: list[socket.socket] | None = None) -> None: ...


def bind_api_socket(host: str) -> socket.socket:
    try:
        address = ip_address(host)
    except ValueError as error:
        raise ValueError("API socket host must be a loopback IP address") from error
    if not address.is_loopback:
        raise ValueError("API socket host must be a loopback IP address")
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    listener = socket.socket(family, socket.SOCK_STREAM)
    try:
        listener.bind((address.compressed, 0))
        listener.listen(socket.SOMAXCONN)
    except BaseException:
        listener.close()
        raise
    return listener


def run_server(
    app: FastAPI,
    *,
    bind_host: str,
    port: int,
    startup_handshake: bool,
    output: TextIO | None = None,
    server_factory: Callable[[Any], ServerLike] = uvicorn.Server,
    config_factory: Callable[..., Any] = uvicorn.Config,
) -> None:
    config = config_factory(
        app,
        host=bind_host,
        port=port,
        access_log=False,
        log_config=None,
    )
    server = server_factory(config)

    def request_shutdown() -> None:
        server.should_exit = True

    app.state.request_shutdown = request_shutdown
    if not startup_handshake:
        server.run()
        return

    listener = bind_api_socket(bind_host)
    try:
        actual_port = int(listener.getsockname()[1])
        handshake = {
            "port": actual_port,
            "apiVersion": API_VERSION,
            "pid": os.getpid(),
            "parentPid": os.getppid(),
        }
        stream = sys.stdout if output is None else output
        stream.write(json.dumps(handshake, separators=(",", ":")) + "\n")
        stream.flush()
        server.run(sockets=[listener])
    finally:
        listener.close()


def run_api(
    config: WorkflowRuntimeConfig,
    session_token: str,
    browser_origins: tuple[str, ...] = (),
    *,
    startup_handshake: bool = False,
) -> int:
    if (
        not session_token
        or len(session_token) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in session_token)
    ):
        raise ValueError("session token must contain between 1 and 4096 characters")
    settings, services = assemble_api_services(
        config,
        SecretStr(session_token),
        browser_origins=browser_origins,
    )
    app = create_app(settings, services)
    run_server(
        app,
        bind_host=settings.bind_host,
        port=settings.port,
        startup_handshake=startup_handshake,
    )
    return 0
