from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from gs_video.api.auth import valid_bearer_header
from gs_video.api.schemas import ApiSettings, ErrorEnvelope


ALLOWED_CORS_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})
ALLOWED_CORS_HEADERS = frozenset({"authorization", "content-type"})


class LocalSecurityBoundary:
    def __init__(self, app: ASGIApp, *, settings: ApiSettings) -> None:
        self._app = app
        self._settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._is_protected(scope):
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        method = str(scope.get("method", "")).upper()
        if self._is_preflight(method, headers):
            if not self._cors_allowed(headers):
                await self._reject(
                    scope,
                    receive,
                    send,
                    400,
                    ErrorEnvelope(
                        code="cors_forbidden",
                        category="cors",
                        message="The CORS preflight request is not allowed.",
                        retryable=False,
                    ),
                )
                return
            await self._app(scope, receive, send)
            return

        if not valid_bearer_header(headers.get("authorization"), self._settings):
            await self._reject(
                scope,
                receive,
                send,
                401,
                ErrorEnvelope(
                    code="invalid_session",
                    category="authentication",
                    message="A valid Bearer session token is required.",
                    retryable=False,
                ),
            )
            return
        origin = headers.get("origin")
        if origin is not None and origin not in self._settings.allowed_origins:
            await self._reject(
                scope,
                receive,
                send,
                400,
                ErrorEnvelope(
                    code="cors_forbidden",
                    category="cors",
                    message="The request Origin is not allowed.",
                    retryable=False,
                ),
            )
            return

        limit = self._body_limit(scope)
        declared = headers.get("content-length")
        if declared is not None:
            try:
                declared_size = int(declared)
            except ValueError:
                await self._invalid_content_length(scope, receive, send)
                return
            if declared_size < 0:
                await self._invalid_content_length(scope, receive, send)
                return
            if declared_size > limit:
                await self._body_too_large(scope, receive, send)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            block = message.get("body", b"")
            if len(body) + len(block) > limit:
                await self._body_too_large(scope, receive, send)
                return
            body.extend(block)
            if not message.get("more_body", False):
                break

        replay = _replay_body(bytes(body))
        await self._app(scope, replay, send)

    @staticmethod
    def _is_protected(scope: Scope) -> bool:
        path = str(scope.get("path", ""))
        return path == "/healthz" or path.startswith("/api/v1/")

    @staticmethod
    def _is_preflight(method: str, headers: Headers) -> bool:
        return (
            method == "OPTIONS"
            and headers.get("origin") is not None
            and headers.get("access-control-request-method") is not None
        )

    def _cors_allowed(self, headers: Headers) -> bool:
        origin = headers.get("origin")
        method = headers.get("access-control-request-method", "").upper()
        requested_headers = {
            value.strip().lower()
            for value in headers.get("access-control-request-headers", "").split(",")
            if value.strip()
        }
        return (
            origin in self._settings.allowed_origins
            and method in ALLOWED_CORS_METHODS
            and requested_headers <= ALLOWED_CORS_HEADERS
        )

    def _body_limit(self, scope: Scope) -> int:
        path = str(scope.get("path", ""))
        method = str(scope.get("method", "")).upper()
        if method == "PUT" and path.startswith("/api/v1/uploads/") and "/chunks/" in path:
            return self._settings.max_chunk_body_size
        return self._settings.max_json_body_size

    async def _invalid_content_length(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        await self._reject(
            scope,
            receive,
            send,
            400,
            ErrorEnvelope(
                code="invalid_content_length",
                category="validation",
                message="The request Content-Length is invalid.",
                retryable=False,
            ),
        )

    async def _body_too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._reject(
            scope,
            receive,
            send,
            413,
            ErrorEnvelope(
                code="request_body_too_large",
                category="validation",
                message="The request body exceeds the configured limit.",
                retryable=False,
            ),
        )

    async def _reject(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        status_code: int,
        envelope: ErrorEnvelope,
    ) -> None:
        origin = Headers(scope=scope).get("origin")
        response_headers: dict[str, str] = {}
        if origin in self._settings.allowed_origins:
            response_headers = {
                "Access-Control-Allow-Origin": str(origin),
                "Vary": "Origin",
            }
        response = JSONResponse(
            status_code=status_code,
            content=envelope.model_dump(mode="json"),
            headers=response_headers,
        )
        await response(scope, receive, send)


def _replay_body(body: bytes) -> Callable[[], Awaitable[Message]]:
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive
