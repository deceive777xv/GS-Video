from __future__ import annotations

import secrets
from typing import Annotated, cast

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gs_video.api.schemas import ApiError, ApiSettings


_bearer = HTTPBearer(auto_error=False)


def get_settings(request: Request) -> ApiSettings:
    return cast(ApiSettings, request.app.state.settings)


def get_services(request: Request) -> object:
    return request.app.state.services


def require_session(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[ApiSettings, Depends(get_settings)],
) -> None:
    valid = (
        credentials is not None
        and credentials.scheme.lower() == "bearer"
        and secrets.compare_digest(credentials.credentials, settings.session_token)
    )
    if not valid:
        raise ApiError(
            401,
            code="invalid_session",
            category="authentication",
            message="A valid Bearer session token is required.",
        )
