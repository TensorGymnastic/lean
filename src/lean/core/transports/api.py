"""Universal FastAPI factory + bearer-auth + domain-error mappers.

Builds a FastAPI app with bearer-auth on every route except ``/health``.
Domains register their own routes via ``register_fn(app, services, settings)``.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from lean.core.config.settings import CoreSettings

_security = HTTPBearer(auto_error=False)
TokenCreds = Annotated[HTTPAuthorizationCredentials | None, Security(_security)]


async def _verify_token(creds: TokenCreds, settings: CoreSettings) -> None:
    expected = settings.api_key
    if not creds or not hmac.compare_digest(creds.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer token",
        )


def build_api(
    *,
    name: str,
    version: str,
    description: str = "",
    register_fn: Callable[..., Any] | None = None,
    settings: CoreSettings,
    services: dict[str, object] | None = None,
) -> FastAPI:
    """Build the FastAPI app with bearer-auth and the domain's routes."""
    app = FastAPI(
        title=name,
        version=version,
        description=description or f"{name} corpus API",
    )

    async def _token_dep(creds: TokenCreds = Security(_security)) -> None:
        await _verify_token(creds, settings)

    _ = _token_dep  # noqa: F841 — wired into register_fn below

    @app.exception_handler(ValueError)
    async def _value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})

    @app.exception_handler(PermissionError)
    async def _permission_error_handler(_request: Request, exc: PermissionError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def _not_found_handler(_request: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    if register_fn is not None:
        register_fn(app, services or {}, settings)

    return app


def serve_api(app: FastAPI, *, host: str, port: int, reload: bool = False) -> None:
    """Start the FastAPI server via uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, reload=reload)


__all__ = ["build_api", "serve_api"]
