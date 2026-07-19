"""Universal FastMCP factory.

Builds a FastMCP app, registers the domain's tools/resources/prompts,
and exposes ``serve()`` for stdio/http transports.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP

from lean.core.auth.bearer import BearerTokenMiddleware
from lean.core.config.settings import CoreSettings

logger = logging.getLogger(__name__)


def build_mcp(
    *,
    name: str,
    version: str,
    description: str = "",
    register_fn: Callable[..., Any] | None = None,
    services: dict[str, object] | None = None,
    settings: CoreSettings | None = None,
) -> FastMCP:
    """Build the FastMCP app and let the domain register its tools.

    ``register_fn`` is ``domain.register_mcp(mcp, services, settings)``.
    Services are constructed by ``TransportBuilder`` and passed in.
    """
    mcp = FastMCP(name)
    if register_fn is not None:
        register_fn(mcp, services or {}, settings)
    return mcp


def serve_mcp(
    mcp: FastMCP,
    *,
    transport: str,
    host: str,
    port: int,
    settings: CoreSettings,
) -> None:
    """Start the MCP server. ``transport`` is 'stdio' or 'http'."""
    if transport == "http":
        import uvicorn

        app = mcp.http_app(transport="http")
        secured = BearerTokenMiddleware(app, token=settings.api_key)
        uvicorn.run(secured, host=host, port=port)
    else:
        mcp.run(transport="stdio")


__all__ = ["build_mcp", "serve_mcp"]
