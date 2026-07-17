"""Entry point: ``python -m lean.mcp_server [--transport stdio|http] [--port N]``."""

from __future__ import annotations

import argparse
import logging
import sys

from lean.config.settings import Settings
from lean.mcp_server.tools import mcp


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    settings = Settings()

    parser = argparse.ArgumentParser(description="lean MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--port", type=int, default=settings.mcp_http_port, help="HTTP port")
    parser.add_argument("--host", default=settings.mcp_http_host, help="HTTP bind host")
    args = parser.parse_args(argv)

    if args.transport == "http":
        import uvicorn

        from lean.auth.bearer import BearerTokenMiddleware

        app = mcp.http_app(transport="http")
        secured = BearerTokenMiddleware(app, token=settings.lean_mcp_api_key)
        uvicorn.run(secured, host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
