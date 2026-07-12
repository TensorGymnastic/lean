"""Entry point: ``python -m lean.mcp_server [--transport stdio|http] [--port N]``."""

from __future__ import annotations

import argparse
import logging
import sys

from lean.mcp_server.tools import mcp
from lean.settings import Settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    parser = argparse.ArgumentParser(description="lean MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    args = parser.parse_args()

    Settings()

    if args.transport == "http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
