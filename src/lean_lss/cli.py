"""lean-lss CLI entry point — built on lean-core's TransportBuilder.

Usage:
    lean ingest <path>
    lean search "..."
    lean list-documents
    lean health
    lean mcp-serve
    lean api-serve
    lean db-init
"""

from __future__ import annotations

from lean_core.transports import TransportBuilder

from lean_lss.domain import LssDomain


def main() -> None:
    """Run the lean-lss Typer CLI."""
    builder = TransportBuilder(LssDomain())
    builder.cli()


if __name__ == "__main__":
    main()
