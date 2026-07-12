"""Typer CLI for lean: ingest, search, mcp-serve, db-init."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="lean",
    help="Lean Six Sigma corpus MCP server and CLI.",
    no_args_is_help=True,
)


@app.command()
def ingest(path: str) -> None:
    """Ingest a PDF into the corpus."""
    from lean.mcp_server.tools import ingest_pdf

    result = asyncio.run(ingest_pdf(path))
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def search(
    query: str,
    k: int = typer.Option(5, help="Number of results"),
) -> None:
    """Semantic search over the corpus."""
    from lean.retrieval.search import search as _search

    chunks = _search(query, k=k)
    if not chunks:
        typer.echo("No results found.")
        return
    for c in chunks:
        score_str = f"[{c.score:.3f}] " if c.score else ""
        typer.echo(f"{score_str}{c.section_path} (chunk {c.chunk_index})\n  {c.content[:200]}...\n")


@app.command()
def mcp_serve(
    transport: str = typer.Option("stdio", help="stdio or http"),
    port: int = typer.Option(8765, help="HTTP port"),
) -> None:
    """Run the MCP server."""
    sys.argv = ["lean-mcp", "--transport", transport, "--port", str(port)]
    from lean.mcp_server.__main__ import main

    main()


@app.command()
def db_init() -> None:
    """Apply db/schemas/*.sql to local Supabase."""
    import os

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        typer.echo("Error: SUPABASE_DB_URL not set", err=True)
        raise typer.Exit(1)
    for sql_file in sorted(Path("db/schemas").glob("*.sql")):
        typer.echo(f"applying {sql_file}...")
        subprocess.run(["psql", db_url, "-f", str(sql_file)], check=True)
    typer.echo("schema applied.")


if __name__ == "__main__":
    app()
