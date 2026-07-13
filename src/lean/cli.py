"""Typer CLI for lean: full parity with MCP tools."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="lean",
    help="Lean Six Sigma corpus MCP server and CLI.",
    no_args_is_help=True,
)


def _output(data: object, json_mode: bool) -> None:
    if hasattr(data, "model_dump_json"):
        typer.echo(data.model_dump_json(indent=2))
        return
    if isinstance(data, list):
        if json_mode:
            typer.echo(
                json.dumps(
                    [d.model_dump(mode="json") if hasattr(d, "model_dump") else d for d in data],
                    indent=2,
                    default=str,
                )
            )
        else:
            for item in data:
                if hasattr(item, "model_dump_json"):
                    typer.echo(item.model_dump_json(indent=2))
                else:
                    typer.echo(item)
    else:
        typer.echo(json.dumps(data, indent=2, default=str))


@app.command()
def ingest(
    path: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Ingest a PDF into the corpus."""
    from lean.mcp_server.tools import ingest_pdf

    result = asyncio.run(ingest_pdf(path))
    _output(result, json_output)


@app.command()
def search(
    query: str,
    k: int = typer.Option(5, help="Number of results"),
    doc_id: str = typer.Option(None, help="Filter by document UUID"),
    section: str = typer.Option(None, help="Filter by section substring"),
    author: str = typer.Option(None, help="Filter by author substring"),
    year_min: int = typer.Option(None, help="Filter by min publication year"),
    year_max: int = typer.Option(None, help="Filter by max publication year"),
    min_score: float = typer.Option(None, help="Minimum cosine similarity"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Semantic + hybrid search over the corpus."""
    from lean.retrieval.search import search as _search

    chunks = _search(
        query,
        k=k,
        doc_id=doc_id,
        section=section,
        author=author,
        year_min=year_min,
        year_max=year_max,
        min_score=min_score,
    )
    if json_output:
        _output(chunks, True)
        return
    if not chunks:
        typer.echo("No results found.")
        return
    for c in chunks:
        score_str = f"[{c.score:.4f}] " if c.score else ""
        typer.echo(f"{score_str}{c.section_path} (chunk {c.chunk_index})\n  {c.content[:200]}...\n")


@app.command(name="list-documents")
def list_documents(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List all documents in the corpus."""
    from lean.mcp_server.tools import list_documents as _list

    docs = asyncio.run(_list())
    if json_output:
        _output(docs, True)
        return
    for d in docs:
        auth = f"  {', '.join(d.authors)}" if d.authors else ""
        typer.echo(f"{d.id}  {d.title or '(untitled)'}  [{d.chunk_count} chunks]{auth}")


@app.command(name="corpus-stats")
def corpus_stats(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show corpus statistics."""
    from lean.mcp_server.tools import corpus_stats as _stats

    stats = asyncio.run(_stats())
    _output(stats, json_output)


@app.command(name="get-chunk")
def get_chunk(
    chunk_id: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Retrieve a single chunk by ID."""
    from lean.mcp_server.tools import get_chunk as _get

    chunk = asyncio.run(_get(chunk_id))
    if chunk is None:
        typer.echo("Chunk not found.", err=True)
        raise typer.Exit(1)
    _output(chunk, json_output)


@app.command(name="get-markdown")
def get_markdown(doc_id: str) -> None:
    """Get the extracted markdown for a document."""
    from lean.mcp_server.tools import get_document_markdown as _md

    md = asyncio.run(_md(doc_id))
    typer.echo(md)


@app.command()
def delete(doc_id: str) -> None:
    """Delete a document and all its chunks."""
    from lean.mcp_server.tools import delete_document as _del

    result = asyncio.run(_del(doc_id))
    typer.echo(f"Deleted: {result}")


@app.command()
def reingest(doc_id: str) -> None:
    """Re-ingest a document (re-extract with current settings)."""
    from lean.mcp_server.tools import reingest as _re

    result = asyncio.run(_re(doc_id))
    _output(result, False)


@app.command(name="mcp-serve")
def mcp_serve(
    transport: str = typer.Option("stdio", help="stdio or http"),
    port: int = typer.Option(None, help="HTTP port (default: from settings)"),
) -> None:
    """Run the MCP server."""
    from lean.settings import get_settings

    if port is None:
        port = get_settings().mcp_http_port
    sys.argv = ["lean-mcp", "--transport", transport, "--port", str(port)]
    from lean.mcp_server.__main__ import main

    main()


@app.command(name="db-init")
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
