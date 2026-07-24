"""Code domain — tool surface.

Universal tools are imported from ``lean.core.tools.universal``. This
module keeps only code-specific entry points: ingest_file,
ingest_directory, and search with section/author filters.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from lean.core.adapters import cli_command, mcp_tool, output, rest_route
from lean.core.models import Chunk, IngestResult
from lean.core.services.ingestion import ingest_pdf as _ingest_pdf
from lean.core.services.search import search as _search
from lean.core.tools.universal import *  # noqa: F403, F405


@mcp_tool
async def ingest_file(path: str) -> IngestResult:
    """Ingest a single file (markdown, txt, source code, …)."""
    return await _ingest_pdf(path)


@mcp_tool
async def search(
    query: str,
    k: int = 5,
    doc_id: str | None = None,
    section: str | None = None,
    author: str | None = None,
    min_score: float | None = None,
    chunk_type: str | None = None,
) -> list[Chunk]:
    """Search the corpus. Returns ranked chunks."""
    import anyio

    return await anyio.to_thread.run_sync(
        lambda: _search(
            query,
            k=k,
            doc_id=doc_id,
            section=section,
            author=author,
            min_score=min_score,
            chunk_type=chunk_type,
        )
    )


@rest_route("GET", "/search")
async def search_rest(
    query: str,
    k: int = 5,
    doc_id: str | None = None,
    section: str | None = None,
    author: str | None = None,
    min_score: float | None = None,
    chunk_type: str | None = None,
) -> list[dict[str, object]]:
    """Semantic + hybrid search over the corpus."""
    import anyio

    chunks = await anyio.to_thread.run_sync(
        lambda: _search(
            query,
            k=k,
            doc_id=doc_id,
            section=section,
            author=author,
            min_score=min_score,
            chunk_type=chunk_type,
        )
    )
    return [c.model_dump(mode="json") for c in chunks]


@rest_route("POST", "/ingest")
async def ingest_rest(payload: dict[str, object]) -> dict[str, object]:
    """Ingest a file by path."""
    from fastapi import HTTPException

    path = str(payload.get("path", ""))
    if not path:
        raise HTTPException(status_code=400, detail="missing 'path' in payload")
    result = await _ingest_pdf(path)
    return result.model_dump(mode="json")


@cli_command(name="ingest")
def ingest_cli(path: str, json_output: bool = typer.Option(False, "--json")) -> None:
    """Ingest a single file."""
    result = asyncio.run(_ingest_pdf(path))
    output(result, json_output)


@cli_command(name="ingest-directory")
def ingest_directory_cli(
    directory: str = typer.Option(..., "--dir", help="Directory to walk"),
    pattern: str = typer.Option("*.md", "--pattern", help="Glob to match files against"),
    recursive: bool = typer.Option(True, "-r/--no-recursive", help="Recurse into subdirs"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Ingest every file in a directory matching a glob."""
    root = Path(directory)
    if not root.is_dir():
        typer.echo(f"Not a directory: {directory}", err=True)
        raise typer.Exit(1)
    matches: list[Path] = []
    matches = [p for p in root.rglob(pattern)] if recursive else [p for p in root.glob(pattern)]
    if not matches:
        typer.echo(f"No files matched {pattern!r} under {directory}", err=True)
        raise typer.Exit(1)
    success = failed = 0
    for path in matches:
        try:
            res = asyncio.run(_ingest_pdf(str(path)))
            success += 1
            typer.echo(f"OK    {path} ({res.chunk_count} chunks, {res.elapsed_seconds:.1f}s)")
        except Exception as exc:
            failed += 1
            typer.echo(f"FAIL  {path} ({exc})")
    if json_output:
        typer.echo(json.dumps({"success": success, "failed": failed}, indent=2))
    else:
        typer.echo(f"\nSuccess: {success}  Failed: {failed}")


@cli_command(name="search")
def search_cli(
    query: str,
    k: int = typer.Option(5, help="Number of results"),
    doc_id: str = typer.Option(None, help="Filter by document UUID"),
    section: str = typer.Option(None, help="Filter by section substring"),
    author: str = typer.Option(None, help="Filter by author substring"),
    min_score: float = typer.Option(None, help="Minimum cosine similarity"),
    chunk_type: str = typer.Option(None, help="Filter by chunk type: text|image"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Search the corpus."""
    chunks = _search(
        query,
        k=k,
        doc_id=doc_id,
        section=section,
        author=author,
        min_score=min_score,
        chunk_type=chunk_type,
    )
    if json_output:
        output(chunks, True)
        return
    if not chunks:
        typer.echo("No results found.")
        return
    for c in chunks:
        score_str = f"[{c.score:.4f}] " if c.score else ""
        typer.echo(f"{score_str}{c.section_path} (chunk {c.chunk_index})\n  {c.content[:200]}...\n")
