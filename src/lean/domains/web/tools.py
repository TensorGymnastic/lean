"""Web domain — tool surface.

Universal tools are imported from ``lean.core.tools.universal``. This
module keeps only web-specific entry points: ingest_url, ingest_list,
and a simplified search.
"""

from __future__ import annotations

import asyncio
import json

import typer

from lean.core.adapters import cli_command, mcp_tool, output, rest_route
from lean.core.models import Chunk, IngestResult
from lean.core.services.ingestion import ingest_pdf as _ingest_pdf
from lean.core.services.search import search as _search
from lean.core.tools.universal import *  # noqa: F403, F405


@mcp_tool
async def ingest_url(url: str) -> IngestResult:
    """Fetch a URL and ingest its content as markdown."""
    return await _ingest_pdf(url)


@mcp_tool
async def search(
    query: str,
    k: int = 5,
    doc_id: str | None = None,
    min_score: float | None = None,
) -> list[Chunk]:
    """Search the corpus of web pages."""
    import anyio

    return await anyio.to_thread.run_sync(
        lambda: _search(query, k=k, doc_id=doc_id, min_score=min_score)
    )


@rest_route("POST", "/ingest")
async def ingest_url_rest(payload: dict[str, object]) -> dict[str, object]:
    """Fetch a URL and ingest its content."""
    from fastapi import HTTPException

    url = str(payload.get("url", ""))
    if not url:
        raise HTTPException(status_code=400, detail="missing 'url' in payload")
    result = await _ingest_pdf(url)
    return result.model_dump(mode="json")


@rest_route("GET", "/search")
async def search_rest(
    query: str,
    k: int = 5,
    doc_id: str | None = None,
    min_score: float | None = None,
) -> list[dict[str, object]]:
    chunks = _search(query, k=k, doc_id=doc_id, min_score=min_score)
    return [c.model_dump(mode="json") for c in chunks]


@cli_command(name="ingest")
def ingest_url_cli(url: str, json_output: bool = typer.Option(False, "--json")) -> None:
    """Fetch a URL and ingest its content."""
    result = asyncio.run(_ingest_pdf(url))
    output(result, json_output)


@cli_command(name="ingest-list")
def ingest_list_cli(
    file: str = typer.Option(..., "--file", help="Path to file with one URL per line"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Ingest every URL listed in a file (one per line)."""
    from pathlib import Path

    urls = [line.strip() for line in Path(file).read_text().splitlines() if line.strip()]
    success = failed = 0
    for url in urls:
        try:
            res = asyncio.run(_ingest_pdf(url))
            success += 1
            typer.echo(f"OK    {url} ({res.chunk_count} chunks)")
        except Exception as exc:
            failed += 1
            typer.echo(f"FAIL  {url} ({exc})")
    if json_output:
        typer.echo(json.dumps({"success": success, "failed": failed}, indent=2))
    else:
        typer.echo(f"\nSuccess: {success}  Failed: {failed}")


@cli_command(name="search")
def search_cli(
    query: str,
    k: int = typer.Option(5, help="Number of results"),
    doc_id: str = typer.Option(None, help="Filter by document UUID"),
    min_score: float = typer.Option(None, help="Minimum cosine similarity"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Search the corpus."""
    chunks = _search(query, k=k, doc_id=doc_id, min_score=min_score)
    if json_output:
        output(chunks, True)
        return
    if not chunks:
        typer.echo("No results found.")
        return
    for c in chunks:
        score_str = f"[{c.score:.4f}] " if c.score else ""
        typer.echo(f"{score_str}{c.section_path} (chunk {c.chunk_index})\n  {c.content[:200]}...\n")
