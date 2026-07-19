"""Web domain — tool surface.

Search and corpus tools over URL-scraped markdown pages.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import typer

from lean.core.adapters import cli_command, mcp_tool, output, rest_route
from lean.core.models import Chunk, CorpusStats, DocumentSummary, IngestResult
from lean.core.services.corpus import (
    corpus_stats as _corpus_stats,
)
from lean.core.services.corpus import (
    delete_document as _delete_document,
)
from lean.core.services.corpus import (
    get_chunk as _get_chunk,
)
from lean.core.services.corpus import (
    get_document_markdown as _get_markdown,
)
from lean.core.services.corpus import (
    list_documents as _list_documents,
)
from lean.core.services.ingestion import ingest_pdf as _ingest_pdf
from lean.core.services.search import search as _search


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


@mcp_tool
async def get_chunk(chunk_id: str) -> Chunk | None:
    """Retrieve a single chunk by ID."""
    import anyio

    return await anyio.to_thread.run_sync(lambda: _get_chunk(chunk_id))


@mcp_tool
async def list_documents() -> list[DocumentSummary]:
    """List all ingested URLs."""
    import anyio

    return await anyio.to_thread.run_sync(_list_documents)


@mcp_tool
async def corpus_stats() -> CorpusStats:
    """Show corpus statistics."""
    import anyio

    return await anyio.to_thread.run_sync(_corpus_stats)


@mcp_tool
async def get_document_markdown(document_id: str) -> str:
    """Get the markdown for a previously-ingested page."""
    import anyio

    return await anyio.to_thread.run_sync(lambda: _get_markdown(document_id))


@mcp_tool
async def delete_document(document_id: str) -> dict[str, str]:
    """Delete an ingested URL and all its chunks."""
    import anyio

    return await anyio.to_thread.run_sync(lambda: _delete_document(document_id))


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


@rest_route("GET", "/documents")
async def list_documents_rest() -> list[dict[str, object]]:
    docs = _list_documents()
    return [d.model_dump(mode="json") for d in docs]


@rest_route("GET", "/stats")
async def corpus_stats_rest() -> dict[str, object]:
    stats = _corpus_stats()
    return stats.model_dump(mode="json")


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


@cli_command(name="list-documents")
def list_documents_cli(json_output: bool = typer.Option(False, "--json")) -> None:
    """List all ingested URLs."""
    docs = _list_documents()
    output(docs, json_output)


@cli_command(name="corpus-stats")
def corpus_stats_cli(json_output: bool = typer.Option(False, "--json")) -> None:
    """Show corpus statistics."""
    stats = _corpus_stats()
    output(stats, json_output)


@cli_command
def delete(doc_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
    """Delete an ingested URL and all its chunks."""
    result = _delete_document(doc_id)
    output(result, json_output)


def register_mcp(mcp: Any) -> None:
    from lean.core.adapters import register_mcp_tools

    register_mcp_tools(mcp, sys.modules[__name__])


def register_api(app: Any) -> None:
    from lean.core.adapters import register_api_tools

    register_api_tools(app, sys.modules[__name__])


def register_cli(app: typer.Typer) -> None:
    from lean.core.adapters import register_cli_tools

    register_cli_tools(app, sys.modules[__name__])


__all__ = ["register_mcp", "register_api", "register_cli"]
