"""PDF LSS domain — tool surface.

Universal tools (get_chunk, list_documents, corpus_stats, etc.) are
imported from ``lean.core.tools.universal``. This module keeps only
PDF-specific entry points: ingest_pdf, reingest, reingest-all, and
the full search with year/author/section filters.
"""

from __future__ import annotations

import asyncio
import json

import typer

from lean.core.adapters import cli_command, mcp_tool, output, rest_route
from lean.core.models import Chunk, IngestResult
from lean.core.services.ingestion import ingest_pdf as _ingest_pdf
from lean.core.services.ingestion import reingest as _reingest
from lean.core.services.search import search as _search
from lean.core.tools.universal import *  # noqa: F403, F405


@mcp_tool
async def ingest_pdf(path: str) -> IngestResult:
    """Ingest a PDF into the corpus (uses marker → OCR → markitdown chain)."""
    return await _ingest_pdf(path)


@mcp_tool
async def search(
    query: str,
    k: int = 5,
    doc_id: str | None = None,
    section: str | None = None,
    author: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    min_score: float | None = None,
    chunk_type: str | None = None,
) -> list[Chunk]:
    """Semantic + hybrid search over the corpus."""
    import anyio

    return await anyio.to_thread.run_sync(
        lambda: _search(
            query,
            k=k,
            doc_id=doc_id,
            section=section,
            author=author,
            year_min=year_min,
            year_max=year_max,
            min_score=min_score,
            chunk_type=chunk_type,
        )
    )


@mcp_tool
async def reingest(document_id: str) -> IngestResult:
    """Re-ingest a document with current settings."""
    return await _reingest(document_id)


@rest_route("GET", "/search")
async def search_rest(
    query: str,
    k: int = 5,
    doc_id: str | None = None,
    section: str | None = None,
    author: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
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
            year_min=year_min,
            year_max=year_max,
            min_score=min_score,
            chunk_type=chunk_type,
        )
    )
    return [c.model_dump(mode="json") for c in chunks]


@rest_route("POST", "/ingest")
async def ingest_rest(payload: dict[str, object]) -> dict[str, object]:
    """Ingest a PDF (path in ``payload['path']``)."""
    from fastapi import HTTPException

    path = str(payload.get("path", ""))
    if not path:
        raise HTTPException(status_code=400, detail="missing 'path' in payload")
    result = await _ingest_pdf(path)
    return result.model_dump(mode="json")


@cli_command
def ingest(path: str, json_output: bool = typer.Option(False, "--json")) -> None:
    """Ingest a PDF into the corpus."""
    result = asyncio.run(_ingest_pdf(path))
    output(result, json_output)


@cli_command(name="search")
def search_cli(
    query: str,
    k: int = typer.Option(5, help="Number of results"),
    doc_id: str = typer.Option(None, help="Filter by document UUID"),
    section: str = typer.Option(None, help="Filter by section substring"),
    author: str = typer.Option(None, help="Filter by author substring"),
    year_min: int = typer.Option(None, help="Filter by min publication year"),
    year_max: int = typer.Option(None, help="Filter by max publication year"),
    min_score: float = typer.Option(None, help="Minimum cosine similarity"),
    chunk_type: str = typer.Option(None, help="Filter by chunk type: text|image"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Semantic + hybrid search over the corpus."""
    chunks = _search(
        query,
        k=k,
        doc_id=doc_id,
        section=section,
        author=author,
        year_min=year_min,
        year_max=year_max,
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


@cli_command(name="reingest")
def reingest_cli(doc_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
    """Re-ingest a document (re-extract with current settings)."""
    result = asyncio.run(_reingest(doc_id))
    output(result, json_output)


@cli_command(name="reingest-all")
def reingest_all(
    force: bool = typer.Option(False, "--force"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Reingest all documents (smallest first, skips OCR'd unless --force)."""
    from lean.core.models.schemas import ExtractionMethod
    from lean.core.services.corpus import list_documents as _list_documents

    docs = sorted(_list_documents(), key=lambda d: d.chunk_count)
    results: list[dict[str, object]] = []
    success = failed = skipped = 0
    for doc in docs:
        if not force and doc.extraction_method == ExtractionMethod.UNLIMITED_OCR:
            skipped += 1
            results.append({"id": doc.id, "status": "skipped"})
            typer.echo(f"SKIP  {doc.id}  (already OCR'd, {doc.chunk_count} chunks)")
            continue
        typer.echo(f"START {doc.id}  ({doc.page_count or '?'} pages, was {doc.extraction_method})")
        try:
            res = asyncio.run(_reingest(doc.id))
            success += 1
            results.append(
                {
                    "id": doc.id,
                    "status": "ok",
                    "chunks": res.chunk_count,
                    "seconds": res.elapsed_seconds,
                }
            )
            typer.echo(f"DONE  {doc.id}  ({res.chunk_count} chunks, {res.elapsed_seconds:.1f}s)")
        except Exception as exc:
            failed += 1
            results.append({"id": doc.id, "status": "failed", "error": str(exc)})
            typer.echo(f"FAIL  {doc.id}  ({exc})")
    if json_output:
        typer.echo(
            json.dumps(
                {"success": success, "failed": failed, "skipped": skipped, "details": results},
                indent=2,
                default=str,
            )
        )
    else:
        typer.echo(f"\nSuccess: {success}  Failed: {failed}  Skipped: {skipped}")
