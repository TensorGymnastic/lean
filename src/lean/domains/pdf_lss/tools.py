"""PDF LSS domain — tool surface.

Auto-discovered by the YAML loader: every public decorated callable is
registered as either an MCP tool, REST route, or CLI command.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import typer

from lean.core.adapters import (
    cli_command,
    mcp_tool,
    output,
    rest_route,
)
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
from lean.core.services.ingestion import reingest as _reingest
from lean.core.services.search import search as _search


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
async def get_chunk(chunk_id: str) -> Chunk | None:
    """Retrieve a single chunk by ID."""
    import anyio

    return await anyio.to_thread.run_sync(lambda: _get_chunk(chunk_id))


@mcp_tool
async def list_documents() -> list[DocumentSummary]:
    """List all documents in the corpus, newest first."""
    import anyio

    return await anyio.to_thread.run_sync(_list_documents)


@mcp_tool
async def get_document_markdown(document_id: str) -> str:
    """Get the extracted markdown for a document."""
    import anyio

    return await anyio.to_thread.run_sync(lambda: _get_markdown(document_id))


@mcp_tool
async def delete_document(document_id: str) -> dict[str, str]:
    """Delete a document and all its chunks."""
    import anyio

    return await anyio.to_thread.run_sync(lambda: _delete_document(document_id))


@mcp_tool
async def reingest(document_id: str) -> IngestResult:
    """Re-ingest a document with current settings."""
    return await _reingest(document_id)


@mcp_tool
async def corpus_stats() -> CorpusStats:
    """Show corpus statistics."""
    import anyio

    return await anyio.to_thread.run_sync(_corpus_stats)


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


@rest_route("GET", "/documents")
async def list_documents_rest() -> list[dict[str, object]]:
    """List all documents in the corpus, newest first."""
    docs = _list_documents()
    return [d.model_dump(mode="json") for d in docs]


@rest_route("GET", "/chunks/{chunk_id}")
async def get_chunk_rest(chunk_id: str) -> dict[str, object]:
    """Retrieve a single chunk by ID."""
    from fastapi import HTTPException

    chunk = _get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    return chunk.model_dump(mode="json")


@rest_route("GET", "/documents/{document_id}/markdown")
async def get_document_markdown_rest(document_id: str) -> dict[str, str]:
    """Get the extracted markdown for a document."""
    from fastapi import HTTPException

    try:
        markdown = _get_markdown(document_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="document not found") from None
    return {"document_id": document_id, "markdown": markdown}


@rest_route("DELETE", "/documents/{document_id}")
async def delete_document_rest(document_id: str) -> dict[str, str]:
    """Delete a document and all its chunks."""
    return _delete_document(document_id)


@rest_route("GET", "/stats")
async def corpus_stats_rest() -> dict[str, object]:
    """Show corpus statistics."""
    stats = _corpus_stats()
    return stats.model_dump(mode="json")


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


@cli_command(name="list-documents")
def list_documents_cli(json_output: bool = typer.Option(False, "--json")) -> None:
    """List all documents in the corpus."""
    docs = _list_documents()
    output(docs, json_output)


@cli_command(name="corpus-stats")
def corpus_stats_cli(json_output: bool = typer.Option(False, "--json")) -> None:
    """Show corpus statistics."""
    stats = _corpus_stats()
    output(stats, json_output)


@cli_command(name="get-chunk")
def get_chunk_cli(
    chunk_id: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Retrieve a single chunk by ID."""
    chunk = _get_chunk(chunk_id)
    if chunk is None:
        typer.echo("Chunk not found.", err=True)
        raise typer.Exit(1)
    output(chunk, json_output)


@cli_command(name="get-markdown")
def get_markdown_cli(
    doc_id: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Get the extracted markdown for a document."""
    md = _get_markdown(doc_id)
    if json_output:
        typer.echo(json.dumps({"doc_id": doc_id, "markdown": md}, indent=2))
    else:
        typer.echo(md)


@cli_command
def delete(doc_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
    """Delete a document and all its chunks."""
    result = _delete_document(doc_id)
    output(result, json_output)


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
