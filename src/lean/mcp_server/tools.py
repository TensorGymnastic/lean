"""MCP tools — thin delegates to the service layer."""

from __future__ import annotations

import anyio
from fastmcp import FastMCP

from lean.models.schemas import Chunk, CorpusStats, DocumentSummary, IngestResult
from lean.services.corpus import corpus_stats as _corpus_stats
from lean.services.corpus import delete_document as _delete
from lean.services.corpus import get_chunk as _get_chunk
from lean.services.corpus import get_document_markdown as _get_markdown
from lean.services.corpus import list_documents as _list
from lean.services.ingestion import ingest_pdf as _ingest
from lean.services.ingestion import reingest as _reingest
from lean.services.search import search as _search

mcp = FastMCP("lean")


@mcp.tool
async def ingest_pdf(path: str) -> IngestResult:
    """Ingest a PDF into the lean corpus."""
    return await _ingest(path)


@mcp.tool
async def search(
    query: str,
    k: int = 5,
    doc_id: str | None = None,
    section: str | None = None,
    author: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    min_score: float | None = None,
) -> list[Chunk]:
    """Semantic + hybrid search over the Lean Six Sigma corpus."""
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
        )
    )


@mcp.tool
async def get_chunk(chunk_id: str) -> Chunk | None:
    """Retrieve a single chunk by ID."""
    return await anyio.to_thread.run_sync(lambda: _get_chunk(chunk_id))


@mcp.tool
async def list_documents() -> list[DocumentSummary]:
    """List all documents in the corpus."""
    return await anyio.to_thread.run_sync(_list)


@mcp.tool
async def get_document_markdown(document_id: str) -> str:
    """Get the extracted markdown for a document."""
    return await anyio.to_thread.run_sync(lambda: _get_markdown(document_id))


@mcp.tool
async def delete_document(document_id: str) -> dict[str, str]:
    """Delete a document and all its chunks."""
    return await anyio.to_thread.run_sync(lambda: _delete(document_id))


@mcp.tool
async def reingest(document_id: str) -> IngestResult:
    """Re-ingest a document."""
    return await _reingest(document_id)


@mcp.tool
async def corpus_stats() -> CorpusStats:
    """Show corpus statistics."""
    return await anyio.to_thread.run_sync(_corpus_stats)
