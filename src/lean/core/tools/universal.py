"""Universal MCP tools, REST routes, and CLI commands shared by every domain.

Importing this module into a domain's ``tools.py`` makes these functions
discoverable by ``register_mcp_tools`` / ``register_api_tools`` /
``register_cli_tools``, which scan for the ``__lean_tool_kind__`` attribute
set by the ``@mcp_tool`` / ``@rest_route`` / ``@cli_command`` decorators.

Domain-specific tools (``ingest_pdf``, ``ingest_file``, ``ingest_url``,
``reingest``, ``search`` with domain-specific filters) stay in the domain's
``tools.py``.
"""

from __future__ import annotations

import json

import anyio
import typer

from lean.core.adapters import cli_command, mcp_tool, output, rest_route
from lean.core.models import Chunk, CorpusStats, DocumentSummary
from lean.core.services.corpus import corpus_stats as _corpus_stats
from lean.core.services.corpus import delete_document as _delete_document
from lean.core.services.corpus import get_chunk as _get_chunk
from lean.core.services.corpus import get_document_markdown as _get_markdown
from lean.core.services.corpus import list_documents as _list_documents

# ── MCP tools ──────────────────────────────────────────────────────────


@mcp_tool
async def get_chunk(chunk_id: str) -> Chunk | None:
    """Retrieve a single chunk by ID."""
    return await anyio.to_thread.run_sync(lambda: _get_chunk(chunk_id))


@mcp_tool
async def list_documents() -> list[DocumentSummary]:
    """List all documents in the corpus, newest first."""
    return await anyio.to_thread.run_sync(_list_documents)


@mcp_tool
async def corpus_stats() -> CorpusStats:
    """Show corpus statistics."""
    return await anyio.to_thread.run_sync(_corpus_stats)


@mcp_tool
async def get_document_markdown(document_id: str) -> str:
    """Get the extracted markdown for a document."""
    return await anyio.to_thread.run_sync(lambda: _get_markdown(document_id))


@mcp_tool
async def delete_document(document_id: str) -> dict[str, str]:
    """Delete a document and all its chunks."""
    return await anyio.to_thread.run_sync(lambda: _delete_document(document_id))


# ── REST routes ────────────────────────────────────────────────────────


@rest_route("GET", "/documents")
async def list_documents_rest() -> list[dict[str, object]]:
    """List all documents in the corpus, newest first."""
    docs = _list_documents()
    return [d.model_dump(mode="json") for d in docs]


@rest_route("GET", "/stats")
async def corpus_stats_rest() -> dict[str, object]:
    """Show corpus statistics."""
    stats = _corpus_stats()
    return stats.model_dump(mode="json")


@rest_route("GET", "/chunks/{chunk_id}")
async def get_chunk_rest(chunk_id: str) -> dict[str, object]:
    """Retrieve a single chunk by ID."""
    from fastapi import HTTPException

    chunk = _get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    return chunk.model_dump(mode="json")


@rest_route("DELETE", "/documents/{document_id}")
async def delete_document_rest(document_id: str) -> dict[str, str]:
    """Delete a document and all its chunks."""
    return _delete_document(document_id)


@rest_route("GET", "/documents/{document_id}/markdown")
async def get_document_markdown_rest(document_id: str) -> dict[str, str]:
    """Get the extracted markdown for a document."""
    from fastapi import HTTPException

    try:
        markdown = _get_markdown(document_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="document not found") from None
    return {"document_id": document_id, "markdown": markdown}


# ── CLI commands ───────────────────────────────────────────────────────


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


__all__ = [
    "get_chunk",
    "list_documents",
    "corpus_stats",
    "get_document_markdown",
    "delete_document",
    "list_documents_rest",
    "corpus_stats_rest",
    "get_chunk_rest",
    "delete_document_rest",
    "get_document_markdown_rest",
    "list_documents_cli",
    "corpus_stats_cli",
    "get_chunk_cli",
    "get_markdown_cli",
    "delete",
]
