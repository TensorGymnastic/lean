"""Code domain — tool surface.

Auto-discoverable MCP tools, REST routes, and CLI commands for the
code/markdown-file corpus. Reuses the universal services from
lean (corpus_stats, list_documents, get_chunk, etc.) and adds
domain-specific entry points (ingest_directory, ingest_git_repo).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
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


@mcp_tool
async def get_chunk(chunk_id: str) -> Chunk | None:
    """Retrieve a single chunk by ID."""
    import anyio

    return await anyio.to_thread.run_sync(lambda: _get_chunk(chunk_id))


@mcp_tool
async def list_documents() -> list[DocumentSummary]:
    """List all documents in the corpus."""
    import anyio

    return await anyio.to_thread.run_sync(_list_documents)


@mcp_tool
async def corpus_stats() -> CorpusStats:
    """Show corpus statistics."""
    import anyio

    return await anyio.to_thread.run_sync(_corpus_stats)


@mcp_tool
async def get_document_markdown(document_id: str) -> str:
    """Get the full markdown for a document."""
    import anyio

    return await anyio.to_thread.run_sync(lambda: _get_markdown(document_id))


@mcp_tool
async def delete_document(document_id: str) -> dict[str, str]:
    """Delete a document and all its chunks."""
    import anyio

    return await anyio.to_thread.run_sync(lambda: _delete_document(document_id))


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


@rest_route("GET", "/documents")
async def list_documents_rest() -> list[dict[str, object]]:
    docs = _list_documents()
    return [d.model_dump(mode="json") for d in docs]


@rest_route("GET", "/chunks/{chunk_id}")
async def get_chunk_rest(chunk_id: str) -> dict[str, object]:
    chunk = _get_chunk(chunk_id)
    if chunk is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="chunk not found")
    return chunk.model_dump(mode="json")


@rest_route("GET", "/stats")
async def corpus_stats_rest() -> dict[str, object]:
    stats = _corpus_stats()
    return stats.model_dump(mode="json")


@rest_route("DELETE", "/documents/{document_id}")
async def delete_document_rest(document_id: str) -> dict[str, str]:
    return _delete_document(document_id)


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
    """Get the full markdown for a document."""
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
