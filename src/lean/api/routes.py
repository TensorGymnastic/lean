"""FastAPI mirror of MCP tools for non-agent HTTP clients.

Thin delegates to the service layer — same handlers as the MCP tools,
REST shape. Bearer token auth on all non-health endpoints.
Serves on port 8766 when started via ``make api-serve``.
"""

from __future__ import annotations

import hmac
from importlib.metadata import version as _pkg_version
from typing import Annotated, Any

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from lean.config.settings import get_settings
from lean.services.corpus import corpus_stats as _corpus_stats
from lean.services.corpus import delete_document as _delete
from lean.services.corpus import get_chunk as _get_chunk
from lean.services.corpus import get_document_markdown as _get_markdown
from lean.services.corpus import list_documents as _list
from lean.services.ingestion import ingest_pdf as _ingest
from lean.services.search import search as _search

app = FastAPI(
    title="lean",
    version=_pkg_version("lean"),
    description="Lean Six Sigma MCP corpus API",
)
_security = HTTPBearer(auto_error=False)

TokenCreds = Annotated[HTTPAuthorizationCredentials | None, Security(_security)]


@app.exception_handler(ValueError)
async def _value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    """Map domain validation errors (empty query, PDF too large, bad UUID) to HTTP 400."""
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


@app.exception_handler(PermissionError)
async def _permission_error_handler(_request: Request, exc: PermissionError) -> JSONResponse:
    """Map path-traversal / corpus-root violations to HTTP 403."""
    return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def _not_found_handler(_request: Request, exc: FileNotFoundError) -> JSONResponse:
    """Map missing file errors to HTTP 404."""
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})


async def _verify_token(creds: TokenCreds) -> None:
    settings = get_settings()
    expected = settings.lean_mcp_api_key
    if not creds or not hmac.compare_digest(creds.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer token",
        )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/documents", dependencies=[Depends(_verify_token)])
async def list_documents() -> list[dict[str, Any]]:
    docs = await anyio.to_thread.run_sync(_list)
    return [d.model_dump(mode="json") for d in docs]


@app.get("/search", dependencies=[Depends(_verify_token)])
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
) -> list[dict[str, Any]]:
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


@app.post("/ingest", dependencies=[Depends(_verify_token)])
async def ingest(path: str) -> dict[str, Any]:
    result = await _ingest(path)
    return result.model_dump(mode="json")


@app.get("/stats", dependencies=[Depends(_verify_token)])
async def stats() -> dict[str, Any]:
    result = await anyio.to_thread.run_sync(_corpus_stats)
    return result.model_dump(mode="json")


@app.get("/chunks/{chunk_id}", dependencies=[Depends(_verify_token)])
async def get_chunk(chunk_id: str) -> dict[str, Any]:
    chunk = await anyio.to_thread.run_sync(lambda: _get_chunk(chunk_id))
    if chunk is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="chunk not found")
    return chunk.model_dump(mode="json")


@app.get("/documents/{document_id}/markdown", dependencies=[Depends(_verify_token)])
async def get_document_markdown(document_id: str) -> dict[str, str]:
    try:
        markdown = await anyio.to_thread.run_sync(lambda: _get_markdown(document_id))
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="document not found"
        ) from None
    return {"document_id": document_id, "markdown": markdown}


@app.delete("/documents/{document_id}", dependencies=[Depends(_verify_token)])
async def delete_document(document_id: str) -> dict[str, str]:
    return await anyio.to_thread.run_sync(lambda: _delete(document_id))
