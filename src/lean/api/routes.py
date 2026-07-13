"""FastAPI mirror of MCP tools for non-agent HTTP clients.

Same handlers as the MCP tools, REST shape. Bearer token auth on all
non-health endpoints. Serves on port 8766 when started via `make api-serve`.
"""

from __future__ import annotations

import hmac
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from lean.settings import Settings

app = FastAPI(title="lean", version="0.1.0", description="Lean Six Sigma MCP corpus API")
_security = HTTPBearer(auto_error=False)

TokenCreds = Annotated[HTTPAuthorizationCredentials | None, Security(_security)]


async def _verify_token(creds: TokenCreds) -> None:
    settings = Settings()
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
    from lean.mcp_server.tools import list_documents

    docs = await list_documents()
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
) -> list[dict[str, Any]]:
    from lean.mcp_server.tools import search as _search

    chunks = await _search(
        query,
        k=k,
        doc_id=doc_id,
        section=section,
        author=author,
        year_min=year_min,
        year_max=year_max,
        min_score=min_score,
    )
    return [c.model_dump(mode="json") for c in chunks]


@app.post("/ingest", dependencies=[Depends(_verify_token)])
async def ingest(path: str) -> dict[str, Any]:
    from lean.mcp_server.tools import ingest_pdf

    result = await ingest_pdf(path)
    return result.model_dump(mode="json")


@app.get("/stats", dependencies=[Depends(_verify_token)])
async def stats() -> dict[str, Any]:
    from lean.mcp_server.tools import corpus_stats

    result = await corpus_stats()
    return result.model_dump(mode="json")
