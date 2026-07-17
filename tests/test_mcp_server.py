"""Unit tests for MCP server tools, resources, and prompts."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

_VALID_KEY = "x" * 32


@pytest.fixture
def monkeypatch_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)


@pytest.mark.asyncio
async def test_mcp_tool_search(monkeypatch_env):
    from lean.mcp_server.tools import search
    from lean.models.schemas import Chunk

    fake_chunks = [
        Chunk(
            id="00000000-0000-0000-0000-000000000001",
            document_id="00000000-0000-0000-0000-000000000002",
            chunk_index=0,
            section_path="Ch 1",
            heading_text="DMAIC",
            token_count=100,
            content="DMAIC content",
            score=0.9,
        )
    ]
    with patch("lean.mcp_server.tools._search", return_value=fake_chunks):
        result = await search("DMAIC", k=5)
    assert len(result) == 1
    assert result[0].content == "DMAIC content"


@pytest.mark.asyncio
async def test_mcp_tool_get_chunk(monkeypatch_env):
    from lean.mcp_server.tools import get_chunk
    from lean.models.schemas import Chunk

    fake_chunk = Chunk(
        id="00000000-0000-0000-0000-000000000001",
        document_id="00000000-0000-0000-0000-000000000002",
        chunk_index=0,
        section_path="Ch 1",
        token_count=50,
        content="content",
    )
    with patch("lean.mcp_server.tools._get_chunk", return_value=fake_chunk):
        result = await get_chunk("00000000-0000-0000-0000-000000000001")
    assert result is not None
    assert result.content == "content"


@pytest.mark.asyncio
async def test_mcp_tool_get_chunk_not_found(monkeypatch_env):
    from lean.mcp_server.tools import get_chunk

    with patch("lean.mcp_server.tools._get_chunk", return_value=None):
        result = await get_chunk("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_mcp_tool_list_documents(monkeypatch_env):
    from datetime import datetime

    from lean.mcp_server.tools import list_documents
    from lean.models.schemas import DocumentSummary, ExtractionMethod

    fake_docs = [
        DocumentSummary(
            id="00000000-0000-0000-0000-000000000001",
            source_path="data/test.pdf",
            title="Test",
            authors=[],
            extraction_method=ExtractionMethod.MARKITDOWN,
            chunk_count=5,
            ingested_at=datetime(2026, 7, 12),
        )
    ]
    with patch("lean.mcp_server.tools._list", return_value=fake_docs):
        result = await list_documents()
    assert len(result) == 1
    assert result[0].title == "Test"


@pytest.mark.asyncio
async def test_mcp_tool_get_document_markdown(monkeypatch_env):
    from lean.mcp_server.tools import get_document_markdown

    with patch("lean.mcp_server.tools._get_markdown", return_value="# Markdown"):
        result = await get_document_markdown("doc-id")
    assert result == "# Markdown"


@pytest.mark.asyncio
async def test_mcp_tool_delete_document(monkeypatch_env):
    from lean.mcp_server.tools import delete_document

    with patch("lean.mcp_server.tools._delete", return_value={"deleted": "doc-id"}):
        result = await delete_document("doc-id")
    assert result["deleted"] == "doc-id"


@pytest.mark.asyncio
async def test_mcp_tool_reingest(monkeypatch_env):
    from lean.mcp_server.tools import reingest
    from lean.models.schemas import ExtractionMethod, IngestResult

    fake_result = IngestResult(
        document_id="00000000-0000-0000-0000-000000000001",
        source_sha256="abc",
        page_count=10,
        extraction_method=ExtractionMethod.MARKITDOWN,
        chunk_count=5,
        elapsed_seconds=2.0,
    )
    with patch("lean.mcp_server.tools._reingest", new_callable=AsyncMock, return_value=fake_result):
        result = await reingest("doc-id")
    assert result.chunk_count == 5


@pytest.mark.asyncio
async def test_mcp_tool_corpus_stats(monkeypatch_env):
    from lean.mcp_server.tools import corpus_stats
    from lean.models.schemas import CorpusStats

    fake_stats = CorpusStats(
        document_count=5,
        chunk_count=100,
        total_tokens=50000,
        extraction_method_breakdown={"markitdown": 5},
        embedding_dim=1024,
        embedding_model="test-model",
    )
    with patch("lean.mcp_server.tools._corpus_stats", return_value=fake_stats):
        result = await corpus_stats()
    assert result.document_count == 5


@pytest.mark.asyncio
async def test_mcp_tool_ingest_pdf(monkeypatch_env):
    from lean.mcp_server.tools import ingest_pdf
    from lean.models.schemas import ExtractionMethod, IngestResult

    fake_result = IngestResult(
        document_id="00000000-0000-0000-0000-000000000001",
        source_sha256="abc",
        page_count=10,
        extraction_method=ExtractionMethod.MARKITDOWN,
        chunk_count=5,
        elapsed_seconds=2.0,
    )
    with patch("lean.mcp_server.tools._ingest", new_callable=AsyncMock, return_value=fake_result):
        result = await ingest_pdf("data/test.pdf")
    assert result.chunk_count == 5
