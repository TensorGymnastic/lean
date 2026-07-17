"""Unit tests for MCP server resources."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

_VALID_KEY = "x" * 32


@pytest.fixture
def monkeypatch_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)


@pytest.mark.asyncio
async def test_documents_resource(monkeypatch_env):
    from datetime import datetime

    from lean.mcp_server.resources import documents_resource
    from lean.models.schemas import DocumentSummary, ExtractionMethod

    fake_docs = [
        DocumentSummary(
            id="00000000-0000-0000-0000-000000000001",
            source_path="data/test.pdf",
            title="Book",
            authors=["Author"],
            page_count=100,
            extraction_method=ExtractionMethod.MARKITDOWN,
            chunk_count=10,
            ingested_at=datetime(2026, 7, 12),
        )
    ]
    with patch("lean.services.corpus.list_documents", return_value=fake_docs):
        result = await documents_resource()
    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["title"] == "Book"


@pytest.mark.asyncio
async def test_markdown_resource(monkeypatch_env):
    from lean.mcp_server.resources import markdown_resource

    with patch("lean.services.corpus.get_document_markdown", return_value="# Title"):
        result = await markdown_resource("doc-id")
    assert result == "# Title"


@pytest.mark.asyncio
async def test_chunks_resource(monkeypatch_env):
    from lean.mcp_server.resources import chunks_resource
    from lean.models.schemas import Chunk

    fake_chunks = [
        Chunk(
            id="00000000-0000-0000-0000-000000000001",
            document_id="00000000-0000-0000-0000-000000000002",
            chunk_index=0,
            section_path="Ch 1",
            heading_text="Heading",
            token_count=50,
            content="content",
        )
    ]
    with patch("lean.services.corpus.list_chunks_by_document", return_value=fake_chunks):
        result = await chunks_resource("doc-id")
    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["heading_text"] == "Heading"


@pytest.mark.asyncio
async def test_stats_resource(monkeypatch_env):
    from lean.mcp_server.resources import stats_resource
    from lean.models.schemas import CorpusStats

    fake_stats = CorpusStats(
        document_count=5,
        chunk_count=100,
        total_tokens=50000,
        extraction_method_breakdown={"markitdown": 5},
        embedding_dim=1024,
        embedding_model="test",
    )
    with patch("lean.services.corpus.corpus_stats", return_value=fake_stats):
        result = await stats_resource()
    data = json.loads(result)
    assert data["document_count"] == 5
