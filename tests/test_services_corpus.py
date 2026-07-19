"""Unit tests for services/corpus.py — list, get, delete, stats."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from lean.models.schemas import CorpusStats, DocumentSummary, ExtractionMethod

_VALID_KEY = "x" * 32


@pytest.fixture
def monkeypatch_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)


def test_list_documents(monkeypatch_env):
    from lean.services.corpus import list_documents

    fake_docs = [
        DocumentSummary(
            id="00000000-0000-0000-0000-000000000001",
            source_path="data/test.pdf",
            title="Book A",
            authors=["Author"],
            page_count=100,
            extraction_method=ExtractionMethod.MARKITDOWN,
            chunk_count=10,
            ingested_at=datetime(2026, 7, 12),
        )
    ]
    mock_repo = MagicMock()
    mock_repo.list_documents.return_value = fake_docs
    mock_conn = MagicMock()
    with (
        patch("lean.services.corpus.DocumentRepo", return_value=mock_repo),
        patch("lean.services.corpus.StoreConnection") as mock_store,
    ):
        mock_store.from_env.return_value = mock_conn
        result = list_documents()
    assert len(result) == 1
    assert result[0].title == "Book A"
    mock_conn.close.assert_called_once()


def test_get_chunk_found(monkeypatch_env):
    from lean.models.schemas import Chunk
    from lean.services.corpus import get_chunk

    fake_chunk = Chunk(
        id="00000000-0000-0000-0000-000000000001",
        document_id="00000000-0000-0000-0000-000000000002",
        chunk_index=0,
        section_path="Ch 1",
        token_count=50,
        content="content here",
    )
    mock_repo = MagicMock()
    mock_repo.get_chunk.return_value = fake_chunk
    mock_conn = MagicMock()
    with (
        patch("lean.services.corpus.ChunkRepo", return_value=mock_repo),
        patch("lean.services.corpus.StoreConnection") as mock_store,
    ):
        mock_store.from_env.return_value = mock_conn
        result = get_chunk("00000000-0000-0000-0000-000000000001")
    assert result is not None
    assert result.content == "content here"


def test_get_chunk_not_found(monkeypatch_env):
    from lean.services.corpus import get_chunk

    mock_repo = MagicMock()
    mock_repo.get_chunk.return_value = None
    mock_conn = MagicMock()
    with (
        patch("lean.services.corpus.ChunkRepo", return_value=mock_repo),
        patch("lean.services.corpus.StoreConnection") as mock_store,
    ):
        mock_store.from_env.return_value = mock_conn
        result = get_chunk("00000000-0000-0000-0000-000000000099")
    assert result is None


def test_get_document_markdown(monkeypatch_env):
    from lean.models.schemas import Chunk
    from lean.services.corpus import get_document_markdown

    fake_chunks = [
        Chunk(
            id=f"00000000-0000-0000-0000-00000000000{i}",
            document_id="00000000-0000-0000-0000-000000000002",
            chunk_index=i,
            section_path="Ch 1",
            token_count=50,
            content=f"Part {i}",
        )
        for i in range(3)
    ]
    mock_repo = MagicMock()
    mock_repo.get_chunks_by_document.return_value = fake_chunks
    mock_conn = MagicMock()
    with (
        patch("lean.services.corpus.ChunkRepo", return_value=mock_repo),
        patch("lean.services.corpus.StoreConnection") as mock_store,
    ):
        mock_store.from_env.return_value = mock_conn
        result = get_document_markdown("00000000-0000-0000-0000-000000000001")
    assert "Part 0" in result
    assert "Part 2" in result


def test_get_document_markdown_empty_raises(monkeypatch_env):
    from lean.services.corpus import get_document_markdown

    mock_repo = MagicMock()
    mock_repo.get_chunks_by_document.return_value = []
    mock_conn = MagicMock()
    with (
        patch("lean.services.corpus.ChunkRepo", return_value=mock_repo),
        patch("lean.services.corpus.StoreConnection") as mock_store,
    ):
        mock_store.from_env.return_value = mock_conn
        with pytest.raises(KeyError, match="not found"):
            get_document_markdown("00000000-0000-0000-0000-000000000001")


def test_delete_document(monkeypatch_env):
    from lean.services.corpus import delete_document

    mock_repo = MagicMock()
    mock_conn = MagicMock()
    with (
        patch("lean.services.corpus.DocumentRepo", return_value=mock_repo),
        patch("lean.services.corpus.StoreConnection") as mock_store,
    ):
        mock_store.from_env.return_value = mock_conn
        result = delete_document("00000000-0000-0000-0000-000000000001")
    assert result["deleted"] == "00000000-0000-0000-0000-000000000001"
    mock_repo.delete_document.assert_called_once()


def test_corpus_stats(monkeypatch_env):
    from lean.services.corpus import corpus_stats

    fake_stats = CorpusStats(
        document_count=10,
        chunk_count=2535,
        total_tokens=859776,
        extraction_method_breakdown={"unlimited_ocr": 4, "markitdown": 6},
        embedding_dim=1024,
        embedding_model="LiquidAI/LFM2.5-Embedding-350M",
    )
    mock_analytics = MagicMock()
    mock_analytics.corpus_stats.return_value = fake_stats
    mock_chunks = MagicMock()
    mock_chunks.find_duplicate_image_hashes.return_value = []
    mock_conn = MagicMock()
    with (
        patch("lean.services.corpus.AnalyticsRepo", return_value=mock_analytics),
        patch("lean.services.corpus.ChunkRepo", return_value=mock_chunks),
        patch("lean.services.corpus.StoreConnection") as mock_store,
    ):
        mock_store.from_env.return_value = mock_conn
        result = corpus_stats()
    assert result.document_count == 10
    assert result.chunk_count == 2535
    assert result.embedding_dim == 1024
    assert result.duplicate_image_hashes == []


def test_corpus_stats_includes_duplicate_image_hashes(monkeypatch_env):
    """duplicate_image_hashes is wired into corpus_stats (gives the store method a reader)."""
    from lean.services.corpus import corpus_stats

    fake_stats = CorpusStats(
        document_count=3,
        chunk_count=100,
        total_tokens=5000,
        extraction_method_breakdown={"markitdown": 3},
        embedding_dim=1024,
        embedding_model="test-model",
    )
    mock_analytics = MagicMock()
    mock_analytics.corpus_stats.return_value = fake_stats
    mock_chunks = MagicMock()
    mock_chunks.find_duplicate_image_hashes.return_value = [
        ("abc123", 4),
        ("def456", 2),
    ]
    mock_conn = MagicMock()
    with (
        patch("lean.services.corpus.AnalyticsRepo", return_value=mock_analytics),
        patch("lean.services.corpus.ChunkRepo", return_value=mock_chunks),
        patch("lean.services.corpus.StoreConnection") as mock_store,
    ):
        mock_store.from_env.return_value = mock_conn
        result = corpus_stats()
    assert result.duplicate_image_hashes == [
        {"image_hash": "abc123", "count": 4},
        {"image_hash": "def456", "count": 2},
    ]
    mock_chunks.find_duplicate_image_hashes.assert_called_once()
