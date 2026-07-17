"""Unit tests for the ingestion service pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

_VALID_KEY = "x" * 32


@pytest.fixture
def monkeypatch_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.config.settings import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "corpus_root", str(tmp_path))
    yield settings
    get_settings.cache_clear()


@pytest.fixture
def fake_pdf(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")
    return pdf_path


def test_ingest_rejects_path_outside_corpus_root(monkeypatch_settings, tmp_path):
    import asyncio

    from lean.services.ingestion import ingest_pdf

    outside = tmp_path.parent / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(PermissionError, match="outside corpus root"):
        asyncio.run(ingest_pdf(str(outside)))


def test_ingest_rejects_nonexistent_file(monkeypatch_settings, tmp_path):
    import asyncio

    from lean.services.ingestion import ingest_pdf

    with pytest.raises(FileNotFoundError, match="PDF not found"):
        asyncio.run(ingest_pdf(str(tmp_path / "nonexistent.pdf")))


def test_ingest_pipeline_success(monkeypatch_settings, fake_pdf):
    import asyncio

    from lean.chunker.markdown_ast import Section
    from lean.chunker.recursive import ChunkResult
    from lean.models.schemas import ExtractionMethod
    from lean.services.ingestion import ingest_pdf

    fake_sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="DMAIC content")]
    fake_chunks = [
        ChunkResult(
            section_path="Ch 1",
            heading_text="Ch 1",
            chunk_index=0,
            token_count=10,
            content="DMAIC content",
        )
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1] * 1024]

    mock_conn = MagicMock()
    mock_doc_repo = MagicMock()
    mock_doc_repo.upsert_document.return_value = __import__("uuid").uuid4()
    mock_chunk_repo = MagicMock()

    from lean.extraction.metadata import PdfMetadata

    fake_meta = PdfMetadata(title="Test Book", authors=["Author"], year=2024)

    with (
        patch(
            "lean.services.ingestion.extract_pdf_markdown",
            return_value=("# DMAIC", 1, ExtractionMethod.MARKITDOWN),
        ),
        patch("lean.services.ingestion.build_sections", return_value=fake_sections),
        patch("lean.services.ingestion.chunk_sections", return_value=fake_chunks),
        patch("lean.services.ingestion.get_embedder", return_value=mock_embedder),
        patch("lean.services.ingestion.extract_metadata", return_value=fake_meta),
        patch("lean.services.ingestion.StoreConnection") as mock_store_cls,
        patch("lean.services.ingestion.DocumentRepo", return_value=mock_doc_repo),
        patch("lean.services.ingestion.ChunkRepo", return_value=mock_chunk_repo),
    ):
        mock_store_cls.from_env.return_value = mock_conn
        result = asyncio.run(ingest_pdf(str(fake_pdf)))

    assert result.chunk_count == 1
    assert result.page_count == 1
    assert result.extraction_method == ExtractionMethod.MARKITDOWN
    mock_embedder.embed_documents.assert_called_once()
    mock_doc_repo.upsert_document.assert_called_once()
    mock_chunk_repo.replace_chunks.assert_called_once()
    mock_conn.close.assert_called_once()

    call_kwargs = mock_doc_repo.upsert_document.call_args.kwargs
    assert call_kwargs["metadata"]["keywords"] == []
    assert call_kwargs["metadata"]["subject"] is None
    assert call_kwargs["metadata"]["toc"] == []
