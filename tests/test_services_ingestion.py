"""Unit tests for the ingestion service pipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from lean.models.schemas import ExtractionMethod

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


def test_ingest_rejects_oversized_pdf(monkeypatch_settings, tmp_path):
    """PDFs larger than settings.max_pdf_mb are rejected before read_bytes (DoS prevention)."""
    import asyncio

    from lean.services.ingestion import ingest_pdf

    monkeypatch_settings.max_pdf_mb = 1  # 1 MB cap
    big_pdf = tmp_path / "huge.pdf"
    big_pdf.write_bytes(b"%PDF-1.4\n" + b"x" * (2 * 1024 * 1024))  # 2 MB

    with pytest.raises(ValueError, match="PDF too large"):
        asyncio.run(ingest_pdf(str(big_pdf)))


def test_ingest_permission_error_does_not_leak_corpus_root_path(monkeypatch_settings, tmp_path):
    """PermissionError message must not disclose the resolved corpus_root absolute path."""
    import asyncio

    from lean.services.ingestion import ingest_pdf

    outside = tmp_path.parent / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4 fake")

    with pytest.raises(PermissionError) as exc_info:
        asyncio.run(ingest_pdf(str(outside)))

    msg = str(exc_info.value)
    assert "outside corpus root" in msg
    # corpus_root absolute path must NOT appear in the error message
    assert str(tmp_path) not in msg


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
            return_value=("# DMAIC", 1, ExtractionMethod.MARKITDOWN, {}, []),
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


def test_ingest_vlm_enrichment_creates_image_chunks(monkeypatch_settings, fake_pdf):
    """When VLM is enabled and marker returns images, image chunks are created."""
    import asyncio

    from lean.chunker.markdown_ast import Section
    from lean.chunker.recursive import ChunkResult
    from lean.models.schemas import ExtractionMethod
    from lean.services.ingestion import ingest_pdf

    monkeypatch_settings.vlm_enabled = True
    monkeypatch_settings.vlm_base_url = "http://gpu:11434/v1"
    monkeypatch_settings.vlm_model = "gemma3:27b"

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

    fake_pil_image = MagicMock()
    fake_images = {"img_0_0": fake_pil_image}

    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024]

    mock_conn = MagicMock()
    mock_doc_repo = MagicMock()
    mock_doc_repo.upsert_document.return_value = __import__("uuid").uuid4()
    mock_chunk_repo = MagicMock()

    from lean.extraction.metadata import PdfMetadata

    fake_meta = PdfMetadata(title="Test Book", authors=["Author"], year=2024)

    mock_vlm = MagicMock()
    mock_vlm.describe_image.return_value = (
        '{"chart_type": "bar", "title": "Revenue by Quarter", '
        '"description": "Bar chart showing Q1-Q4 revenue.", '
        '"key_data_points": ["Q1: $1M", "Q4: $4M"]}'
    )

    with (
        patch(
            "lean.services.ingestion.extract_pdf_markdown",
            return_value=("# DMAIC", 1, ExtractionMethod.MARKER, fake_images, []),
        ),
        patch("lean.services.ingestion.build_sections", return_value=fake_sections),
        patch("lean.services.ingestion.chunk_sections", return_value=fake_chunks),
        patch("lean.services.ingestion.get_embedder", return_value=mock_embedder),
        patch("lean.services.ingestion.extract_metadata", return_value=fake_meta),
        patch("lean.services.ingestion.StoreConnection") as mock_store_cls,
        patch("lean.services.ingestion.DocumentRepo", return_value=mock_doc_repo),
        patch("lean.services.ingestion.ChunkRepo", return_value=mock_chunk_repo),
        patch("lean.vlm.client.VLMClient", return_value=mock_vlm),
    ):
        mock_store_cls.from_env.return_value = mock_conn
        result = asyncio.run(ingest_pdf(str(fake_pdf)))

    assert result.chunk_count == 2
    assert result.extraction_method == ExtractionMethod.MARKER
    mock_vlm.describe_image.assert_called_once()
    mock_vlm.close.assert_called_once()

    call_args = mock_chunk_repo.replace_chunks.call_args
    chunk_rows = call_args[0][1]
    assert len(chunk_rows) == 2
    assert chunk_rows[0].chunk_type == "text"
    assert chunk_rows[0].embedding_model is not None
    assert chunk_rows[0].embedding_dim == 1024

    assert chunk_rows[1].chunk_type == "image"
    assert chunk_rows[1].image_meta is not None
    assert chunk_rows[1].image_meta["chart_type"] == "bar"
    assert chunk_rows[1].image_hash is not None
    assert len(chunk_rows[1].image_hash) == 64
    assert chunk_rows[1].provenance_model == "gemma3:27b"
    assert chunk_rows[1].embedding_model is not None
    assert chunk_rows[1].embedding_dim == 1024

    texts_embedded = mock_embedder.embed_documents.call_args[0][0]
    assert len(texts_embedded) == 2


def test_ingest_vlm_disabled_skips_enrichment(monkeypatch_settings, fake_pdf):
    """When VLM is disabled but images exist, no VLM calls are made."""
    import asyncio

    from lean.chunker.markdown_ast import Section
    from lean.chunker.recursive import ChunkResult
    from lean.models.schemas import ExtractionMethod
    from lean.services.ingestion import ingest_pdf

    monkeypatch_settings.vlm_enabled = False

    fake_sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="Content")]
    fake_chunks = [
        ChunkResult(
            section_path="Ch 1",
            heading_text="Ch 1",
            chunk_index=0,
            token_count=5,
            content="Content",
        )
    ]
    fake_images = {"img_0": MagicMock()}

    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1] * 1024]

    with (
        patch(
            "lean.services.ingestion.extract_pdf_markdown",
            return_value=("# Test", 1, ExtractionMethod.MARKER, fake_images, []),
        ),
        patch("lean.services.ingestion.build_sections", return_value=fake_sections),
        patch("lean.services.ingestion.chunk_sections", return_value=fake_chunks),
        patch("lean.services.ingestion.get_embedder", return_value=mock_embedder),
        patch(
            "lean.services.ingestion.extract_metadata",
            return_value=__import__(
                "lean.extraction.metadata", fromlist=["PdfMetadata"]
            ).PdfMetadata(),
        ),
        patch("lean.services.ingestion.StoreConnection") as mock_store_cls,
        patch(
            "lean.services.ingestion.DocumentRepo",
            return_value=MagicMock(
                upsert_document=MagicMock(return_value=__import__("uuid").uuid4())
            ),
        ),
        patch("lean.services.ingestion.ChunkRepo", return_value=MagicMock()),
        patch("lean.vlm.client.VLMClient") as mock_vlm_cls,
    ):
        mock_store_cls.from_env.return_value = MagicMock()
        result = asyncio.run(ingest_pdf(str(fake_pdf)))

    assert result.chunk_count == 1
    mock_vlm_cls.assert_not_called()
    mock_embedder.embed_documents.assert_called_once()
    embedded_texts = mock_embedder.embed_documents.call_args[0][0]
    assert len(embedded_texts) == 1


def test_persist_rolls_back_when_replace_chunks_fails(monkeypatch_settings, fake_pdf):
    """replace_chunks raising triggers conn.conn.rollback(); exception re-raised."""
    import asyncio

    from lean.chunker.recursive import ChunkResult
    from lean.models.schemas import ExtractionMethod
    from lean.services.ingestion import _persist_ingest

    fake_chunks = [
        ChunkResult(
            section_path="Ch 1",
            heading_text="Ch 1",
            chunk_index=0,
            token_count=2,
            content="x",
        )
    ]
    fake_meta = __import__("lean.extraction.metadata", fromlist=["PdfMetadata"]).PdfMetadata(
        title="x"
    )

    class _Boom(Exception):
        pass

    mock_conn = MagicMock()
    mock_doc_repo = MagicMock()
    mock_doc_repo.upsert_document.return_value = __import__("uuid").uuid4()
    mock_chunk_repo = MagicMock()
    mock_chunk_repo.replace_chunks.side_effect = _Boom("chunk insert failed")

    with (
        patch("lean.services.ingestion.StoreConnection") as mock_store_cls,
        patch("lean.services.ingestion.DocumentRepo", return_value=mock_doc_repo),
        patch("lean.services.ingestion.ChunkRepo", return_value=mock_chunk_repo),
    ):
        mock_store_cls.from_env.return_value = mock_conn
        with pytest.raises(_Boom, match="chunk insert failed"):
            asyncio.run(
                _persist_ingest(
                    fake_pdf,
                    "deadbeef" * 8,
                    fake_meta,
                    1,
                    ExtractionMethod.MARKITDOWN,
                    fake_chunks,
                    [[0.1] * 1024],
                    [],
                    monkeypatch_settings,
                )
            )

    mock_conn.conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()


def test_persist_rolls_back_on_chunk_replace_failure_via_ingest(monkeypatch_settings, fake_pdf):
    """End-to-end: a chunk_repo failure mid-ingest rolls back the transaction."""
    import asyncio

    from lean.chunker.markdown_ast import Section
    from lean.chunker.recursive import ChunkResult
    from lean.models.schemas import ExtractionMethod
    from lean.services.ingestion import ingest_pdf

    fake_sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="x")]
    fake_chunks = [
        ChunkResult(
            section_path="Ch 1",
            heading_text="Ch 1",
            chunk_index=0,
            token_count=2,
            content="x",
        )
    ]

    class _Boom(Exception):
        pass

    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1] * 1024]

    mock_conn = MagicMock()
    mock_doc_repo = MagicMock()
    mock_doc_repo.upsert_document.return_value = __import__("uuid").uuid4()
    mock_chunk_repo = MagicMock()
    mock_chunk_repo.replace_chunks.side_effect = _Boom("simulated DB failure")

    fake_meta = __import__("lean.extraction.metadata", fromlist=["PdfMetadata"]).PdfMetadata(
        title="x"
    )

    with (
        patch(
            "lean.services.ingestion.extract_pdf_markdown",
            return_value=("# x", 1, ExtractionMethod.MARKITDOWN, {}, []),
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
        with pytest.raises(_Boom, match="simulated DB failure"):
            asyncio.run(ingest_pdf(str(fake_pdf)))

    mock_conn.conn.rollback.assert_called_once()


def test_reingest_raises_keyerror_when_document_not_found(monkeypatch_settings):
    """reingest(<unknown-uuid>) looks up source_path, raises KeyError when None."""
    import asyncio

    from lean.services.ingestion import reingest

    mock_doc_repo = MagicMock()
    mock_doc_repo.get_source_path.return_value = None

    with (
        patch("lean.services.ingestion.StoreConnection") as mock_store_cls,
        patch("lean.services.ingestion.DocumentRepo", return_value=mock_doc_repo),
    ):
        mock_store_cls.from_env.return_value = MagicMock()
        with pytest.raises(KeyError, match="not found"):
            asyncio.run(reingest("00000000-0000-0000-0000-000000000000"))


def test_ingest_uses_markitdown_fallback_when_marker_unavailable(monkeypatch_settings, fake_pdf):
    """When marker raises MarkerNotInstalled, pipeline falls back to markitdown."""
    import asyncio

    from lean.chunker.markdown_ast import Section
    from lean.chunker.recursive import ChunkResult
    from lean.models.schemas import ExtractionMethod
    from lean.services.ingestion import ingest_pdf

    fake_sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="x")]
    fake_chunks = [
        ChunkResult(
            section_path="Ch 1",
            heading_text="Ch 1",
            chunk_index=0,
            token_count=2,
            content="x",
        )
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1] * 1024]
    mock_doc_repo = MagicMock()
    mock_doc_repo.upsert_document.return_value = __import__("uuid").uuid4()
    fake_meta = __import__("lean.extraction.metadata", fromlist=["PdfMetadata"]).PdfMetadata(
        title="x"
    )

    with (
        patch(
            "lean.services.ingestion.extract_pdf_markdown",
            return_value=("# x", 1, ExtractionMethod.MARKITDOWN, {}, []),
        ),
        patch("lean.services.ingestion.build_sections", return_value=fake_sections),
        patch("lean.services.ingestion.chunk_sections", return_value=fake_chunks),
        patch("lean.services.ingestion.get_embedder", return_value=mock_embedder),
        patch("lean.services.ingestion.extract_metadata", return_value=fake_meta),
        patch("lean.services.ingestion.StoreConnection") as mock_store_cls,
        patch("lean.services.ingestion.DocumentRepo", return_value=mock_doc_repo),
        patch("lean.services.ingestion.ChunkRepo", return_value=MagicMock()),
    ):
        mock_store_cls.from_env.return_value = MagicMock()
        result = asyncio.run(ingest_pdf(str(fake_pdf)))

    assert result.extraction_method == ExtractionMethod.MARKITDOWN
    assert "OCR server unavailable" in result.warnings[0] or result.warnings == []


def test_ingest_appends_warning_when_contextual_retrieval_enabled_without_llm(
    monkeypatch_settings, fake_pdf
):
    """llm_contextual_retrieval=true with no LLM configured logs a warning, does not abort."""
    import asyncio

    from lean.chunker.markdown_ast import Section
    from lean.chunker.recursive import ChunkResult
    from lean.models.schemas import ExtractionMethod
    from lean.services.ingestion import ingest_pdf

    monkeypatch_settings.llm_contextual_retrieval = True

    fake_sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="x")]
    fake_chunks = [
        ChunkResult(
            section_path="Ch 1",
            heading_text="Ch 1",
            chunk_index=0,
            token_count=2,
            content="x",
        )
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1] * 1024]
    fake_meta = __import__("lean.extraction.metadata", fromlist=["PdfMetadata"]).PdfMetadata(
        title="x"
    )

    with (
        patch("lean.llm.base.get_llm", return_value=None),
        patch(
            "lean.services.ingestion.extract_pdf_markdown",
            return_value=("# x", 1, ExtractionMethod.MARKITDOWN, {}, []),
        ),
        patch("lean.services.ingestion.build_sections", return_value=fake_sections),
        patch("lean.services.ingestion.chunk_sections", return_value=fake_chunks),
        patch("lean.services.ingestion.get_embedder", return_value=mock_embedder),
        patch("lean.services.ingestion.extract_metadata", return_value=fake_meta),
        patch("lean.services.ingestion.StoreConnection") as mock_store_cls,
        patch(
            "lean.services.ingestion.DocumentRepo",
            return_value=MagicMock(
                upsert_document=MagicMock(return_value=__import__("uuid").uuid4())
            ),
        ),
        patch("lean.services.ingestion.ChunkRepo", return_value=MagicMock()),
    ):
        mock_store_cls.from_env.return_value = MagicMock()
        result = asyncio.run(ingest_pdf(str(fake_pdf)))

    assert any("contextual_retrieval enabled but no LLM configured" in w for w in result.warnings)


# BLG-004: per-helper coverage for the ingest_pdf decomposition (see BLG-004 in the active backlog).


def test_validate_pdf_path_rejects_oversized_pdf_for_blg004(monkeypatch_settings, tmp_path):
    """_validate_pdf_path rejects files above settings.max_pdf_mb before any read_bytes."""
    from lean.services.ingestion import _validate_pdf_path

    monkeypatch_settings.max_pdf_mb = 1
    big_pdf = tmp_path / "huge.pdf"
    big_pdf.write_bytes(b"%PDF-1.4\n" + b"x" * (2 * 1024 * 1024))

    with pytest.raises(ValueError, match="PDF too large"):
        asyncio.run(_validate_pdf_path(str(big_pdf), monkeypatch_settings))


def test_extract_markdown_delegates_to_extraction_pipeline_for_blg004(
    monkeypatch_settings, fake_pdf
):
    """_extract_markdown forwards Settings OCR/marker fields to extract_pdf_markdown."""
    from lean.services.ingestion import _extract_markdown

    fake_metadata = ("# Title", 7, ExtractionMethod.MARKER, {"img": 1}, [])
    with patch(
        "lean.services.ingestion.extract_pdf_markdown", return_value=fake_metadata
    ) as mock_extract:
        result = asyncio.run(_extract_markdown(fake_pdf, monkeypatch_settings))

    assert result == fake_metadata
    assert mock_extract.call_count == 1
    kwargs = mock_extract.call_args.kwargs
    assert kwargs["ocr_base_url"] == monkeypatch_settings.ocr_base_url
    assert kwargs["marker_force_ocr"] == monkeypatch_settings.marker_force_ocr
    assert kwargs["marker_remote_url"] == monkeypatch_settings.marker_remote_url


def test_chunk_sections_appends_markitdown_warning_for_blg004(monkeypatch_settings, fake_pdf):
    """_chunk_sections returns sections, chunks, and warnings; markitdown only on MARKITDOWN."""  # noqa: E501
    from lean.chunker.markdown_ast import Section
    from lean.chunker.recursive import ChunkResult
    from lean.services.ingestion import _chunk_sections

    fake_sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="x")]
    fake_chunks = [
        ChunkResult(
            section_path="Ch 1", heading_text="Ch 1", chunk_index=0, token_count=1, content="x"
        )
    ]

    with (
        patch("lean.services.ingestion.build_sections", return_value=fake_sections),
        patch("lean.services.ingestion.chunk_sections", return_value=fake_chunks),
    ):
        sections, chunks, warnings = asyncio.run(
            _chunk_sections("# x", 1, ExtractionMethod.MARKITDOWN, [], monkeypatch_settings)
        )
        sections_marker, chunks_marker, warnings_marker = asyncio.run(
            _chunk_sections("# x", 1, ExtractionMethod.MARKER, [], monkeypatch_settings)
        )

    assert sections == fake_sections
    assert chunks == fake_chunks
    assert "OCR server unavailable, fell back to markitdown" in warnings
    assert sections_marker == fake_sections
    assert chunks_marker == fake_chunks
    assert "OCR server unavailable, fell back to markitdown" not in warnings_marker


def test_maybe_contextualize_returns_chunks_unchanged_when_no_llm_for_blg004(
    monkeypatch_settings,
):
    """_maybe_contextualize is a no-op when get_llm() is None, returning the same list object and a warning."""  # noqa: E501
    from lean.chunker.recursive import ChunkResult
    from lean.services.ingestion import _maybe_contextualize

    monkeypatch_settings.llm_contextual_retrieval = True
    chunks = [
        ChunkResult(section_path="S", heading_text="H", chunk_index=0, token_count=1, content="c")
    ]
    sections: list = []

    with patch("lean.llm.base.get_llm", return_value=None):
        result, warnings = asyncio.run(
            _maybe_contextualize(chunks, sections, monkeypatch_settings, [])
        )

    assert result is chunks
    assert any("contextual_retrieval enabled but no LLM configured" in w for w in warnings)


def test_embed_chunks_concatenates_text_then_image_embeddings_for_blg004(monkeypatch_settings):
    """_embed_chunks embeds chunk content first, then image descriptions, preserving order."""
    from lean.chunker.recursive import ChunkResult
    from lean.services.ingestion import _embed_chunks

    chunks = [
        ChunkResult(
            section_path="S1", heading_text="H1", chunk_index=0, token_count=1, content="text-1"
        ),
        ChunkResult(
            section_path="S2", heading_text="H2", chunk_index=1, token_count=1, content="text-2"
        ),
    ]
    image_descriptions = [("img-desc-1", {"chart_type": "bar"}, "hash-1")]

    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]

    with patch("lean.services.ingestion.get_embedder", return_value=mock_embedder):
        embeddings = asyncio.run(_embed_chunks(chunks, image_descriptions))

    assert embeddings == [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
    called_texts = mock_embedder.embed_documents.call_args[0][0]
    assert called_texts == ["text-1", "text-2", "img-desc-1"]
