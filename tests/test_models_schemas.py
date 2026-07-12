"""Tests for lean.models.schemas."""

from __future__ import annotations

from datetime import datetime

from lean.models.schemas import Chunk, CorpusStats, DocumentSummary, IngestResult


def test_document_summary_round_trip() -> None:
    """DocumentSummary survives model_dump -> model_validate."""
    doc = DocumentSummary(
        id="abc-123",
        source_path="data/foo.pdf",
        title="Foo",
        authors=["Alice", "Bob"],
        page_count=100,
        extraction_method="unlimited_ocr",
        chunk_count=42,
        ingested_at=datetime(2026, 7, 12, 14, 0, 0),
    )
    data = doc.model_dump(mode="json")
    round_tripped = DocumentSummary.model_validate(data)
    assert round_tripped == doc


def test_chunk_score_optional() -> None:
    """Chunk.score defaults to None (populated only by search)."""
    chunk = Chunk(
        id="c1",
        document_id="d1",
        chunk_index=0,
        section_path="Chapter 1 > Intro",
        heading_text="Intro",
        page_start=1,
        page_end=2,
        token_count=400,
        content="Some content",
    )
    assert chunk.score is None

    # Can set score
    chunk.score = 0.95
    assert chunk.score == 0.95


def test_ingest_result_warnings_default_empty() -> None:
    """IngestResult.warnings defaults to empty list."""
    result = IngestResult(
        document_id="d1",
        source_sha256="abc",
        page_count=10,
        extraction_method="markitdown",
        chunk_count=5,
        elapsed_seconds=3.2,
    )
    assert result.warnings == []


def test_corpus_stats_serialization() -> None:
    """CorpusStats with breakdown dict serializes correctly."""
    stats = CorpusStats(
        document_count=4,
        chunk_count=200,
        total_tokens=80000,
        extraction_method_breakdown={"unlimited_ocr": 3, "markitdown": 1},
        embedding_dim=1024,
        embedding_model="LiquidAI/LFM2.5-Embedding-350M",
        last_ingested_at=datetime(2026, 7, 12, 14, 0, 0),
    )
    assert stats.extraction_method_breakdown["unlimited_ocr"] == 3
    json_str = stats.model_dump_json()
    assert "unlimited_ocr" in json_str


def test_corpus_stats_last_ingested_optional() -> None:
    """CorpusStats.last_ingested_at can be None for empty corpus."""
    stats = CorpusStats(
        document_count=0,
        chunk_count=0,
        total_tokens=0,
        extraction_method_breakdown={},
        embedding_dim=1024,
        embedding_model="LiquidAI/LFM2.5-Embedding-350M",
    )
    assert stats.last_ingested_at is None
