"""Pydantic schemas shared across MCP tools, API routes, and storage layer."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 (needed at runtime by Pydantic)
from enum import StrEnum

from pydantic import BaseModel, Field


class ExtractionMethod(StrEnum):
    """Which extraction path produced this document's markdown."""

    UNLIMITED_OCR = "unlimited_ocr"
    MARKITDOWN = "markitdown"


class DocumentSummary(BaseModel):
    """Lightweight document view returned by ``list_documents``."""

    id: str
    source_path: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    page_count: int | None = None
    extraction_method: ExtractionMethod
    chunk_count: int
    ingested_at: datetime


class Chunk(BaseModel):
    """A single retrievable text chunk with its corpus location.

    ``score`` is populated only by ``search()``; ``get_chunk`` leaves it ``None``.
    """

    id: str
    document_id: str
    chunk_index: int
    section_path: str
    heading_text: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    token_count: int
    content: str
    score: float | None = None


class IngestResult(BaseModel):
    """Return value of ``ingest_pdf`` and ``reingest`` tools."""

    document_id: str
    source_sha256: str
    page_count: int
    extraction_method: ExtractionMethod
    chunk_count: int
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)


class CorpusStats(BaseModel):
    """Return value of ``corpus_stats`` tool."""

    document_count: int
    chunk_count: int
    total_tokens: int
    extraction_method_breakdown: dict[str, int]
    embedding_dim: int
    embedding_model: str
    last_ingested_at: datetime | None = None
