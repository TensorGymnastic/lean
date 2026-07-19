"""Pydantic schemas shared across MCP tools, API routes, and storage layer."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 (needed at runtime by Pydantic)
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

CHUNK_TYPE_TEXT = "text"
CHUNK_TYPE_IMAGE = "image"
"""Single source of truth for ``chunks.chunk_type`` values. Mirrored at
the DB level by ``db/schemas/012_add_chunk_type_check.sql`` (CHECK
constraint enforcing the same two values)."""


class ExtractionMethod(StrEnum):
    """Which extraction path produced this document's markdown."""

    MARKER = "marker"
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
    chunk_type: str = CHUNK_TYPE_TEXT
    image_meta: dict[str, Any] | None = None
    bbox: dict[str, Any] | None = None
    image_hash: str | None = None
    provenance_model: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    score: float | None = None

    @classmethod
    def from_row(cls, r: dict[str, Any]) -> Chunk:
        """Build a Chunk from a psycopg dict_row (no score)."""
        return cls(
            id=str(r["id"]),
            document_id=str(r["document_id"]),
            chunk_index=r["chunk_index"],
            section_path=r["section_path"],
            heading_text=r["heading_text"],
            page_start=r["page_start"],
            page_end=r["page_end"],
            token_count=r["token_count"],
            content=r["content"],
            chunk_type=r.get("chunk_type", CHUNK_TYPE_TEXT),
            image_meta=r.get("image_meta"),
            bbox=r.get("bbox"),
            image_hash=r.get("image_hash"),
            provenance_model=r.get("provenance_model"),
            embedding_model=r.get("embedding_model"),
            embedding_dim=r.get("embedding_dim"),
        )


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
    duplicate_image_hashes: list[dict[str, object]] = Field(default_factory=list)
    """Diagnostic: image SHA-256 hashes appearing in more than one chunk.
    Each entry is ``{"image_hash": str, "count": int}``. Empty when no
    duplicates exist. Surfaced via ``corpus_stats`` so the dedup query has
    a reader (was previously a store method with no caller)."""
