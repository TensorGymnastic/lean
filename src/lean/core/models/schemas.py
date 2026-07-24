"""Pydantic schemas shared across MCP tools, API routes, and storage layer.

Universal types live here. Domain-specific extensions (e.g.
``ChunkType`` for multimodal corpora) belong in the domain package.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 (needed at runtime by Pydantic)
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

CHUNK_TYPE_TEXT = "text"
CHUNK_TYPE_IMAGE = "image"
"""Universal chunk-type literals. Mirrored at the DB layer by a CHECK
constraint that domains add in their own migration (e.g. ``012_add_chunk_type_check.sql``).
"""


class ExtractionMethod(StrEnum):
    """Which extraction path produced a document's content.

    The three values below match the universal backends shipped in
    ``lean.core.extraction``. Domains that add their own backends
    (e.g. Docling) should subclass this enum and add their values.
    """

    UNKNOWN = "unknown"
    MARKER = "marker"
    UNLIMITED_OCR = "unlimited_ocr"
    MARKITDOWN = "markitdown"
    TEXT_FILE = "text_file"
    WEB_URL = "web_url"


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
    """A single retrievable chunk with its corpus location.

    Universal fields only. Domains extend with their own optional fields
    (e.g. ``image_hash`` for multimodal corpora, ``code_symbol`` for
    source-code corpora).
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
    score: float | None = None

    @classmethod
    def from_row(cls, r: dict[str, Any]) -> Chunk:
        """Build a Chunk from a psycopg dict_row (no score, universal fields).

        Uses ``.get()`` for nullable fields so partial SELECT lists still work.
        """
        return cls(
            id=str(r["id"]),
            document_id=str(r["document_id"]),
            chunk_index=r["chunk_index"],
            section_path=r["section_path"],
            heading_text=r.get("heading_text"),
            page_start=r.get("page_start"),
            page_end=r.get("page_end"),
            token_count=r["token_count"],
            content=r["content"],
            chunk_type=r.get("chunk_type", CHUNK_TYPE_TEXT),
        )


class IngestResult(BaseModel):
    """Return value of ``ingest`` / ``reingest`` operations."""

    document_id: str
    source_sha256: str
    page_count: int
    extraction_method: ExtractionMethod
    chunk_count: int
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)


class CorpusStats(BaseModel):
    """Return value of ``corpus_stats`` operation.

    Universal fields only. Domains may add their own diagnostic fields
    (e.g. duplicate image hashes for multimodal corpora) by subclassing
    this base.
    """

    document_count: int
    chunk_count: int
    total_tokens: int
    extraction_method_breakdown: dict[str, int]
    embedding_dim: int
    embedding_model: str
    last_ingested_at: datetime | None = None


__all__ = [
    "CHUNK_TYPE_TEXT",
    "CHUNK_TYPE_IMAGE",
    "ExtractionMethod",
    "DocumentSummary",
    "Chunk",
    "IngestResult",
    "CorpusStats",
]
