"""Pydantic schemas shared across lean-core. Domains extend with their
own typed fields.
"""

from lean.core.models.schemas import (
    CHUNK_TYPE_IMAGE,
    CHUNK_TYPE_TEXT,
    Chunk,
    CorpusStats,
    DocumentSummary,
    ExtractionMethod,
    IngestResult,
)

__all__ = [
    "CHUNK_TYPE_TEXT",
    "CHUNK_TYPE_IMAGE",
    "ExtractionMethod",
    "DocumentSummary",
    "Chunk",
    "IngestResult",
    "CorpusStats",
]
