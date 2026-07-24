"""Corpus management service: read/delete operations on stored documents.

Thin CRUD wrappers over the focused store repos. Each function opens its
own ``StoreConnection``, delegates, and closes. Extracted from
``mcp_server.tools`` so transports stay thin.
"""

from __future__ import annotations

from uuid import UUID

from lean.core.config.settings import get_settings
from lean.core.models.schemas import Chunk, CorpusStats, DocumentSummary
from lean.core.store.analytics import AnalyticsRepo
from lean.core.store.base import StoreConnection
from lean.core.store.chunks import ChunkRepo
from lean.core.store.documents import DocumentRepo


def list_documents() -> list[DocumentSummary]:
    """List all documents in the corpus, newest first."""
    with StoreConnection.from_env() as conn:
        return DocumentRepo(conn).list_documents()


def delete_document(document_id: str) -> dict[str, str]:
    """Delete a document and cascade-delete its chunks.

    Returns ``{"deleted": document_id}`` on success. Raises ``KeyError``
    if the document id is not a valid UUID.
    """
    doc_uuid = UUID(document_id)
    with StoreConnection.from_env() as conn:
        DocumentRepo(conn).delete_document(doc_uuid)
    return {"deleted": document_id}


def corpus_stats() -> CorpusStats:
    """Return corpus statistics: document/chunk counts, extraction breakdown.

    ``embedding_dim`` and ``embedding_model`` are sourced from ``Settings``
    so the response is self-describing without an embedding round-trip.
    """
    settings = get_settings()
    with StoreConnection.from_env() as conn:
        return AnalyticsRepo(conn).corpus_stats(
            embedding_dim=settings.embedding_dim,
            embedding_model=settings.embedding_model,
        )


def get_chunk(chunk_id: str) -> Chunk | None:
    """Fetch a single chunk by its UUID. Returns ``None`` if not found."""
    with StoreConnection.from_env() as conn:
        return ChunkRepo(conn).get_chunk(UUID(chunk_id))


def list_chunks_by_document(document_id: str) -> list[Chunk]:
    """List chunks for a document, ordered by chunk_index. No embeddings."""
    doc_uuid = UUID(document_id)
    with StoreConnection.from_env() as conn:
        return ChunkRepo(conn).get_chunks_by_document(doc_uuid)


def get_document_markdown(document_id: str) -> str:
    """Return the document's markdown by reconstructing from stored chunks.

    Raises ``KeyError`` if the document id does not exist.
    """
    doc_uuid = UUID(document_id)
    with StoreConnection.from_env() as conn:
        chunks = ChunkRepo(conn).get_chunks_by_document(doc_uuid)
    if not chunks:
        raise KeyError(f"document {document_id} not found or has no chunks")
    return "\n\n".join(c.content for c in chunks)
