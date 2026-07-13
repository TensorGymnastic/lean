"""Corpus management service: read/delete operations on stored documents.

Thin CRUD wrappers over the focused store repos. Each function opens its
own ``StoreConnection``, delegates, and closes. Extracted from
``mcp_server.tools`` so transports stay thin.
"""

from __future__ import annotations

from uuid import UUID

from lean.config.settings import Settings
from lean.models.schemas import Chunk, CorpusStats, DocumentSummary
from lean.store.analytics import AnalyticsRepo
from lean.store.base import StoreConnection
from lean.store.chunks import ChunkRepo
from lean.store.documents import DocumentRepo


def list_documents() -> list[DocumentSummary]:
    """List all documents in the corpus, newest first."""
    conn = StoreConnection.from_env()
    try:
        return DocumentRepo(conn).list_documents()
    finally:
        conn.close()


def delete_document(document_id: str) -> dict[str, str]:
    """Delete a document and cascade-delete its chunks.

    Returns ``{"deleted": document_id}`` on success. Raises ``KeyError``
    if the document id is not a valid UUID.
    """
    doc_uuid = UUID(document_id)
    conn = StoreConnection.from_env()
    try:
        DocumentRepo(conn).delete_document(doc_uuid)
    finally:
        conn.close()
    return {"deleted": document_id}


def corpus_stats() -> CorpusStats:
    """Return corpus statistics: document/chunk counts, extraction breakdown.

    ``embedding_dim`` and ``embedding_model`` are sourced from ``Settings``
    so the response is self-describing without an embedding round-trip.
    """
    settings = Settings()
    conn = StoreConnection.from_env()
    try:
        return AnalyticsRepo(conn).corpus_stats(
            embedding_dim=settings.embedding_dim,
            embedding_model=settings.embedding_model,
        )
    finally:
        conn.close()


def get_chunk(chunk_id: str) -> Chunk | None:
    """Fetch a single chunk by its UUID. Returns ``None`` if not found."""
    conn = StoreConnection.from_env()
    try:
        return ChunkRepo(conn).get_chunk(UUID(chunk_id))
    finally:
        conn.close()


def get_document_markdown(document_id: str) -> str:
    """Return a placeholder string pointing at the document's markdown storage path.

    The document's full markdown is stored in Supabase Storage; fetching
    the object will be wired in a follow-up. Raises ``KeyError`` if the
    document id does not exist.
    """
    doc_uuid = UUID(document_id)
    conn = StoreConnection.from_env()
    try:
        paths = DocumentRepo(conn).get_document_storage_paths(doc_uuid)
    finally:
        conn.close()
    if paths is None:
        raise KeyError(f"document {document_id} not found")
    # paths = (source_storage_path, markdown_storage_path)
    markdown_storage_path = paths[1]
    return f"[markdown at storage path {markdown_storage_path}]"
