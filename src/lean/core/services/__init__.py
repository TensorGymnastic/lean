"""Universal service orchestration: ingestion, search, corpus CRUD.

Each service opens its own ``StoreConnection`` (per-call) and closes
when done. Services are the only place business logic lives — transports
(CLI / MCP / REST) and stores (CRUD repos) are thin.
"""

from lean.core.services.corpus import (
    corpus_stats,
    delete_document,
    get_chunk,
    get_document_markdown,
    list_chunks_by_document,
    list_documents,
)
from lean.core.services.ingestion import ingest_pdf, reingest
from lean.core.services.search import SearchRequest, search

__all__ = [
    "search",
    "SearchRequest",
    "ingest_pdf",
    "reingest",
    "list_documents",
    "list_chunks_by_document",
    "get_chunk",
    "get_document_markdown",
    "delete_document",
    "corpus_stats",
]
