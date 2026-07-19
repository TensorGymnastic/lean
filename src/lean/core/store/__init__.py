"""Universal pgvector CRUD — split into focused repos.

Modules:
    base       — StoreConnection: owns the psycopg connection.
    documents  — DocumentRepo: documents table CRUD.
    chunks     — ChunkRepo + ChunkRow: chunks table CRUD (universal columns).
    search     — SearchEngine + SearchHit: vector/BM25 search + RRF.
    analytics  — AnalyticsRepo: query logging + corpus stats.

Domains extend with their own repos for schema-specific columns
(e.g. lean-lss adds ImageChunkRepo for ``image_hash``, ``bbox``).
"""

from lean.core.store.analytics import AnalyticsRepo
from lean.core.store.base import StoreConnection
from lean.core.store.chunks import ChunkRepo, ChunkRow
from lean.core.store.documents import DocumentRepo
from lean.core.store.search import SearchEngine, SearchHit

__all__ = [
    "StoreConnection",
    "DocumentRepo",
    "ChunkRepo",
    "ChunkRow",
    "SearchEngine",
    "SearchHit",
    "AnalyticsRepo",
]
