"""Compatibility shim — delegates to the focused store modules.

The original 417-LOC ``PgVectorStore`` monolith was split into:

    StoreConnection  (base)       — connection management
    DocumentRepo     (documents)  — document CRUD
    ChunkRepo        (chunks)     — chunk CRUD
    SearchEngine     (search)     — vector + BM25 search, RRF
    AnalyticsRepo    (analytics)  — query logging, corpus stats

New code should import from those modules directly. ``PgVectorStore`` is
retained here as a facade that wires the repos together and re-exposes the
old method names, so existing callers keep working without modification.
It will be removed once all callers migrate to the focused repos.
"""

from __future__ import annotations

from uuid import UUID  # noqa: TC003

from lean.models.schemas import Chunk, CorpusStats, DocumentSummary
from lean.store.analytics import AnalyticsRepo
from lean.store.base import StoreConnection
from lean.store.chunks import ChunkRepo, ChunkRow
from lean.store.documents import DocumentRepo
from lean.store.search import SearchEngine, SearchHit

__all__ = [
    "AnalyticsRepo",
    "ChunkRepo",
    "ChunkRow",
    "DocumentRepo",
    "PgVectorStore",
    "SearchEngine",
    "SearchHit",
    "StoreConnection",
]


class PgVectorStore(StoreConnection):
    """Legacy facade over the focused repos.

    Preserves the original monolith's method surface so existing callers
    (``store.search()``, ``store.upsert_document()``, ``store._conn`` …)
    keep working. Prefer the individual repo modules in new code.
    """

    def __init__(self, db_url: str) -> None:
        super().__init__(db_url)
        self._documents = DocumentRepo(self)
        self._chunks_repo = ChunkRepo(self)
        self._search = SearchEngine(self)
        self._analytics = AnalyticsRepo(self)

    # --- documents ---
    def upsert_document(self, **kwargs: object) -> UUID:
        return self._documents.upsert_document(**kwargs)  # type: ignore[arg-type]

    def delete_document(self, document_id: UUID) -> None:
        self._documents.delete_document(document_id)

    def list_documents(self) -> list[DocumentSummary]:
        return self._documents.list_documents()

    def get_document_storage_paths(self, document_id: UUID) -> tuple[str, str] | None:
        return self._documents.get_document_storage_paths(document_id)

    # --- chunks ---
    def replace_chunks(self, document_id: UUID, chunks: list[ChunkRow]) -> None:
        self._chunks_repo.replace_chunks(document_id, chunks)

    def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        return self._chunks_repo.get_chunk(chunk_id)

    def count_chunks(self) -> int:
        return self._chunks_repo.count_chunks()

    # --- search ---
    def search(self, **kwargs: object) -> list[SearchHit]:
        return self._search.vector_search(**kwargs)  # type: ignore[arg-type]

    def bm25_search(self, **kwargs: object) -> list[SearchHit]:
        return self._search.bm25_search(**kwargs)  # type: ignore[arg-type]

    @staticmethod
    def reciprocal_rank_fusion(
        vector_hits: list[SearchHit],
        bm25_hits: list[SearchHit],
        *,
        k: int = 5,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        return SearchEngine.reciprocal_rank_fusion(vector_hits, bm25_hits, k=k, rrf_k=rrf_k)

    # --- analytics ---
    def log_query(self, **kwargs: object) -> None:
        self._analytics.log_query(**kwargs)  # type: ignore[arg-type]

    def count_documents(self) -> int:
        return self._analytics.count_documents()

    def corpus_stats(self, *, embedding_dim: int, embedding_model: str) -> CorpusStats:
        return self._analytics.corpus_stats(
            embedding_dim=embedding_dim, embedding_model=embedding_model
        )
