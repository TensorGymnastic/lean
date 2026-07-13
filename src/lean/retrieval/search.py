"""High-level retrieval: embed query → pgvector cosine search + query logging."""

from __future__ import annotations

import logging
import time
from uuid import UUID

from lean.config.settings import Settings
from lean.infrastructure.embedder import get_embedder
from lean.models.schemas import Chunk
from lean.store.pgvector import PgVectorStore

logger = logging.getLogger(__name__)

_store: PgVectorStore | None = None


def _get_store() -> PgVectorStore:
    global _store
    if _store is None:
        _store = PgVectorStore.from_env()
    return _store


def search(
    query: str,
    *,
    k: int = 5,
    doc_id: str | None = None,
    section: str | None = None,
    author: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    min_score: float | None = None,
    agent_id: str | None = None,
) -> list[Chunk]:
    """Embed query → search (hybrid if enabled) → rerank (if enabled) → postprocess."""
    settings = Settings()
    embedder = get_embedder()
    store = _get_store()

    start = time.monotonic()
    query_vec = embedder.embed_query(query)

    doc_uuid = UUID(doc_id) if doc_id else None

    if settings.hybrid_search_enabled:
        fetch_k = max(k * 4, 20)
        vector_hits = store.search(
            query_embedding=query_vec,
            k=fetch_k,
            doc_id=doc_uuid,
            section_substring=section,
            author=author,
            year_min=year_min,
            year_max=year_max,
            min_score=min_score,
        )
        bm25_hits = store.bm25_search(
            query_text=query,
            k=fetch_k,
            doc_id=doc_uuid,
            section_substring=section,
            author=author,
            year_min=year_min,
            year_max=year_max,
        )
        hits = store.reciprocal_rank_fusion(vector_hits, bm25_hits, k=fetch_k)
    else:
        hits = store.search(
            query_embedding=query_vec,
            k=k,
            doc_id=doc_uuid,
            section_substring=section,
            author=author,
            year_min=year_min,
            year_max=year_max,
            min_score=min_score,
        )

    if settings.rerank_enabled and hits:
        from lean.retrieval.reranker import rerank as _rerank

        hits = _rerank(
            hits,
            query,
            model=settings.rerank_model,
            top_n=settings.rerank_top_n,
            device=settings.embedding_device,
        )

    effective_min = min_score if min_score is not None else settings.min_similarity
    if effective_min > 0:
        from lean.retrieval.postprocessors import similarity_filter

        hits = similarity_filter(hits, effective_min)

    from lean.retrieval.postprocessors import long_context_reorder

    hits = long_context_reorder(hits)

    hits = hits[:k]
    latency_ms = int((time.monotonic() - start) * 1000)

    filters: dict[str, object] = {
        "doc_id": doc_id,
        "section": section,
        "author": author,
        "year_min": year_min,
        "year_max": year_max,
        "min_score": min_score,
        "hybrid": settings.hybrid_search_enabled,
        "reranked": settings.rerank_enabled,
    }
    try:
        store.log_query(
            query_text=query,
            k=k,
            filters=filters,
            hit_chunk_ids=[UUID(h.chunk.id) for h in hits],
            hit_scores=[h.score for h in hits],
            latency_ms=latency_ms,
            agent_id=agent_id,
        )
    except Exception:
        logger.warning("failed to log query", exc_info=True)

    logger.info(
        "search q=%r k=%d hits=%d latency=%dms hybrid=%s reranked=%s",
        query[:60],
        k,
        len(hits),
        latency_ms,
        settings.hybrid_search_enabled,
        settings.rerank_enabled,
    )
    return [h.chunk for h in hits]
