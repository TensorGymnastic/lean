"""Search service: embed query → vector/BM25 search → rerank → postprocess.

Orchestrates the retrieval pipeline end-to-end. Extracted from
``retrieval.search.search`` so transports can call it without inheriting
the module-level store singleton. Each call opens its own
``StoreConnection`` and closes it when done.
"""

from __future__ import annotations

import logging
import time
from uuid import UUID

from lean.config.settings import Settings
from lean.infrastructure.embedder import get_embedder
from lean.models.schemas import Chunk
from lean.store.analytics import AnalyticsRepo
from lean.store.base import StoreConnection
from lean.store.search import SearchEngine

logger = logging.getLogger(__name__)


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
    """Embed query → search (hybrid if enabled) → rerank (if enabled) → postprocess.

    Pipeline:
        1. Embed the query with the singleton LiquidLMF embedder.
        2. If ``hybrid_search_enabled``: fan out to vector + BM25 and fuse
           via Reciprocal Rank Fusion. Otherwise vector search only.
        3. If ``rerank_enabled`` and there are hits: cross-encoder rerank.
        4. Apply ``similarity_filter`` (min_score cutoff, defaulting to
           ``settings.min_similarity`` when not supplied).
        5. Apply ``long_context_reorder`` to combat "lost in the middle".
        6. Truncate to ``k``.
        7. Log the query (best-effort) via ``AnalyticsRepo``.
    """
    settings = Settings()
    embedder = get_embedder()

    start = time.monotonic()
    query_vec = embedder.embed_query(query)
    doc_uuid = UUID(doc_id) if doc_id else None

    conn = StoreConnection.from_env()
    try:
        engine = SearchEngine(conn)
        analytics = AnalyticsRepo(conn)

        if settings.hybrid_search_enabled:
            fetch_k = max(k * settings.fetch_multiplier, 40)
            vector_hits = engine.vector_search(
                query_embedding=query_vec,
                k=fetch_k,
                doc_id=doc_uuid,
                section_substring=section,
                author=author,
                year_min=year_min,
                year_max=year_max,
                min_score=min_score,
            )
            bm25_hits = engine.bm25_search(
                query_text=query,
                k=fetch_k,
                doc_id=doc_uuid,
                section_substring=section,
                author=author,
                year_min=year_min,
                year_max=year_max,
            )
            hits = engine.reciprocal_rank_fusion(vector_hits, bm25_hits, k=fetch_k)
        else:
            hits = engine.vector_search(
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
            analytics.log_query(
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
    finally:
        conn.close()

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
