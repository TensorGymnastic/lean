"""Search service: query transform → embed → search → rerank → postprocess.

Orchestrates the retrieval pipeline end-to-end. Supports optional LLM-powered
query transforms (HyDE, multi-query) when an LLM sidecar is configured.
Each call opens its own StoreConnection and closes it when done.
"""

from __future__ import annotations

import logging
import time
from uuid import UUID

from lean.config.settings import get_settings
from lean.infrastructure.embedder import get_embedder
from lean.llm.base import get_llm
from lean.models.schemas import Chunk
from lean.store.analytics import AnalyticsRepo
from lean.store.base import StoreConnection
from lean.store.search import SearchEngine, SearchHit

logger = logging.getLogger(__name__)

VALID_CHUNK_TYPES = frozenset({"text", "image"})


def search(
    query: str,
    *,
    k: int | None = None,
    doc_id: str | None = None,
    section: str | None = None,
    author: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    min_score: float | None = None,
    chunk_type: str | None = None,
) -> list[Chunk]:
    """Embed query → search (hybrid if enabled) → rerank → postprocess.

    Pipeline:
        1. (Optional) Multi-query: generate N paraphrases via LLM.
        2. (Optional) HyDE: generate hypothetical doc for vector embedding.
        3. Embed query → vector search + BM25 → RRF fusion.
        4. (Optional) Cross-encoder rerank.
        5. Similarity filter + long-context reorder.
        6. Truncate to k.
    """
    settings = get_settings()
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if chunk_type is not None and chunk_type not in VALID_CHUNK_TYPES:
        raise ValueError(
            f"chunk_type must be one of {sorted(VALID_CHUNK_TYPES)} or None, got '{chunk_type}'"
        )
    if len(query) > settings.search_max_query_len:
        raise ValueError(f"query exceeds max length ({settings.search_max_query_len} chars)")
    if k is None:
        k = settings.search_top_k
    k = max(1, min(k, settings.search_max_k))
    embedder = get_embedder()

    start = time.monotonic()
    doc_uuid = UUID(doc_id) if doc_id else None

    queries = [query]
    llm = get_llm()
    if settings.llm_multi_query and llm:
        from lean.retrieval.query_transform import multi_query_transform

        queries = multi_query_transform(
            query,
            llm,
            num_queries=settings.llm_multi_query_count,
            max_tokens=settings.llm_generate_max_tokens,
            temperature=settings.llm_generate_temperature,
        )

    conn = StoreConnection.from_env()
    hits: list[SearchHit] = []
    latency_ms = -1
    try:
        engine = SearchEngine(conn)
        analytics = AnalyticsRepo(conn)

        fetch_k = min(
            max(k * settings.fetch_multiplier, settings.fetch_k_floor)
            if settings.hybrid_search_enabled
            else k,
            settings.search_fetch_k_cap,
        )

        fused_lists: list[list[SearchHit]] = []
        for q in queries:
            embed_text = q
            if settings.llm_hyde and llm:
                from lean.retrieval.query_transform import hyde_transform

                embed_text = hyde_transform(
                    q,
                    llm,
                    max_tokens=settings.llm_generate_max_tokens,
                    temperature=settings.llm_generate_temperature,
                )

            query_vec = embedder.embed_query(embed_text)

            if settings.hybrid_search_enabled:
                vector_hits = engine.vector_search(
                    query_embedding=query_vec,
                    k=fetch_k,
                    doc_id=doc_uuid,
                    section_substring=section,
                    author=author,
                    year_min=year_min,
                    year_max=year_max,
                    min_score=min_score,
                    chunk_type=chunk_type,
                )
                bm25_hits = engine.bm25_search(
                    query_text=q,
                    k=fetch_k,
                    doc_id=doc_uuid,
                    section_substring=section,
                    author=author,
                    year_min=year_min,
                    year_max=year_max,
                    chunk_type=chunk_type,
                )
                fused_lists.append(
                    engine.reciprocal_rank_fusion(
                        vector_hits, bm25_hits, k=fetch_k, rrf_k=settings.rrf_k
                    )
                )
            else:
                fused_lists.append(
                    engine.vector_search(
                        query_embedding=query_vec,
                        k=fetch_k,
                        doc_id=doc_uuid,
                        section_substring=section,
                        author=author,
                        year_min=year_min,
                        year_max=year_max,
                        min_score=min_score,
                        chunk_type=chunk_type,
                    )
                )

        hits = fused_lists[0]
        for extra in fused_lists[1:]:
            hits = engine.reciprocal_rank_fusion(hits, extra, k=fetch_k, rrf_k=settings.rrf_k)

        if settings.rerank_enabled and hits:
            from lean.retrieval.reranker import rerank as _rerank

            hits = _rerank(
                hits,
                query,
                model=settings.rerank_model,
                top_n=settings.rerank_top_n,
                device=settings.embedding_device,
                revision=settings.rerank_model_revision,
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
            "multi_query": len(queries) > 1,
            "hyde": settings.llm_hyde and llm is not None,
        }
        try:
            analytics.log_query(
                query_text=query,
                k=k,
                filters=filters,
                hit_chunk_ids=[UUID(h.chunk.id) for h in hits],
                hit_scores=[h.score for h in hits],
                latency_ms=latency_ms,
            )
        except Exception:
            logger.warning("failed to log query", exc_info=True)

        return [h.chunk for h in hits]
    finally:
        conn.close()
        logger.info(
            "search q=%r k=%d hits=%d latency=%dms hybrid=%s reranked=%s queries=%d",
            query[:60],
            k,
            len(hits),
            latency_ms,
            settings.hybrid_search_enabled,
            settings.rerank_enabled,
            len(queries),
        )
