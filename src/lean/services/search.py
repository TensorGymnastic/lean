"""Search service: query transform → embed → search → rerank → postprocess.

Orchestrates the retrieval pipeline end-to-end. Supports optional LLM-powered
query transforms (HyDE, multi-query) when an LLM sidecar is configured.
Each call opens its own StoreConnection and closes it when done.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from uuid import UUID

from lean.config.settings import Settings, get_settings
from lean.infrastructure.embedder import Embedder, get_embedder
from lean.llm.base import LLMClient, get_llm
from lean.models.schemas import Chunk
from lean.store.analytics import AnalyticsRepo
from lean.store.base import StoreConnection
from lean.store.search import SearchEngine, SearchHit

logger = logging.getLogger(__name__)

VALID_CHUNK_TYPES = frozenset({"text", "image"})


@dataclass(frozen=True)
class SearchRequest:
    """Constant filter parameters shared across all per-query fetches in one search call.

    ``query_text`` varies per paraphrase and is passed alongside the request.
    """

    doc_uuid: UUID | None
    section: str | None
    author: str | None
    year_min: int | None
    year_max: int | None
    min_score: float | None
    chunk_type: str | None
    fetch_k: int


def _validate_search_inputs(
    query: str,
    k: int | None,
    chunk_type: str | None,
    settings: Settings,
) -> tuple[str, int, str | None]:
    """Validate query/k/chunk_type against settings. Returns (query, clamped_k, chunk_type)."""
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
    return query, max(1, min(k, settings.search_max_k)), chunk_type


def _expand_queries(query: str, llm: LLMClient | None, settings: Settings) -> list[str]:
    """Return [query] plus optional LLM paraphrases when multi-query is enabled."""
    if not settings.llm_multi_query or not llm:
        return [query]
    from lean.retrieval.query_transform import multi_query_transform

    return multi_query_transform(
        query,
        llm,
        num_queries=settings.llm_multi_query_count,
        max_tokens=settings.llm_generate_max_tokens,
        temperature=settings.llm_generate_temperature,
    )


def _compute_fetch_k(k: int, settings: Settings) -> int:
    """Wide-candidate fetch_k: max(k × multiplier, floor), then capped."""
    base = (
        max(k * settings.fetch_multiplier, settings.fetch_k_floor)
        if settings.hybrid_search_enabled
        else k
    )
    return min(base, settings.search_fetch_k_cap)


def _vector_fetch(
    engine: SearchEngine,
    query_vec: list[float],
    request: SearchRequest,
) -> list[SearchHit]:
    """Vector-only pgvector search with all filters from request applied."""
    return engine.vector_search(
        query_embedding=query_vec,
        k=request.fetch_k,
        doc_id=request.doc_uuid,
        section_substring=request.section,
        author=request.author,
        year_min=request.year_min,
        year_max=request.year_max,
        min_score=request.min_score,
        chunk_type=request.chunk_type,
    )


def _hybrid_fetch(
    engine: SearchEngine,
    query_vec: list[float],
    query_text: str,
    request: SearchRequest,
    settings: Settings,
) -> list[SearchHit]:
    """Hybrid vector + BM25 search fused via Reciprocal Rank Fusion."""
    vector_hits = _vector_fetch(engine, query_vec, request)
    bm25_hits = engine.bm25_search(
        query_text=query_text,
        k=request.fetch_k,
        doc_id=request.doc_uuid,
        section_substring=request.section,
        author=request.author,
        year_min=request.year_min,
        year_max=request.year_max,
        chunk_type=request.chunk_type,
    )
    return engine.reciprocal_rank_fusion(
        vector_hits, bm25_hits, k=request.fetch_k, rrf_k=settings.rrf_k
    )


def _fetch_one_query(
    engine: SearchEngine,
    embedder: Embedder,
    llm: LLMClient | None,
    query_text: str,
    request: SearchRequest,
    settings: Settings,
) -> list[SearchHit]:
    """Single-query fetch: optional HyDE → embed → vector ± BM25 → RRF."""
    embed_text = query_text
    if settings.llm_hyde and llm:
        from lean.retrieval.query_transform import hyde_transform

        embed_text = hyde_transform(
            query_text,
            llm,
            max_tokens=settings.llm_generate_max_tokens,
            temperature=settings.llm_generate_temperature,
        )

    query_vec = embedder.embed_query(embed_text)

    if not settings.hybrid_search_enabled:
        return _vector_fetch(engine, query_vec, request)
    return _hybrid_fetch(engine, query_vec, query_text, request, settings)


def _rerank_hits(hits: list[SearchHit], query: str, settings: Settings) -> list[SearchHit]:
    """Cross-encoder rerank. No-op when disabled or hits is empty."""
    if not settings.rerank_enabled or not hits:
        return hits
    from lean.retrieval.reranker import rerank as _rerank

    return _rerank(
        hits,
        query,
        model=settings.rerank_model,
        top_n=settings.rerank_top_n,
        device=settings.embedding_device,
        revision=settings.rerank_model_revision,
    )


def _postprocess_hits(
    hits: list[SearchHit],
    min_score: float | None,
    k: int,
    settings: Settings,
) -> list[SearchHit]:
    """Similarity filter → long-context reorder → truncate to k."""
    effective_min = min_score if min_score is not None else settings.min_similarity
    if effective_min > 0:
        from lean.retrieval.postprocessors import similarity_filter

        hits = similarity_filter(hits, effective_min)
    from lean.retrieval.postprocessors import long_context_reorder

    hits = long_context_reorder(hits)
    return hits[:k]


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
    query, k, chunk_type = _validate_search_inputs(query, k, chunk_type, settings)

    start = time.monotonic()
    embedder = get_embedder()
    llm = get_llm()
    doc_uuid = UUID(doc_id) if doc_id else None
    queries = _expand_queries(query, llm, settings)

    conn = StoreConnection.from_env()
    hits: list[SearchHit] = []
    latency_ms = -1
    try:
        engine = SearchEngine(conn)
        analytics = AnalyticsRepo(conn)
        request = SearchRequest(
            doc_uuid=doc_uuid,
            section=section,
            author=author,
            year_min=year_min,
            year_max=year_max,
            min_score=min_score,
            chunk_type=chunk_type,
            fetch_k=_compute_fetch_k(k, settings),
        )

        fused_lists = [
            _fetch_one_query(engine, embedder, llm, q, request, settings) for q in queries
        ]

        hits = fused_lists[0]
        for extra in fused_lists[1:]:
            hits = engine.reciprocal_rank_fusion(
                hits, extra, k=request.fetch_k, rrf_k=settings.rrf_k
            )

        hits = _rerank_hits(hits, query, settings)
        hits = _postprocess_hits(hits, min_score, k, settings)

        latency_ms = int((time.monotonic() - start) * 1000)
        filters_log: dict[str, object] = {
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
                filters=filters_log,
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
