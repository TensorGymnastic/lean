"""Search service: query transform → embed → search → rerank → postprocess.

Orchestrates the retrieval pipeline end-to-end. Supports optional LLM-
powered query transforms (HyDE, multi-query) when an LLM sidecar is
configured. Each call opens its own StoreConnection and closes when done.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from uuid import UUID

from lean.core.config.settings import CoreSettings, get_settings
from lean.core.infrastructure.embedder import Embedder, get_embedder
from lean.core.llm.base import LLMClient, get_llm
from lean.core.models import CHUNK_TYPE_IMAGE, CHUNK_TYPE_TEXT, Chunk
from lean.core.store.analytics import AnalyticsRepo
from lean.core.store.base import StoreConnection
from lean.core.store.search import SearchEngine, SearchHit

logger = logging.getLogger(__name__)

VALID_CHUNK_TYPES = frozenset({CHUNK_TYPE_TEXT, CHUNK_TYPE_IMAGE})


@dataclass(frozen=True)
class SearchRequest:
    """Constant filter parameters shared across all per-query fetches in one search call."""

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
    settings: CoreSettings,
) -> tuple[str, int, str | None]:
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


def _expand_queries(query: str, llm: LLMClient | None, settings: CoreSettings) -> list[str]:
    if not settings.llm_multi_query or not llm:
        return [query]
    from lean.core.retrieval.query_transform import multi_query_transform

    return multi_query_transform(
        query,
        llm,
        num_queries=settings.llm_multi_query_count,
        max_tokens=settings.llm_generate_max_tokens,
        temperature=settings.llm_generate_temperature,
    )


def _compute_fetch_k(k: int, settings: CoreSettings) -> int:
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
    settings: CoreSettings,
) -> list[SearchHit]:
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
    settings: CoreSettings,
) -> list[SearchHit]:
    embed_text = query_text
    if settings.llm_hyde and llm:
        from lean.core.retrieval.query_transform import hyde_transform

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


def _fuse_multi_query(
    engine: SearchEngine,
    per_query_lists: list[list[SearchHit]],
    request: SearchRequest,
    settings: CoreSettings,
) -> list[SearchHit]:
    """Fuse per-query hit lists into one ranked list via RRF."""
    if len(per_query_lists) == 1:
        return per_query_lists[0]
    hits = per_query_lists[0]
    for extra in per_query_lists[1:]:
        hits = engine.reciprocal_rank_fusion(hits, extra, k=request.fetch_k, rrf_k=settings.rrf_k)
    return hits


def _rerank_hits(hits: list[SearchHit], query: str, settings: CoreSettings) -> list[SearchHit]:
    if not settings.rerank_enabled or not hits:
        return hits
    from lean.core.retrieval.reranker import rerank as _rerank

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
    settings: CoreSettings,
) -> list[SearchHit]:
    effective_min = min_score if min_score is not None else settings.min_similarity
    if effective_min > 0:
        from lean.core.retrieval.postprocessors import similarity_filter

        hits = similarity_filter(hits, effective_min)
    from lean.core.retrieval.postprocessors import long_context_reorder

    hits = long_context_reorder(hits)
    return hits[:k]


def _log_query_analytics(
    conn: StoreConnection,
    *,
    query: str,
    k: int,
    hits: list[SearchHit],
    settings: CoreSettings,
    llm: LLMClient | None,
    queries: list[str],
    latency_ms: int,
    filters: dict[str, object],
) -> None:
    try:
        AnalyticsRepo(conn).log_query(
            query_text=query,
            k=k,
            filters=filters,
            hit_chunk_ids=[UUID(h.chunk.id) for h in hits],
            hit_scores=[h.score for h in hits],
            latency_ms=latency_ms,
        )
    except Exception:
        logger.warning("failed to log query", exc_info=True)


def _build_search_request(
    *,
    doc_id: str | None,
    section: str | None,
    author: str | None,
    year_min: int | None,
    year_max: int | None,
    min_score: float | None,
    chunk_type: str | None,
    k: int,
    settings: CoreSettings,
) -> SearchRequest:
    return SearchRequest(
        doc_uuid=UUID(doc_id) if doc_id else None,
        section=section,
        author=author,
        year_min=year_min,
        year_max=year_max,
        min_score=min_score,
        chunk_type=chunk_type,
        fetch_k=_compute_fetch_k(k, settings),
    )


def _build_filters_log(
    *,
    doc_id: str | None,
    section: str | None,
    author: str | None,
    year_min: int | None,
    year_max: int | None,
    min_score: float | None,
    settings: CoreSettings,
    llm: LLMClient | None,
    n_queries: int,
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "section": section,
        "author": author,
        "year_min": year_min,
        "year_max": year_max,
        "min_score": min_score,
        "hybrid": settings.hybrid_search_enabled,
        "reranked": settings.rerank_enabled,
        "multi_query": n_queries > 1,
        "hyde": settings.llm_hyde and llm is not None,
    }


def _execute_search(
    *,
    query: str,
    k: int,
    request: SearchRequest,
    queries: list[str],
    embedder: Embedder,
    llm: LLMClient | None,
    min_score: float | None,
    settings: CoreSettings,
    conn: StoreConnection,
) -> list[SearchHit]:
    """Run the search pipeline on the given open connection."""
    engine = SearchEngine(conn)
    per_query = [_fetch_one_query(engine, embedder, llm, q, request, settings) for q in queries]
    hits = _fuse_multi_query(engine, per_query, request, settings)
    hits = _rerank_hits(hits, query, settings)
    return _postprocess_hits(hits, min_score, k, settings)


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
    embedder = get_embedder()
    llm = get_llm()
    queries = _expand_queries(query, llm, settings)
    request = _build_search_request(
        doc_id=doc_id,
        section=section,
        author=author,
        year_min=year_min,
        year_max=year_max,
        min_score=min_score,
        chunk_type=chunk_type,
        k=k,
        settings=settings,
    )

    with StoreConnection.from_env() as conn:
        start = time.monotonic()
        hits = _execute_search(
            query=query,
            k=k,
            request=request,
            queries=queries,
            embedder=embedder,
            llm=llm,
            min_score=min_score,
            settings=settings,
            conn=conn,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        _log_query_analytics(
            conn,
            query=query,
            k=k,
            hits=hits,
            settings=settings,
            llm=llm,
            queries=queries,
            latency_ms=latency_ms,
            filters=_build_filters_log(
                doc_id=doc_id,
                section=section,
                author=author,
                year_min=year_min,
                year_max=year_max,
                min_score=min_score,
                settings=settings,
                llm=llm,
                n_queries=len(queries),
            ),
        )

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
    return [h.chunk for h in hits]


__all__ = ["SearchRequest", "search"]
