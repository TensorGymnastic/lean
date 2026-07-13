"""High-level retrieval: embed query → pgvector cosine search + query logging."""

from __future__ import annotations

import logging
import time
from uuid import UUID

from lean.embeddings.liquid_lmf import LiquidLMFEmbedder
from lean.models.schemas import Chunk
from lean.settings import Settings
from lean.store.pgvector import PgVectorStore, SearchHit

logger = logging.getLogger(__name__)

_embedder: LiquidLMFEmbedder | None = None
_store: PgVectorStore | None = None


def _get_embedder() -> LiquidLMFEmbedder:
    global _embedder
    if _embedder is None:
        settings = Settings()
        _embedder = LiquidLMFEmbedder(
            model=settings.embedding_model,
            hf_token=settings.hf_token,
            device=settings.embedding_device,
            dim=settings.embedding_dim,
        )
    return _embedder


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
    """Embed the query and search pgvector with optional metadata filters."""
    embedder = _get_embedder()
    store = _get_store()

    start = time.monotonic()
    query_vec = embedder.embed_query(query)
    hits: list[SearchHit] = store.search(
        query_embedding=query_vec,
        k=k,
        doc_id=UUID(doc_id) if doc_id else None,
        section_substring=section,
        author=author,
        year_min=year_min,
        year_max=year_max,
        min_score=min_score,
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    filters: dict[str, object] = {
        "doc_id": doc_id,
        "section": section,
        "author": author,
        "year_min": year_min,
        "year_max": year_max,
        "min_score": min_score,
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
        "search q=%r k=%d hits=%d latency=%dms",
        query[:60],
        k,
        len(hits),
        latency_ms,
    )
    return [h.chunk for h in hits]
