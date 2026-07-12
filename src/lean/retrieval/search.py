"""High-level retrieval: embed query → pgvector cosine search."""

from __future__ import annotations

import logging
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
) -> list[Chunk]:
    """Embed the query and search pgvector.

    Args:
        query: Natural-language search query.
        k: Number of top results to return.
        doc_id: Optional document UUID to restrict search to.
        section: Optional case-insensitive substring to filter section_path.

    Returns:
        List of Chunks sorted by cosine similarity (highest first).
        Each chunk's ``score`` is populated.
    """
    embedder = _get_embedder()
    store = _get_store()

    query_vec = embedder.embed_query(query)
    hits: list[SearchHit] = store.search(
        query_embedding=query_vec,
        k=k,
        doc_id=UUID(doc_id) if doc_id else None,
        section_substring=section,
    )
    return [h.chunk for h in hits]
