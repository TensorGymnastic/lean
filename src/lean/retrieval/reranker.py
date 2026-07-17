"""Cross-encoder re-ranking using sentence-transformers.

Retrieves top-N candidates via vector/BM25 search, then re-sorts them
using a cross-encoder that jointly attends to (query, document) pairs.
This dramatically improves top-3 precision.

Disabled by default. Enable via ``settings.rerank_enabled = True``.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lean.store.search import SearchHit

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_reranker(model_name: str, device: str) -> Any:
    from sentence_transformers import CrossEncoder

    logger.info("loading cross-encoder reranker: %s on %s", model_name, device)
    return CrossEncoder(model_name, device=device)


def rerank(
    hits: list[SearchHit],
    query: str,
    *,
    model: str,
    top_n: int,
    device: str,
) -> list[SearchHit]:
    """Re-sort hits using a cross-encoder model. Returns top_n results."""
    if not hits:
        return []

    encoder = _get_reranker(model, device)
    pairs = [(query, h.chunk.content) for h in hits]
    scores: list[float] = encoder.predict(pairs)

    scored = sorted(zip(hits, scores, strict=True), key=lambda x: x[1], reverse=True)

    from lean.store.search import SearchHit as SH

    result: list[SH] = []
    for hit, score in scored[:top_n]:
        result.append(
            SH(
                chunk=hit.chunk.model_copy(update={"score": float(score)}),
                score=float(score),
            )
        )
    return result
