"""Postprocessors applied to retrieved chunks before returning to the caller.

- similarity_filter: drop chunks below a cosine similarity threshold.
- long_context_reorder: interleave best/worst to combat "lost in the middle".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lean.store.pgvector import SearchHit


def similarity_filter(hits: list[SearchHit], min_score: float) -> list[SearchHit]:
    """Drop hits whose score is below ``min_score``."""
    return [h for h in hits if h.score >= min_score]


def long_context_reorder(hits: list[SearchHit]) -> list[SearchHit]:
    """Reorder so the most relevant hits are at start and end.

    LLMs pay more attention to the beginning and end of the context window
    ("lost in the middle" phenomenon). This interleave places the best hit
    first, the second-best last, the third second, etc.
    """
    if len(hits) <= 1:
        return hits
    sorted_hits = sorted(hits, key=lambda h: h.score, reverse=True)
    front: list[SearchHit] = []
    back: list[SearchHit] = []
    for i, hit in enumerate(sorted_hits):
        if i % 2 == 0:
            front.append(hit)
        else:
            back.append(hit)
    return front + back[::-1]
