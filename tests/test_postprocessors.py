"""Unit tests for retrieval/postprocessors.py."""

from __future__ import annotations

from lean.models.schemas import Chunk
from lean.store.search import SearchHit


def _hit(score: float, content: str = "x") -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=f"chunk-{score}",
            document_id="doc-1",
            chunk_index=0,
            section_path="Ch 1",
            token_count=10,
            content=content,
            score=score,
        ),
        score=score,
    )


def test_similarity_filter_removes_below_threshold():
    from lean.retrieval.postprocessors import similarity_filter

    hits = [_hit(0.9), _hit(0.5), _hit(0.3)]
    result = similarity_filter(hits, min_score=0.5)
    assert len(result) == 2
    assert all(h.score >= 0.5 for h in result)


def test_similarity_filter_keeps_all_when_threshold_low():
    from lean.retrieval.postprocessors import similarity_filter

    hits = [_hit(0.1), _hit(0.2)]
    result = similarity_filter(hits, min_score=0.0)
    assert len(result) == 2


def test_similarity_filter_empty_list():
    from lean.retrieval.postprocessors import similarity_filter

    assert similarity_filter([], min_score=0.5) == []


def test_long_context_reorder_single_hit():
    from lean.retrieval.postprocessors import long_context_reorder

    hits = [_hit(0.9)]
    result = long_context_reorder(hits)
    assert len(result) == 1


def test_long_context_reorder_empty():
    from lean.retrieval.postprocessors import long_context_reorder

    assert long_context_reorder([]) == []


def test_long_context_reorder_interleaves():
    from lean.retrieval.postprocessors import long_context_reorder

    hits = [_hit(0.1), _hit(0.5), _hit(0.3), _hit(0.9)]
    result = long_context_reorder(hits)
    assert len(result) == 4
    assert result[0].score == 0.9
    assert result[-1].score == 0.5


def test_long_context_reorder_two_hits():
    from lean.retrieval.postprocessors import long_context_reorder

    hits = [_hit(0.3), _hit(0.9)]
    result = long_context_reorder(hits)
    assert result[0].score == 0.9
    assert result[1].score == 0.3
