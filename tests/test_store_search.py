"""Unit tests for store/search.py — RRF fusion and metadata filter builder.

These are the pure-Python algorithms inside ``SearchEngine`` that don't need a
database connection.  ``vector_search`` and ``bm25_search`` are covered by
integration tests (test_store_pgvector.py) since they require pgvector.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID

import pytest

from lean.core.models.schemas import Chunk
from lean.core.store.search import SearchEngine, SearchHit

# --- Helpers ------------------------------------------------------------------


def _hit(chunk_id: str, score: float = 0.5) -> SearchHit:
    """Minimal SearchHit for RRF tests — only id and score matter."""
    return SearchHit(
        chunk=Chunk(
            id=chunk_id,
            document_id="doc-1",
            chunk_index=0,
            section_path="Ch 1",
            token_count=10,
            content=f"content for {chunk_id}",
            score=score,
        ),
        score=score,
    )


@pytest.fixture
def engine() -> SearchEngine:
    """SearchEngine with a mock conn — sufficient for pure-method tests."""
    return SearchEngine(MagicMock())


# --- reciprocal_rank_fusion ---------------------------------------------------


class TestRRFBasic:
    """Core RRF behaviour: score formula, ordering, top-k truncation."""

    def test_empty_both_lists(self, engine: SearchEngine) -> None:
        result = engine.reciprocal_rank_fusion([], [], k=5, rrf_k=60)
        assert result == []

    def test_empty_vector_nonempty_bm25(self, engine: SearchEngine) -> None:
        bm25 = [_hit("a"), _hit("b")]
        result = engine.reciprocal_rank_fusion([], bm25, k=5, rrf_k=60)
        assert [r.chunk.id for r in result] == ["a", "b"]

    def test_empty_bm25_nonempty_vector(self, engine: SearchEngine) -> None:
        vector = [_hit("a"), _hit("b")]
        result = engine.reciprocal_rank_fusion(vector, [], k=5, rrf_k=60)
        assert [r.chunk.id for r in result] == ["a", "b"]

    def test_disjoint_lists_preserve_relative_order(self, engine: SearchEngine) -> None:
        """Non-overlapping lists: each chunk appears once with partial score."""
        vector = [_hit("v1"), _hit("v2")]
        bm25 = [_hit("b1"), _hit("b2")]
        result = engine.reciprocal_rank_fusion(vector, bm25, k=10, rrf_k=60)
        ids = {r.chunk.id for r in result}
        assert ids == {"v1", "v2", "b1", "b2"}
        # v1 (rank 0 in vector) should score higher than b1 (rank 0 in bm25)
        # because v1 also benefits from being first overall — but actually both
        # get the same score since each only appears in one list. Check ordering
        # is stable: vector first, then bm25 at same score.
        assert result[0].chunk.id == "v1"


class TestRRFOverlap:
    """Overlapping lists: a chunk in both lists gets summed (boosted) score."""

    def test_chunk_in_both_lists_gets_higher_score(self, engine: SearchEngine) -> None:
        """A chunk appearing in both lists at rank 0 should rank first."""
        shared = _hit("shared")
        vector = [shared, _hit("v_only")]
        bm25 = [shared, _hit("b_only")]
        result = engine.reciprocal_rank_fusion(vector, bm25, k=5, rrf_k=60)

        # "shared" gets 2/(60+1) + 2/(60+1) ≈ 0.0328
        # others get 1/(60+1) ≈ 0.0164
        assert result[0].chunk.id == "shared"
        assert result[0].score > result[1].score

    def test_deduplication(self, engine: SearchEngine) -> None:
        """A chunk appearing in both lists appears only once in output."""
        shared = _hit("shared")
        vector = [shared]
        bm25 = [shared]
        result = engine.reciprocal_rank_fusion(vector, bm25, k=5, rrf_k=60)
        assert len(result) == 1
        assert result[0].chunk.id == "shared"

    def test_rrf_score_formula(self, engine: SearchEngine) -> None:
        """Verify exact RRF score: Σ 1/(rrf_k + rank + 1) per list."""
        rrf_k = 60
        # "x" at rank 0 in both lists
        vector = [_hit("x")]
        bm25 = [_hit("x")]
        result = engine.reciprocal_rank_fusion(vector, bm25, k=5, rrf_k=rrf_k)
        expected_score = 2.0 * (1.0 / (rrf_k + 0 + 1))
        assert result[0].score == pytest.approx(expected_score)

    def test_rrf_score_formula_rank_1(self, engine: SearchEngine) -> None:
        """Chunk at rank 1 in one list only: score = 1/(rrf_k + 2)."""
        rrf_k = 60
        vector = [_hit("a"), _hit("b")]  # "b" is rank 1
        result = engine.reciprocal_rank_fusion(vector, [], k=5, rrf_k=rrf_k)
        # "b" at rank 1: 1/(60+1+1) = 1/62
        b_hit = [r for r in result if r.chunk.id == "b"][0]
        assert b_hit.score == pytest.approx(1.0 / (rrf_k + 1 + 1))


class TestRRFTruncation:
    """Top-k truncation and edge cases."""

    def test_returns_at_most_k(self, engine: SearchEngine) -> None:
        vector = [_hit(f"v{i}") for i in range(10)]
        bm25 = [_hit(f"b{i}") for i in range(10)]
        result = engine.reciprocal_rank_fusion(vector, bm25, k=3, rrf_k=60)
        assert len(result) == 3

    def test_k_larger_than_total(self, engine: SearchEngine) -> None:
        vector = [_hit("a"), _hit("b")]
        bm25 = [_hit("c")]
        result = engine.reciprocal_rank_fusion(vector, bm25, k=100, rrf_k=60)
        assert len(result) == 3


class TestRRFScoreUpdate:
    """Returned hits carry the RRF score, not the original search score."""

    def test_score_is_rrf_score_not_original(self, engine: SearchEngine) -> None:
        original_score = 0.99
        vector = [_hit("a", score=original_score)]
        result = engine.reciprocal_rank_fusion(vector, [], k=5, rrf_k=60)
        # RRF score is 1/61 ≈ 0.016, not 0.99
        assert result[0].score != pytest.approx(original_score)
        assert result[0].score == pytest.approx(1.0 / 61)

    def test_chunk_score_updated_to_rrf(self, engine: SearchEngine) -> None:
        vector = [_hit("a", score=0.99)]
        result = engine.reciprocal_rank_fusion(vector, [], k=5, rrf_k=60)
        assert result[0].chunk.score == pytest.approx(1.0 / 61)


class TestRRFKParameter:
    """The rrf_k parameter controls score distribution."""

    def test_smaller_rrf_k_amplifies_top_ranks(self, engine: SearchEngine) -> None:
        """With rrf_k=1, rank 0 scores 0.5 vs rank 1 scores 0.33 — wider gap."""
        vector = [_hit("a"), _hit("b")]
        result = engine.reciprocal_rank_fusion(vector, [], k=5, rrf_k=1)
        # a: 1/(1+0+1) = 0.5, b: 1/(1+1+1) = 0.333
        assert result[0].score == pytest.approx(0.5)
        assert result[1].score == pytest.approx(1.0 / 3)


# --- _build_metadata_filters --------------------------------------------------


class TestMetadataFiltersNoFilters:
    """No optional filters: only base conditions (doc_id, section)."""

    def test_no_filters_no_join(self, engine: SearchEngine) -> None:
        join, conditions, params = engine._build_metadata_filters(
            doc_id=None,
            section_substring=None,
            author=None,
            year_min=None,
            year_max=None,
        )
        assert join == ""
        assert len(conditions) == 2  # doc_id + section base conditions
        assert params == [None, None, None, None]


class TestMetadataFiltersDocId:
    """doc_id filter operates on chunks table (no join needed)."""

    def test_doc_id_no_join(self, engine: SearchEngine) -> None:
        doc_uuid = UUID("12345678-1234-1234-1234-123456789012")
        join, conditions, params = engine._build_metadata_filters(
            doc_id=doc_uuid,
            section_substring=None,
            author=None,
            year_min=None,
            year_max=None,
        )
        assert join == ""
        assert doc_uuid in params


class TestMetadataFiltersSection:
    """section_substring filter operates on chunks table (no join needed)."""

    def test_section_no_join(self, engine: SearchEngine) -> None:
        join, conditions, params = engine._build_metadata_filters(
            doc_id=None,
            section_substring="DMAIC",
            author=None,
            year_min=None,
            year_max=None,
        )
        assert join == ""
        assert "DMAIC" in params


class TestMetadataFiltersAuthor:
    """author filter requires documents join."""

    def test_author_adds_join(self, engine: SearchEngine) -> None:
        join, conditions, params = engine._build_metadata_filters(
            doc_id=None,
            section_substring=None,
            author="Smith",
            year_min=None,
            year_max=None,
        )
        assert "join public.documents" in join
        assert len(conditions) == 3
        assert any("d.authors" in c for c in conditions)
        assert "Smith" in params


class TestMetadataFiltersYear:
    """year_min/year_max filters require documents join."""

    def test_year_min_adds_join(self, engine: SearchEngine) -> None:
        join, conditions, params = engine._build_metadata_filters(
            doc_id=None,
            section_substring=None,
            author=None,
            year_min=2020,
            year_max=None,
        )
        assert "join public.documents" in join
        assert any("d.year >=" in c for c in conditions)
        assert 2020 in params

    def test_year_max_adds_join(self, engine: SearchEngine) -> None:
        join, conditions, params = engine._build_metadata_filters(
            doc_id=None,
            section_substring=None,
            author=None,
            year_min=None,
            year_max=2024,
        )
        assert "join public.documents" in join
        assert any("d.year <=" in c for c in conditions)
        assert 2024 in params


class TestMetadataFiltersCombined:
    """All filters together: join + all conditions + correct param count."""

    def test_all_filters(self, engine: SearchEngine) -> None:
        doc_uuid = UUID("00000000-0000-0000-0000-000000000001")
        join, conditions, params = engine._build_metadata_filters(
            doc_id=doc_uuid,
            section_substring="DMAIC",
            author="Smith",
            year_min=2020,
            year_max=2024,
        )
        assert "join public.documents" in join
        # 2 base + 3 filter conditions = 5
        assert len(conditions) == 5
        # doc_id×2 + section×2 + author + year_min + year_max = 7
        assert len(params) == 7

    def test_year_min_and_max_without_author_still_joins(self, engine: SearchEngine) -> None:
        """Join is triggered by ANY documents-table filter, not just author."""
        join, _, _ = engine._build_metadata_filters(
            doc_id=None,
            section_substring=None,
            author=None,
            year_min=2020,
            year_max=2024,
        )
        assert "join public.documents" in join

    def test_only_doc_id_and_section_no_join(self, engine: SearchEngine) -> None:
        """doc_id and section are chunks-table columns — no join needed."""
        join, _, _ = engine._build_metadata_filters(
            doc_id=UUID("00000000-0000-0000-0000-000000000001"),
            section_substring="Chapter",
            author=None,
            year_min=None,
            year_max=None,
        )
        assert join == ""


class TestMetadataFiltersChunkType:
    """chunk_type filter: chunks-table only, no documents join needed."""

    def test_chunk_type_image_no_join(self, engine: SearchEngine) -> None:
        """chunk_type=image adds a WHERE-clause but no documents join."""
        join, conditions, params = engine._build_metadata_filters(
            doc_id=None,
            section_substring=None,
            author=None,
            year_min=None,
            year_max=None,
            chunk_type="image",
        )
        assert join == ""
        assert any("c.chunk_type" in c for c in conditions)
        assert "image" in params

    def test_chunk_type_text_with_author_combines(self, engine: SearchEngine) -> None:
        """chunk_type combines with author filter (which triggers the join)."""
        join, conditions, params = engine._build_metadata_filters(
            doc_id=None,
            section_substring=None,
            author="Smith",
            year_min=None,
            year_max=None,
            chunk_type="text",
        )
        assert "join public.documents" in join
        assert len(conditions) == 4
        assert "text" in params

    def test_chunk_type_none_omits_condition(self, engine: SearchEngine) -> None:
        """None chunk_type means 'both text and image' — no WHERE added."""
        join, conditions, params = engine._build_metadata_filters(
            doc_id=None,
            section_substring=None,
            author=None,
            year_min=None,
            year_max=None,
            chunk_type=None,
        )
        assert len(conditions) == 2
        assert not any("c.chunk_type" in c for c in conditions)
        assert all(p is None for p in params)


class TestRRFRankBoundaries:
    """RRF correctness at rank boundaries."""

    def test_high_rank_chunk_appears_in_top_k(self, engine: SearchEngine) -> None:
        """A chunk at rank 0 in BOTH lists beats all single-list competitors."""
        rrf_k = 60
        bm25 = [_hit("v0")] + [_hit(f"b{i}") for i in range(200)]
        vector = [_hit("v0"), _hit("v1")]
        result = engine.reciprocal_rank_fusion(vector, bm25, k=2, rrf_k=rrf_k)
        assert len(result) == 2
        assert result[0].chunk.id == "v0"
        assert result[0].score == pytest.approx(2.0 / (rrf_k + 1))

    def test_rrf_score_positive_always(self, engine: SearchEngine) -> None:
        """Even the worst-ranked hit has a positive RRF score, never zero."""
        rrf_k = 60
        bm25 = [_hit(f"b{i}") for i in range(1000)]
        vector = []
        result = engine.reciprocal_rank_fusion(vector, bm25, k=1000, rrf_k=rrf_k)
        assert len(result) == 1000
        assert result[-1].score > 0
        assert result[-1].score == pytest.approx(1.0 / (rrf_k + 999 + 1))
