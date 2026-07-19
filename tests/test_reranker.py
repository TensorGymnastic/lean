"""Unit tests for retrieval/reranker.py — cross-encoder reranking algorithm.

Patches ``_get_reranker`` (not ``CrossEncoder``) so tests verify the
sort / truncate / score-update logic without loading model weights.

All tests compose the ``make_chunk`` and ``make_search_hit`` factory fixtures
from ``conftest.py`` — no hardcoded UUIDs or inline Chunk construction.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from lean.retrieval.reranker import rerank
from lean.store.search import SearchHit


@pytest.fixture
def mock_encoder() -> MagicMock:
    """A mock cross-encoder with a ``.predict(pairs) -> list[float]`` interface."""
    return MagicMock()


@pytest.fixture
def make_hits(make_chunk, make_search_hit):
    """Factory: build N SearchHit objects with sequential IDs and given scores."""

    def _make(scores: list[float]) -> list[SearchHit]:
        return [
            make_search_hit(
                chunk=make_chunk(id=f"c{i}", content=f"content-{i}", score=s),
                score=s,
            )
            for i, s in enumerate(scores)
        ]

    return _make


class TestRerankEmpty:
    """Edge case: empty input must not invoke the model."""

    def test_empty_returns_empty_list(self) -> None:
        result = rerank([], query="test", model="m", top_n=5, device="cpu")
        assert result == []

    def test_empty_does_not_load_model(self) -> None:
        """No _get_reranker call when hits list is empty."""
        with patch("lean.retrieval.reranker._get_reranker") as mock_get:
            rerank([], query="test", model="m", top_n=5, device="cpu")
            mock_get.assert_not_called()


class TestRerankSorting:
    """Core behaviour: hits are re-sorted by cross-encoder predicted score."""

    def test_sorts_descending_by_predicted_score(self, mock_encoder, make_hits) -> None:
        """Input order is irrelevant; output is sorted by encoder score desc."""
        hits = make_hits([0.9, 0.1, 0.5])
        # Encoder reverses the ranking: last hit scores highest
        mock_encoder.predict.return_value = [0.3, 0.95, 0.6]

        with patch("lean.retrieval.reranker._get_reranker", return_value=mock_encoder):
            result = rerank(hits, "query", model="m", top_n=3, device="cpu")

        assert [r.chunk.id for r in result] == ["c1", "c2", "c0"]
        assert result[0].score == pytest.approx(0.95)
        assert result[1].score == pytest.approx(0.6)
        assert result[2].score == pytest.approx(0.3)

    def test_single_hit_passes_through(self, mock_encoder, make_hits) -> None:
        hits = make_hits([0.42])
        mock_encoder.predict.return_value = [0.88]

        with patch("lean.retrieval.reranker._get_reranker", return_value=mock_encoder):
            result = rerank(hits, "query", model="m", top_n=5, device="cpu")

        assert len(result) == 1
        assert result[0].chunk.id == "c0"
        assert result[0].score == pytest.approx(0.88)


class TestRerankTopN:
    """Truncation: output length never exceeds ``top_n``."""

    def test_truncates_to_top_n(self, mock_encoder, make_hits) -> None:
        hits = make_hits([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_encoder.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]

        with patch("lean.retrieval.reranker._get_reranker", return_value=mock_encoder):
            result = rerank(hits, "query", model="m", top_n=2, device="cpu")

        assert len(result) == 2

    def test_top_n_larger_than_hits_returns_all(self, mock_encoder, make_hits) -> None:
        hits = make_hits([0.3, 0.6])
        mock_encoder.predict.return_value = [0.4, 0.9]

        with patch("lean.retrieval.reranker._get_reranker", return_value=mock_encoder):
            result = rerank(hits, "query", model="m", top_n=10, device="cpu")

        assert len(result) == 2


class TestRerankScoreUpdate:
    """The returned SearchHit objects carry the cross-encoder score, not the original."""

    def test_chunk_score_updated_to_reranker_score(self, mock_encoder, make_hits) -> None:
        original_score = 0.99
        hits = make_hits([original_score])
        mock_encoder.predict.return_value = [0.111]

        with patch("lean.retrieval.reranker._get_reranker", return_value=mock_encoder):
            result = rerank(hits, "query", model="m", top_n=5, device="cpu")

        assert result[0].chunk.score == pytest.approx(0.111)
        assert result[0].score == pytest.approx(0.111)

    def test_original_hits_are_not_mutated(self, mock_encoder, make_hits) -> None:
        """rerank creates new SearchHit/Chunk objects via model_copy; originals are safe."""
        hits = make_hits([0.99])
        mock_encoder.predict.return_value = [0.01]

        with patch("lean.retrieval.reranker._get_reranker", return_value=mock_encoder):
            result = rerank(hits, "query", model="m", top_n=5, device="cpu")

        assert hits[0].chunk.score == pytest.approx(0.99)  # unchanged
        assert result[0].chunk.score == pytest.approx(0.01)  # new object


class TestRerankEncoderInteraction:
    """Verify (query, content) pairs are passed correctly to the encoder."""

    def test_predict_receives_query_content_pairs(
        self, mock_encoder, make_chunk, make_search_hit
    ) -> None:
        hits = [
            make_search_hit(chunk=make_chunk(id="c0", content="alpha text"), score=0.5),
            make_search_hit(chunk=make_chunk(id="c1", content="beta text"), score=0.3),
        ]
        mock_encoder.predict.return_value = [0.8, 0.2]

        with patch("lean.retrieval.reranker._get_reranker", return_value=mock_encoder):
            rerank(hits, "my query", model="m", top_n=2, device="cpu")

        pairs = mock_encoder.predict.call_args[0][0]
        assert pairs == [("my query", "alpha text"), ("my query", "beta text")]

    def test_revision_forwarded_to_get_reranker(self, mock_encoder, make_hits) -> None:
        hits = make_hits([0.5])
        mock_encoder.predict.return_value = [0.9]

        with (
            patch("lean.retrieval.reranker._get_reranker", return_value=mock_encoder) as mock_get,
        ):
            rerank(
                hits, "query", model="cross-encoder-model", top_n=5, device="cpu", revision="abc123"
            )

        mock_get.assert_called_once_with("cross-encoder-model", "cpu", "abc123")

    def test_revision_defaults_to_empty_string(self, mock_encoder, make_hits) -> None:
        hits = make_hits([0.5])
        mock_encoder.predict.return_value = [0.9]

        with patch("lean.retrieval.reranker._get_reranker", return_value=mock_encoder) as mock_get:
            rerank(hits, "query", model="m", top_n=5, device="cpu")

        # Third positional arg (revision) defaults to ""
        assert mock_get.call_args[0][2] == ""
