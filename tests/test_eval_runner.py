"""Unit tests for eval/runner.py — retrieval metrics computation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

_VALID_KEY = "x" * 32


@pytest.fixture
def monkeypatch_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)


def test_dcg_at_k_found_at_rank_1():
    from lean.eval.runner import _dcg_at_k

    assert _dcg_at_k(1, 5) == 1.0


def test_dcg_at_k_found_at_rank_2():
    from lean.eval.runner import _dcg_at_k

    result = _dcg_at_k(2, 5)
    assert 0.0 < result < 1.0


def test_dcg_at_k_not_found():
    from lean.eval.runner import _dcg_at_k

    assert _dcg_at_k(None, 5) == 0.0


def test_dcg_at_k_beyond_k():
    from lean.eval.runner import _dcg_at_k

    assert _dcg_at_k(6, 5) == 0.0


def test_evaluate_perfect_hit_rate(monkeypatch_env):
    from lean.eval.runner import EvalSample, evaluate

    samples = [
        EvalSample(query="test query", expected_chunk_id="chunk-1"),
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 1024

    from lean.models.schemas import Chunk
    from lean.store.search import SearchHit

    fake_hit = SearchHit(
        chunk=Chunk(
            id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            section_path="Ch 1",
            token_count=50,
            content="content",
        ),
        score=0.95,
    )
    mock_engine = MagicMock()
    mock_engine.vector_search.return_value = [fake_hit]

    with (
        patch("lean.eval.runner.get_embedder", return_value=mock_embedder),
        patch("lean.eval.runner.SearchEngine", return_value=mock_engine),
    ):
        result = evaluate(MagicMock(), samples, k=5)

    assert result.hit_rate == 1.0
    assert result.mrr == 1.0
    assert result.ndcg == 1.0
    assert result.recall == 1.0
    assert result.sample_count == 1
    assert result.k == 5


def test_evaluate_zero_hit_rate(monkeypatch_env):
    from lean.eval.runner import EvalSample, evaluate

    samples = [
        EvalSample(query="test query", expected_chunk_id="chunk-missing"),
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 1024

    from lean.models.schemas import Chunk
    from lean.store.search import SearchHit

    wrong_hit = SearchHit(
        chunk=Chunk(
            id="chunk-other",
            document_id="doc-1",
            chunk_index=0,
            section_path="Ch 1",
            token_count=50,
            content="content",
        ),
        score=0.5,
    )
    mock_engine = MagicMock()
    mock_engine.vector_search.return_value = [wrong_hit]

    with (
        patch("lean.eval.runner.get_embedder", return_value=mock_embedder),
        patch("lean.eval.runner.SearchEngine", return_value=mock_engine),
    ):
        result = evaluate(MagicMock(), samples, k=5)

    assert result.hit_rate == 0.0
    assert result.mrr == 0.0
    assert result.ndcg == 0.0
    assert result.recall == 0.0


def test_evaluate_empty_samples(monkeypatch_env):
    from lean.eval.runner import evaluate

    with (
        patch("lean.eval.runner.get_embedder"),
        patch("lean.eval.runner.SearchEngine"),
    ):
        result = evaluate(MagicMock(), [], k=5)

    assert result.sample_count == 0
    assert result.hit_rate == 0.0
