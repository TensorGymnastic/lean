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


def _mock_store_with_rows(rows: list[dict]) -> MagicMock:
    """Build a mock StoreConnection whose cursor returns the given rows from fetchall."""
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = rows
    mock_store = MagicMock()
    mock_store.conn.cursor.return_value.__enter__.return_value = mock_cur
    return mock_store


def _make_eligible_rows(n: int) -> list[dict]:
    """Generate n eligible chunk rows (heading > 5 chars)."""
    return [
        {"id": f"chunk-{i:03d}", "heading_text": f"Section Heading {i}", "content": f"content {i}"}
        for i in range(n)
    ]


def test_build_eval_dataset_returns_exactly_sample_size(monkeypatch_env):
    """build_eval_dataset must return exactly sample_size samples, not all eligible rows."""
    from lean.eval.runner import build_eval_dataset

    rows = _make_eligible_rows(20)
    store = _mock_store_with_rows(rows)

    samples = build_eval_dataset(store, sample_size=5, seed=42)

    assert len(samples) == 5


def test_build_eval_dataset_deterministic_with_same_seed(monkeypatch_env):
    """Two calls with the same seed must return identical samples (same chunk IDs, same order)."""
    from lean.eval.runner import build_eval_dataset

    rows = _make_eligible_rows(50)
    store = _mock_store_with_rows(rows)

    samples_a = build_eval_dataset(store, sample_size=10, seed=42)
    samples_b = build_eval_dataset(store, sample_size=10, seed=42)

    ids_a = [s.expected_chunk_id for s in samples_a]
    ids_b = [s.expected_chunk_id for s in samples_b]
    assert ids_a == ids_b


def test_build_eval_dataset_different_seed_different_samples(monkeypatch_env):
    """Different seeds should (almost certainly) produce different sample sets."""
    from lean.eval.runner import build_eval_dataset

    rows = _make_eligible_rows(50)
    store = _mock_store_with_rows(rows)

    samples_a = build_eval_dataset(store, sample_size=10, seed=42)
    samples_b = build_eval_dataset(store, sample_size=10, seed=999)

    ids_a = {s.expected_chunk_id for s in samples_a}
    ids_b = {s.expected_chunk_id for s in samples_b}
    assert ids_a != ids_b


def test_build_eval_dataset_clamps_when_fewer_eligible_than_sample_size(monkeypatch_env):
    """If fewer eligible chunks than sample_size, return all eligible (no crash)."""
    from lean.eval.runner import build_eval_dataset

    rows = _make_eligible_rows(3)
    store = _mock_store_with_rows(rows)

    samples = build_eval_dataset(store, sample_size=50, seed=42)

    assert len(samples) == 3
