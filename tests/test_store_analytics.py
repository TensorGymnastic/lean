"""Unit tests for store/analytics.py — query logging and corpus stats."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

_VALID_KEY = "x" * 32


def _mock_conn_with_cursor(fetchone_return=None, fetchall_return=None):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchone.return_value = fetchone_return
    cursor.fetchall.return_value = fetchall_return or []
    conn.conn.cursor.return_value = cursor
    return conn, cursor


def test_log_query(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.analytics import AnalyticsRepo

    conn, cursor = _mock_conn_with_cursor()
    repo = AnalyticsRepo(conn)
    repo.log_query(
        query_text="test",
        k=5,
        filters={"hybrid": True},
        hit_chunk_ids=[],
        hit_scores=[],
        latency_ms=10,
    )
    cursor.execute.assert_called_once()
    conn.conn.commit.assert_called_once()


def test_save_eval_run(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.analytics import AnalyticsRepo

    conn, cursor = _mock_conn_with_cursor()
    repo = AnalyticsRepo(conn)
    repo.save_eval_run(
        config={"k": 5},
        hit_rate=0.8,
        mrr=0.7,
        ndcg=0.75,
        recall=0.8,
        mean_latency_ms=50,
        sample_count=50,
        k=5,
    )
    cursor.execute.assert_called_once()
    conn.conn.commit.assert_called_once()


def test_count_documents(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.analytics import AnalyticsRepo

    conn, cursor = _mock_conn_with_cursor(fetchone_return=(10,))
    repo = AnalyticsRepo(conn)
    assert repo.count_documents() == 10


def test_count_documents_no_row(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.analytics import AnalyticsRepo

    conn, cursor = _mock_conn_with_cursor(fetchone_return=None)
    repo = AnalyticsRepo(conn)
    with pytest.raises(RuntimeError, match="count"):
        repo.count_documents()


def test_corpus_stats(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.analytics import AnalyticsRepo

    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    # Simulate sequential fetchone/fetchall calls for corpus_stats
    cursor.fetchone.side_effect = [
        {"count": 10},
        {"count": 2535},
        {"total": 859776},
        {"last": None},
    ]
    cursor.fetchall.return_value = [
        {"extraction_method": "markitdown", "cnt": 6},
        {"extraction_method": "unlimited_ocr", "cnt": 4},
    ]
    conn.conn.cursor.return_value = cursor

    repo = AnalyticsRepo(conn)
    stats = repo.corpus_stats(embedding_dim=1024, embedding_model="test-model")
    assert stats.document_count == 10
    assert stats.chunk_count == 2535
    assert stats.total_tokens == 859776
    assert stats.embedding_dim == 1024
    assert stats.extraction_method_breakdown["markitdown"] == 6
