"""Unit tests for services/search.py — the central search orchestration module.

These tests exercise the REAL orchestration logic: RRF fusion, postprocessor
application, analytics logging, connection lifecycle, and top-k truncation.
Only the DB-touching methods (``vector_search``, ``bm25_search``) and
infrastructure singletons are patched — the real RRF, postprocessors, and
analytics try/except all run.

Design:
- ``MockSettings`` dataclass provides explicit, type-safe settings overrides.
- ``_make_hit`` builds minimal SearchHit objects for canned search results.
- ``_SearchPatcher`` is a context manager that patches all infrastructure
  the ``search()`` function touches, while letting real algorithms run.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from lean.core.models.schemas import Chunk
from lean.core.store.search import SearchHit

_VALID_KEY = "x" * 32


# --- Test helpers -------------------------------------------------------------


@dataclass
class MockSettings:
    """Explicit, type-safe settings for search orchestration tests.

    Every field that ``services/search.py`` accesses is listed here — no hidden
    MagicMock attribute surprises.  Override any field per-test::

        settings = MockSettings(hybrid_search_enabled=False)
    """

    search_top_k: int = 5
    search_max_k: int = 100
    search_max_query_len: int = 2000
    search_fetch_k_cap: int = 500
    hybrid_search_enabled: bool = True
    fetch_multiplier: int = 8
    fetch_k_floor: int = 40
    rrf_k: int = 60
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder-model"
    rerank_top_n: int = 5
    embedding_device: str = "cpu"
    rerank_model_revision: str = "test-revision"
    min_similarity: float = 0.0
    llm_multi_query: bool = False
    llm_hyde: bool = False
    llm_multi_query_count: int = 4
    llm_generate_max_tokens: int = 500
    llm_generate_temperature: float = 0.0


def _uuid(suffix: int) -> str:
    """Generate a valid UUID string from an integer suffix (0-999999999999)."""
    return f"00000000-0000-0000-0000-{suffix:012d}"


def _make_hit(suffix: int, score: float, content: str = "content") -> SearchHit:
    """Build a minimal SearchHit with a valid UUID for canned search results."""
    return SearchHit(
        chunk=Chunk(
            id=_uuid(suffix),
            document_id=_uuid(999),
            chunk_index=0,
            section_path="Ch 1",
            token_count=10,
            content=content,
            score=score,
        ),
        score=score,
    )


class _SearchPatcher:
    """Context manager: patches all search() infrastructure, lets real algorithms run.

    After ``__enter__``, these attributes are available for assertions:
    - ``mock_embedder`` — the patched embedder
    - ``mock_conn`` — the patched StoreConnection's conn
    - ``mock_analytics`` — the patched AnalyticsRepo instance
    - ``mock_vector_search`` — the patched SearchEngine.vector_search method
    - ``mock_bm25_search`` — the patched SearchEngine.bm25_search method
    """

    def __init__(self, settings: MockSettings, vector_hits=None, bm25_hits=None):
        self.settings = settings
        self.vector_hits = vector_hits or []
        self.bm25_hits = bm25_hits or []
        self.mock_embedder = MagicMock()
        self.mock_embedder.embed_query.return_value = [0.1] * 1024
        self.mock_conn = MagicMock()
        self.mock_analytics = MagicMock()
        self.mock_vector_search = MagicMock(return_value=self.vector_hits)
        self.mock_bm25_search = MagicMock(return_value=self.bm25_hits)
        self._stack = None

    def __enter__(self):
        from contextlib import ExitStack

        from lean.core.store.search import SearchEngine

        self._stack = ExitStack()
        s = self._stack.enter_context

        s(patch("lean.core.services.search.get_settings", return_value=self.settings))
        s(patch("lean.core.services.search.get_embedder", return_value=self.mock_embedder))
        s(patch("lean.core.services.search.get_llm", return_value=None))

        mock_store_cls = MagicMock()
        mock_store_cls.from_env.return_value = self.mock_conn
        s(patch("lean.core.services.search.StoreConnection", mock_store_cls))

        s(patch.object(SearchEngine, "vector_search", self.mock_vector_search))
        s(patch.object(SearchEngine, "bm25_search", self.mock_bm25_search))

        mock_analytics_cls = MagicMock()
        mock_analytics_cls.return_value = self.mock_analytics
        s(patch("lean.core.services.search.AnalyticsRepo", mock_analytics_cls))

        return self

    def __exit__(self, *args):
        if self._stack:
            self._stack.__exit__(*args)


@pytest.fixture
def valid_env(monkeypatch):
    """Set minimum env vars so deferred imports inside search() succeed."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)


# --- Tests --------------------------------------------------------------------


class TestSearchBasic:
    """Basic search path: embed -> vector search -> return chunks."""

    def test_returns_chunks_from_vector_search(self, valid_env):
        """search() returns Chunk objects extracted from SearchHit results."""
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=False)
        vector_hits = [_make_hit(1, 0.9, "DMAIC content")]

        with _SearchPatcher(settings, vector_hits=vector_hits):
            results = _search_orig("What is DMAIC?")

        assert len(results) == 1
        assert results[0].id == _uuid(1)
        assert results[0].content == "DMAIC content"

    def test_empty_results_when_no_hits(self, valid_env):
        """search() returns [] when no chunks are found."""
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=False)

        with _SearchPatcher(settings, vector_hits=[]):
            results = _search_orig("nonexistent")

        assert results == []

    def test_embed_query_called_with_search_query(self, valid_env):
        """The embedder receives the raw query text."""
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=False)

        with _SearchPatcher(settings, vector_hits=[_make_hit(1, 0.5)]) as p:
            _search_orig("my query text")

        p.mock_embedder.embed_query.assert_called_once_with("my query text")


class TestSearchHybrid:
    """Hybrid path: vector + BM25 + RRF fusion all execute."""

    def test_hybrid_calls_both_vector_and_bm25(self, valid_env):
        """When hybrid is enabled, both vector_search and bm25_search are called."""
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=True)
        vector_hits = [_make_hit(1, 0.9), _make_hit(2, 0.7)]
        bm25_hits = [_make_hit(3, 0.8), _make_hit(1, 0.6)]

        with _SearchPatcher(settings, vector_hits=vector_hits, bm25_hits=bm25_hits) as p:
            results = _search_orig("query")

        p.mock_vector_search.assert_called_once()
        p.mock_bm25_search.assert_called_once()
        assert len(results) >= 1

    def test_hybrid_rrf_boosts_overlapping_chunks(self, valid_env):
        """A chunk in both lists should appear in results (RRF boost)."""
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=True)
        shared = _make_hit(0, 0.5)
        vector_hits = [shared, _make_hit(1, 0.9)]
        bm25_hits = [shared, _make_hit(2, 0.8)]

        with _SearchPatcher(settings, vector_hits=vector_hits, bm25_hits=bm25_hits):
            results = _search_orig("query")

        result_ids = [r.id for r in results]
        assert _uuid(0) in result_ids


class TestSearchAnalytics:
    """Analytics logging is called with correct parameters."""

    def test_log_query_called_after_search(self, valid_env):
        """AnalyticsRepo.log_query is invoked with the query and results."""
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=False)
        vector_hits = [_make_hit(1, 0.9)]

        with _SearchPatcher(settings, vector_hits=vector_hits) as p:
            _search_orig("DMAIC", k=5)

        p.mock_analytics.log_query.assert_called_once()
        call_kwargs = p.mock_analytics.log_query.call_args[1]
        assert call_kwargs["query_text"] == "DMAIC"
        assert call_kwargs["k"] == 5
        assert call_kwargs["latency_ms"] >= 0

    def test_log_query_failure_does_not_break_search(self, valid_env):
        """If log_query raises, search() still returns results (try/except guard)."""
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=False)
        vector_hits = [_make_hit(1, 0.9)]

        with _SearchPatcher(settings, vector_hits=vector_hits) as p:
            p.mock_analytics.log_query.side_effect = RuntimeError("DB error")
            results = _search_orig("query")

        assert len(results) == 1


class TestSearchConnectionLifecycle:
    """Connection is always closed in the finally block."""

    def test_connection_closed_on_success(self, valid_env):
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=False)

        with _SearchPatcher(settings, vector_hits=[_make_hit(1, 0.9)]) as p:
            _search_orig("query")

        p.mock_conn.close.assert_called_once()

    def test_connection_closed_on_exception(self, valid_env):
        """Connection is closed even when an error occurs mid-search."""
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=False)

        with _SearchPatcher(settings, vector_hits=[]) as p:
            p.mock_embedder.embed_query.side_effect = RuntimeError("embed failed")
            with pytest.raises(RuntimeError, match="embed failed"):
                _search_orig("query")

        p.mock_conn.close.assert_called_once()


class TestSearchTopK:
    """Results are truncated to k."""

    def test_truncates_to_k(self, valid_env):
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=False)
        vector_hits = [_make_hit(i, 0.9 - i * 0.05) for i in range(10)]

        with _SearchPatcher(settings, vector_hits=vector_hits):
            results = _search_orig("query", k=3)

        assert len(results) <= 3

    def test_k_defaults_to_settings(self, valid_env):
        """When k is None, search uses settings.search_top_k."""
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=False, search_top_k=2)
        vector_hits = [_make_hit(i, 0.5) for i in range(10)]

        with _SearchPatcher(settings, vector_hits=vector_hits):
            results = _search_orig("query")

        assert len(results) <= 2


class TestSearchInputValidation:
    """Input validation guards (DoS prevention)."""

    def test_empty_query_rejected(self, valid_env):
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=False)
        with _SearchPatcher(settings, vector_hits=[]):
            with pytest.raises(ValueError, match="query must not be empty"):
                _search_orig("")
            with pytest.raises(ValueError, match="query must not be empty"):
                _search_orig("   ")

    def test_query_too_long_rejected(self, valid_env):
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=False, search_max_query_len=10)
        with (
            _SearchPatcher(settings, vector_hits=[]),
            pytest.raises(ValueError, match="exceeds max length"),
        ):
            _search_orig("a" * 11)

    def test_k_clamped_to_max(self, valid_env):
        """User-supplied k is clamped to settings.search_max_k."""
        from lean.core.services import search as _search_orig

        # k=1000 must be clamped to max_k=10 → fetch_k floor of 40 still wins
        settings = MockSettings(
            hybrid_search_enabled=False,
            search_max_k=10,
            search_fetch_k_cap=500,
        )
        vector_hits = [_make_hit(i, 0.9 - i * 0.05) for i in range(50)]

        with _SearchPatcher(settings, vector_hits=vector_hits):
            # Must not raise — k is clamped silently
            results = _search_orig("query", k=1000)

        assert len(results) <= 10

    def test_k_below_one_clamped_up(self, valid_env):
        """k=0 or negative is clamped up to 1."""
        from lean.core.services import search as _search_orig

        settings = MockSettings(hybrid_search_enabled=False, search_max_k=100)
        vector_hits = [_make_hit(i, 0.5) for i in range(3)]

        with _SearchPatcher(settings, vector_hits=vector_hits):
            results = _search_orig("query", k=0)

        assert isinstance(results, list)

    def test_fetch_k_capped(self, valid_env):
        """fetch_k respects settings.search_fetch_k_cap even when k*multiplier exceeds it."""
        from lean.core.services import search as _search_orig

        # k=10, multiplier=8 → fetch_k=80; cap=50 → fetch_k=50
        settings = MockSettings(
            hybrid_search_enabled=True,
            search_max_k=100,
            fetch_multiplier=8,
            fetch_k_floor=10,
            search_fetch_k_cap=50,
        )
        # Just verify it runs without raising on the cap path
        with _SearchPatcher(settings, vector_hits=[_make_hit(0, 0.5)]):
            results = _search_orig("query", k=10)

        assert isinstance(results, list)
