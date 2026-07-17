"""Shared test infrastructure for the lean test suite.

Design principles:
- Autouse fixture ``_clear_lru_singletons`` prevents cross-test state leakage
  from ``@lru_cache`` singletons (``get_settings``, ``get_embedder``,
  ``get_llm``, ``_get_reranker``).
- Factory fixtures (``make_chunk``, ``make_doc``, ``make_search_hit``) produce
  composable test data with sensible defaults — override any field via kwargs.
- Mock DB fixtures (``mock_cursor``, ``mock_conn``) centralise the
  context-manager-aware psycopg mock pattern used across store-layer tests.
- ``pdf_path`` fixture eliminates the ``_write_minimal_pdf`` duplication.

Tests that need Settings to load request ``monkeypatch_env``; tests that don't
are unaffected.  Existing per-file ``monkeypatch_env`` fixtures shadow this one
until the file is refactored to remove the duplicate.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# --- Shared constants ---------------------------------------------------------

#: A valid API key: 32 chars, not the blacklist value "change-me".
VALID_API_KEY = "x" * 32

#: Embedding dimension for LiquidAI/LFM2.5-Embedding-350M.
EMBED_DIM = 1024

#: Fixed timestamp used across factory defaults — deterministic and timezone-naive.
FIXED_DATE = datetime(2026, 7, 12)

#: Zero-UUIDs for deterministic test data.
UUID_ZERO = "00000000-0000-0000-0000-000000000000"
UUID_ONE = "00000000-0000-0000-0000-000000000001"
UUID_TWO = "00000000-0000-0000-0000-000000000002"


# --- Autouse: singleton cache cleanup ----------------------------------------


@pytest.fixture(autouse=True)
def _clear_lru_singletons() -> Generator[None, None, None]:
    """Clear all ``@lru_cache`` singletons after each test.

    Prevents state leakage when a test fails mid-assertion before reaching an
    inline ``cache_clear()`` call.  Runs as teardown (after ``yield``) so it
    executes even if the test raises.
    """
    yield
    # Import inside the fixture body so module-level import errors in optional
    # deps (torch, sentence-transformers) don't break collection.
    from lean.config.settings import get_settings

    get_settings.cache_clear()

    from lean.infrastructure.embedder import get_embedder

    get_embedder.cache_clear()

    from lean.llm.base import get_llm

    get_llm.cache_clear()

    from lean.retrieval.reranker import _get_reranker

    _get_reranker.cache_clear()


# --- Environment setup --------------------------------------------------------


@pytest.fixture
def monkeypatch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set minimum env vars for ``Settings`` to load without external services."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", VALID_API_KEY)


# --- Factory fixtures ---------------------------------------------------------


@pytest.fixture
def make_chunk() -> Callable[..., object]:
    """Factory: build a :class:`Chunk` with overridable defaults.

    Usage::

        chunk = make_chunk(content="custom text", score=0.9)
        chunk2 = make_chunk(id=UUID_TWO, chunk_index=1)
    """

    def _make(
        *,
        id: str = UUID_ONE,
        document_id: str = UUID_TWO,
        chunk_index: int = 0,
        section_path: str = "Ch 1",
        heading_text: str | None = "DMAIC",
        page_start: int | None = 1,
        page_end: int | None = 2,
        token_count: int = 100,
        content: str = "DMAIC is a structured methodology.",
        score: float | None = None,
    ) -> object:
        from lean.models.schemas import Chunk

        return Chunk(
            id=id,
            document_id=document_id,
            chunk_index=chunk_index,
            section_path=section_path,
            heading_text=heading_text,
            page_start=page_start,
            page_end=page_end,
            token_count=token_count,
            content=content,
            score=score,
        )

    return _make


@pytest.fixture
def make_doc() -> Callable[..., object]:
    """Factory: build a :class:`DocumentSummary` with overridable defaults."""

    def _make(
        *,
        id: str = UUID_ONE,
        source_path: str = "data/test.pdf",
        title: str | None = "Test Document",
        authors: list[str] | None = None,
        page_count: int | None = 100,
        extraction_method: str = "markitdown",
        chunk_count: int = 10,
        ingested_at: datetime = FIXED_DATE,
    ) -> object:
        from lean.models.schemas import DocumentSummary, ExtractionMethod

        return DocumentSummary(
            id=id,
            source_path=source_path,
            title=title,
            authors=authors or [],
            page_count=page_count,
            extraction_method=ExtractionMethod(extraction_method),
            chunk_count=chunk_count,
            ingested_at=ingested_at,
        )

    return _make


@pytest.fixture
def make_search_hit(make_chunk: Callable[..., object]) -> Callable[..., object]:
    """Factory: build a :class:`SearchHit`. Composes :func:`make_chunk`.

    Usage::

        hit = make_search_hit(score=0.92)
        hit2 = make_search_hit(chunk=my_chunk, score=0.5)
    """

    def _make(
        *,
        chunk: object | None = None,
        score: float = 0.85,
    ) -> object:
        from lean.store.search import SearchHit

        return SearchHit(
            chunk=chunk or make_chunk(score=score),
            score=score,
        )

    return _make


# --- Mock DB fixtures ---------------------------------------------------------


@pytest.fixture
def mock_cursor() -> Callable[..., MagicMock]:
    """Factory: build a context-manager-aware mock psycopg cursor.

    Returns a callable that accepts optional ``fetchone_return`` /
    ``fetchall_return`` values::

        cursor = mock_cursor(fetchone_return=row, fetchall_return=rows)
    """

    def _make(
        fetchone_return: object | None = None,
        fetchall_return: list[object] | None = None,
    ) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.return_value = fetchone_return
        cursor.fetchall.return_value = fetchall_return or []
        return cursor

    return _make


@pytest.fixture
def mock_conn(mock_cursor: Callable[..., MagicMock]) -> Callable[..., MagicMock]:
    """Factory: build a mock :class:`StoreConnection` wrapping a cursor.

    Returns a callable that accepts an optional cursor override::

        conn = mock_conn()                    # auto-creates a cursor
        conn = mock_conn(cursor=my_cursor)    # uses provided cursor
    """

    def _make(cursor: MagicMock | None = None) -> MagicMock:
        conn = MagicMock()
        conn.conn.cursor.return_value = cursor or mock_cursor()
        return conn

    return _make


# --- Filesystem fixtures ------------------------------------------------------


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    """Write a minimal PDF to ``tmp_path`` and return its path.

    Eliminates the ``_write_minimal_pdf`` helper duplicated across three
    extraction test files.
    """
    from fpdf import FPDF

    path = tmp_path / "test.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 10, "Test PDF content")
    pdf.output(str(path))
    return path
