"""Unit tests for the search service chunk_type validation."""

from __future__ import annotations

import pytest

_VALID_KEY = "x" * 32


def test_valid_chunk_types_constant():
    from lean.services.search import VALID_CHUNK_TYPES

    assert frozenset({"text", "image"}) == VALID_CHUNK_TYPES


def test_search_rejects_invalid_chunk_type(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.services.search import search

    with pytest.raises(ValueError, match="chunk_type must be one of"):
        search("test query", chunk_type="invalid")


def test_search_rejects_empty_chunk_type_string(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.services.search import search

    with pytest.raises(ValueError, match="chunk_type must be one of"):
        search("test query", chunk_type="images")


def test_search_rejects_typo_chunk_type(monkeypatch):
    """Common typo 'imag' must be caught — the original silent-failure bug."""
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.services.search import search

    with pytest.raises(ValueError, match="chunk_type must be one of"):
        search("test query", chunk_type="imag")
