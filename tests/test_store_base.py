"""Unit tests for store/base.py — StoreConnection context manager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

_VALID_KEY = "x" * 32


def test_store_connection_context_manager(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.base import StoreConnection

    mock_conn = MagicMock()
    with (
        patch("lean.store.base.psycopg.connect", return_value=mock_conn),
        patch("lean.store.base.register_vector"),
        patch("lean.store.base.get_settings") as mock_settings,
    ):
        mock_settings.return_value = MagicMock(supabase_db_url="postgresql://localhost/postgres")
        store = StoreConnection("postgresql://localhost/postgres")
        assert store.conn is mock_conn
        with store as ctx:
            assert ctx is store
        mock_conn.close.assert_called_once()


def test_store_connection_close(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.base import StoreConnection

    mock_conn = MagicMock()
    with (
        patch("lean.store.base.psycopg.connect", return_value=mock_conn),
        patch("lean.store.base.register_vector"),
    ):
        store = StoreConnection("postgresql://localhost/postgres")
        store.close()
        mock_conn.close.assert_called_once()


def test_store_connection_from_env(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.base import StoreConnection

    mock_conn = MagicMock()
    with (
        patch("lean.store.base.psycopg.connect", return_value=mock_conn),
        patch("lean.store.base.register_vector"),
        patch("lean.store.base.get_settings") as mock_settings,
    ):
        mock_settings.return_value = MagicMock(supabase_db_url="postgresql://localhost/postgres")
        store = StoreConnection.from_env()
        assert store.conn is mock_conn
