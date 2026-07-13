"""Connection management for the pgvector store.

A ``StoreConnection`` owns a single psycopg connection with pgvector
registered. It is the single dependency every focused repo module takes
in its constructor; the repos never open or close connections themselves.
"""

from __future__ import annotations

import os
from typing import Any, Self

import psycopg
from pgvector.psycopg import register_vector

from lean.config.settings import Settings


class StoreConnection:
    """Connection-scoped store. Create one per logical operation.

    The connection is held open for the lifetime of the instance.
    Call ``close()`` when done, or use as a context manager.
    """

    _conn: psycopg.Connection[Any]

    def __init__(self, db_url: str) -> None:
        self._conn = psycopg.connect(db_url, autocommit=False)
        register_vector(self._conn)

    @classmethod
    def from_env(cls) -> Self:
        """Create from ``SUPABASE_DB_URL`` environment variable or Settings."""
        db_url = os.environ.get("SUPABASE_DB_URL")
        if db_url is None:
            db_url = Settings().supabase_db_url
        return cls(db_url=db_url)

    @property
    def conn(self) -> psycopg.Connection[Any]:
        """The underlying psycopg connection."""
        return self._conn

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()
