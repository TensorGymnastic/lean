"""Integration tests for the SQL schema (requires local Supabase running).

Marked @pytest.mark.integration — skipped unless SUPABASE_DB_URL is set
and Supabase is reachable. Run with:

    SUPABASE_DB_URL=postgresql://postgres:postgres@localhost:54322/postgres \\
        uv run pytest tests/test_db_schema.py -v -m integration
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def db_url() -> str:
    url = os.environ.get("SUPABASE_DB_URL", "")
    if not url:
        pytest.skip("SUPABASE_DB_URL not set — start Supabase first (make db-init)")
    return url


def test_extensions_installed(db_url: str) -> None:
    """Required extensions (vector, uuid-ossp, pg_trgm) are installed."""
    import psycopg

    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(
            "select extname from pg_extension where extname in ('vector', 'uuid-ossp', 'pg_trgm')"
        )
        names = {row[0] for row in cur.fetchall()}
    assert names == {"vector", "uuid-ossp", "pg_trgm"}


def test_documents_table_exists(db_url: str) -> None:
    """documents table exists with the expected columns."""
    import psycopg

    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(
            "select count(*) from information_schema.tables "
            "where table_schema = 'public' and table_name = 'documents'"
        )
        assert cur.fetchone()[0] == 1

        cur.execute(
            "select column_name from information_schema.columns "
            "where table_schema = 'public' and table_name = 'documents' "
            "order by ordinal_position"
        )
        cols = {row[0] for row in cur.fetchall()}
    expected = {
        "id",
        "source_path",
        "source_sha256",
        "title",
        "authors",
        "publisher",
        "year",
        "page_count",
        "extraction_method",
        "source_storage_path",
        "markdown_storage_path",
        "ingested_at",
        "reingested_at",
        "metadata",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"


def test_chunks_table_has_vector_column(db_url: str) -> None:
    """chunks table has a vector(1024) embedding column."""
    import psycopg

    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(
            "select data_type, udt_name from information_schema.columns "
            "where table_schema = 'public' and table_name = 'chunks' "
            "and column_name = 'embedding'"
        )
        row = cur.fetchone()
    assert row is not None, "embedding column not found"
    assert row[1] == "vector", f"expected vector type, got {row[1]}"


def test_chunks_unique_constraint(db_url: str) -> None:
    """(document_id, chunk_index) has a unique constraint."""
    import psycopg

    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(
            "select conname, contype from pg_constraint "
            "where conrelid = 'public.chunks'::regclass and contype = 'u'"
        )
        constraints = cur.fetchall()
    assert len(constraints) >= 1, "expected at least one UNIQUE constraint on chunks"
