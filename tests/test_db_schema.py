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


def test_chunk_type_check_constraint_rejects_invalid(db_url: str) -> None:
    """Migration 012's CHECK (chunk_type IN ('text','image')) rejects bogus values."""
    import psycopg

    fake_doc_id = "00000000-0000-0000-0000-000000000001"
    with psycopg.connect(db_url, autocommit=False) as conn, conn.cursor() as cur:
        try:
            cur.execute(
                "insert into public.chunks "
                "(document_id, chunk_index, section_path, token_count, content, "
                " embedding, chunk_type) "
                "values (%s, 999999, 'X', 1, 'x', '[0]'::vector, 'bogus_type')",
                (fake_doc_id,),
            )
        except psycopg.errors.CheckViolation:
            conn.rollback()
        else:
            conn.rollback()
            raise AssertionError("CHECK constraint did not reject chunk_type='bogus_type'")


def test_on_delete_cascade_removes_chunks(db_url: str) -> None:
    """FK from chunks.document_id -> documents.id has ON DELETE CASCADE."""
    import psycopg

    fake_doc_id = "00000000-0000-0000-0000-000000000002"
    with psycopg.connect(db_url, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "insert into public.documents "
            "(id, source_path, source_sha256, title, extraction_method) "
            "values (%s, 'cascade-test.pdf', 'cascade-test-%s', 'Cascade Test', 'markitdown') "
            "on conflict (source_sha256) do nothing",
            (fake_doc_id, fake_doc_id),
        )
        cur.execute(
            "insert into public.chunks "
            "(document_id, chunk_index, section_path, token_count, content, embedding) "
            "values (%s, 0, 'X', 1, 'x', '[0]'::vector)",
            (fake_doc_id,),
        )
        cur.execute(
            "select count(*) from public.chunks where document_id = %s",
            (fake_doc_id,),
        )
        assert cur.fetchone()[0] == 1

        cur.execute("delete from public.documents where id = %s", (fake_doc_id,))

        cur.execute(
            "select count(*) from public.chunks where document_id = %s",
            (fake_doc_id,),
        )
        assert cur.fetchone()[0] == 0, "CASCADE did not remove chunks"


def test_vector_1024_embedding_dim_validated(db_url: str) -> None:
    """The vector(1024) type rejects embeddings of a different dimension."""
    import psycopg

    fake_doc_id = "00000000-0000-0000-0000-000000000003"
    with psycopg.connect(db_url, autocommit=False) as conn, conn.cursor() as cur:
        try:
            cur.execute(
                "insert into public.chunks "
                "(document_id, chunk_index, section_path, token_count, content, embedding) "
                "values (%s, 888888, 'X', 1, 'x', %s::vector)",
                (fake_doc_id, "[" + ",".join(["0"] * 1536) + "]"),
            )
        except psycopg.errors.DataException:
            conn.rollback()
        else:
            conn.rollback()
            raise AssertionError("vector(1024) did not reject a 1536-dim embedding")
