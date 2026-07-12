"""Direct Postgres + pgvector operations for documents and chunks.

This module talks to Supabase's Postgres directly via psycopg (not the
Supabase REST API) because pgvector operations and batch inserts are far
more efficient over a real DB connection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from uuid import UUID  # noqa: TC003

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from lean.models.schemas import Chunk


@dataclass
class ChunkRow:
    """Input shape for inserting a chunk with its embedding.

    ``embedding`` is a plain list[float] of length 1024 (LFM2.5-Embedding-350M).
    """

    document_id: UUID
    chunk_index: int
    section_path: str
    heading_text: str | None
    page_start: int | None
    page_end: int | None
    token_count: int
    content: str
    embedding: list[float]


@dataclass
class SearchHit:
    """A search result: a Chunk plus its cosine similarity score."""

    chunk: Chunk
    score: float


class PgVectorStore:
    """Connection-scoped store. Create one per logical operation.

    The connection is held open for the lifetime of the instance.
    Call ``close()`` when done, or use as a context manager.
    """

    def __init__(self, db_url: str) -> None:
        self._conn = psycopg.connect(db_url, autocommit=False)
        register_vector(self._conn)

    @classmethod
    def from_env(cls) -> PgVectorStore:
        """Create from ``SUPABASE_DB_URL`` environment variable."""
        return cls(db_url=os.environ["SUPABASE_DB_URL"])

    def __enter__(self) -> PgVectorStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def upsert_document(
        self,
        *,
        source_path: str,
        source_sha256: str,
        title: str | None,
        extraction_method: str,
        source_storage_path: str,
        markdown_storage_path: str,
        authors: list[str] | None = None,
        publisher: str | None = None,
        year: int | None = None,
        page_count: int | None = None,
    ) -> UUID:
        """Insert or update a document. Returns the document UUID.

        On conflict (same source_sha256), updates the source_path, title,
        and sets reingested_at. Returns the existing or new id.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                insert into public.documents
                    (source_path, source_sha256, title, authors, publisher, year,
                     page_count, extraction_method, source_storage_path,
                     markdown_storage_path)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (source_sha256) do update set
                    source_path = excluded.source_path,
                    title = excluded.title,
                    extraction_method = excluded.extraction_method,
                    reingested_at = now()
                returning id
                """,
                (
                    source_path,
                    source_sha256,
                    title,
                    authors or [],
                    publisher,
                    year,
                    page_count,
                    extraction_method,
                    source_storage_path,
                    markdown_storage_path,
                ),
            )
            row = cur.fetchone()
            self._conn.commit()
            assert row is not None, "upsert_document returned no row"
            return UUID(str(row[0]))

    def replace_chunks(self, document_id: UUID, chunks: list[ChunkRow]) -> None:
        """Delete existing chunks for a document and insert new ones.

        Atomic within a single transaction.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "delete from public.chunks where document_id = %s",
                (document_id,),
            )
            if chunks:
                cur.executemany(
                    """
                    insert into public.chunks
                        (document_id, chunk_index, section_path, heading_text,
                         page_start, page_end, token_count, content, embedding)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            c.document_id,
                            c.chunk_index,
                            c.section_path,
                            c.heading_text,
                            c.page_start,
                            c.page_end,
                            c.token_count,
                            c.content,
                            c.embedding,
                        )
                        for c in chunks
                    ],
                )
        self._conn.commit()

    def search(
        self,
        *,
        query_embedding: list[float],
        k: int = 5,
        doc_id: UUID | None = None,
        section_substring: str | None = None,
    ) -> list[SearchHit]:
        """Cosine similarity search over chunks.

        Filters:
        - ``doc_id``: restrict to a specific document.
        - ``section_substring``: case-insensitive substring match on section_path.

        Returns chunks sorted by cosine similarity (highest first).
        Score is ``1 - cosine_distance`` (so 1.0 = identical, 0.0 = orthogonal).
        """
        query = """
            select c.id, c.document_id, c.chunk_index, c.section_path,
                   c.heading_text, c.page_start, c.page_end, c.token_count,
                   c.content,
                   1 - (c.embedding <=> %s::vector) as score
            from public.chunks c
            where (%s::uuid is null or c.document_id = %s)
              and (%s::text is null or c.section_path ilike '%%' || %s || '%%')
            order by c.embedding <=> %s::vector
            limit %s
        """
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                query,
                (
                    query_embedding,
                    doc_id,
                    doc_id,
                    section_substring,
                    section_substring,
                    query_embedding,
                    k,
                ),
            )
            rows = cur.fetchall()
        return [
            SearchHit(
                chunk=Chunk(
                    id=str(r["id"]),
                    document_id=str(r["document_id"]),
                    chunk_index=r["chunk_index"],
                    section_path=r["section_path"],
                    heading_text=r["heading_text"],
                    page_start=r["page_start"],
                    page_end=r["page_end"],
                    token_count=r["token_count"],
                    content=r["content"],
                    score=float(r["score"]),
                ),
                score=float(r["score"]),
            )
            for r in rows
        ]

    def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        """Fetch a single chunk by ID. Returns None if not found."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                select id, document_id, chunk_index, section_path, heading_text,
                       page_start, page_end, token_count, content
                from public.chunks where id = %s
                """,
                (chunk_id,),
            )
            r = cur.fetchone()
        if r is None:
            return None
        return Chunk(
            id=str(r["id"]),
            document_id=str(r["document_id"]),
            chunk_index=r["chunk_index"],
            section_path=r["section_path"],
            heading_text=r["heading_text"],
            page_start=r["page_start"],
            page_end=r["page_end"],
            token_count=r["token_count"],
            content=r["content"],
        )

    def delete_document(self, document_id: UUID) -> None:
        """Delete a document and cascade-delete its chunks."""
        with self._conn.cursor() as cur:
            cur.execute(
                "delete from public.documents where id = %s",
                (document_id,),
            )
        self._conn.commit()

    def count_documents(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("select count(*) from public.documents")
            row = cur.fetchone()
            assert row is not None, "count(*) returned no row"
            return int(row[0])

    def count_chunks(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("select count(*) from public.chunks")
            row = cur.fetchone()
            assert row is not None, "count(*) returned no row"
            return int(row[0])
