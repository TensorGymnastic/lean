"""Direct Postgres + pgvector operations for documents and chunks.

This module talks to Supabase's Postgres directly via psycopg (not the
Supabase REST API) because pgvector operations and batch inserts are far
more efficient over a real DB connection.
"""

from __future__ import annotations

import json
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
        author: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        min_score: float | None = None,
    ) -> list[SearchHit]:
        """Cosine similarity search over chunks with metadata filters."""
        needs_join = author is not None or year_min is not None or year_max is not None
        join_clause = " join public.documents d on d.id = c.document_id" if needs_join else ""

        conditions = [
            "(%s::uuid is null or c.document_id = %s)",
            "(%s::text is null or c.section_path ilike '%%' || %s || '%%')",
        ]
        params: list[object] = [doc_id, doc_id, section_substring, section_substring]

        if author is not None:
            conditions.append(
                "exists (select 1 from unnest(d.authors) a where a ilike '%%' || %s || '%%')"
            )
            params.append(author)
        if year_min is not None:
            conditions.append("d.year >= %s")
            params.append(year_min)
        if year_max is not None:
            conditions.append("d.year <= %s")
            params.append(year_max)
        if min_score is not None:
            conditions.append("1 - (c.embedding <=> %s::vector) >= %s")
            params.extend([query_embedding, min_score])

        where_clause = " and ".join(conditions)

        query = f"""
            select c.id, c.document_id, c.chunk_index, c.section_path,
                   c.heading_text, c.page_start, c.page_end, c.token_count,
                   c.content,
                   1 - (c.embedding <=> %s::vector) as score
            from public.chunks c{join_clause}
            where {where_clause}
            order by c.embedding <=> %s::vector
            limit %s
        """
        params_final = [query_embedding] + params + [query_embedding, k]
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, tuple(params_final))
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

    def log_query(
        self,
        *,
        query_text: str,
        k: int,
        filters: dict[str, object],
        hit_chunk_ids: list[UUID],
        hit_scores: list[float],
        latency_ms: int,
        agent_id: str | None = None,
    ) -> None:
        """Persist a search query log entry for analytics and evaluation."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                insert into public.query_logs
                    (query_text, k, filters, hit_chunk_ids, hit_scores, latency_ms, agent_id)
                values (%s, %s, %s::jsonb, %s::uuid[], %s, %s, %s)
                """,
                (
                    query_text,
                    k,
                    json.dumps(filters),
                    hit_chunk_ids,
                    hit_scores,
                    latency_ms,
                    agent_id,
                ),
            )
            self._conn.commit()

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
