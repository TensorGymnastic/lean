"""Chunk CRUD operations — the ``chunks`` table."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID  # noqa: TC003

from psycopg.rows import dict_row

from lean.models.schemas import Chunk
from lean.store.base import StoreConnection


@dataclass
class ChunkRow:
    """Input shape for inserting a chunk with its embedding."""

    document_id: UUID
    chunk_index: int
    section_path: str
    heading_text: str | None
    page_start: int | None
    page_end: int | None
    token_count: int
    content: str
    embedding: list[float]


class ChunkRepo:
    """Read/write access to the ``public.chunks`` table."""

    def __init__(self, conn: StoreConnection) -> None:
        self._conn = conn

    def replace_chunks(self, document_id: UUID, chunks: list[ChunkRow]) -> None:
        """Delete existing chunks for a document and insert new ones."""
        with self._conn.conn.cursor() as cur:
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
        self._conn.conn.commit()

    def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        """Fetch a single chunk by ID. Returns None if not found."""
        with self._conn.conn.cursor(row_factory=dict_row) as cur:
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
        return Chunk.from_row(r)

    def count_chunks(self) -> int:
        with self._conn.conn.cursor() as cur:
            cur.execute("select count(*) from public.chunks")
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("count(*) returned no row")
            return int(row[0])

    def get_chunks_by_document(self, doc_id: UUID, limit: int = 1000) -> list[Chunk]:
        """Fetch chunks for a document, ordered by chunk_index. No embeddings."""
        with self._conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                select id, document_id, chunk_index, section_path, heading_text,
                       page_start, page_end, token_count, content
                from public.chunks
                where document_id = %s
                order by chunk_index
                limit %s
                """,
                (doc_id, limit),
            )
            rows = cur.fetchall()
        return [Chunk.from_row(r) for r in rows]
