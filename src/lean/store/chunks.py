"""Chunk CRUD operations — the ``chunks`` table.

Holds the ``ChunkRow`` input shape plus replace/get/count operations.
Takes a ``StoreConnection`` and never opens a connection itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID  # noqa: TC003

from lean.models.schemas import Chunk
from lean.store.base import StoreConnection


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


class ChunkRepo:
    """Read/write access to the ``public.chunks`` table."""

    def __init__(self, conn: StoreConnection) -> None:
        self._conn = conn

    def replace_chunks(self, document_id: UUID, chunks: list[ChunkRow]) -> None:
        """Delete existing chunks for a document and insert new ones.

        Atomic within a single transaction.
        """
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
        from psycopg.rows import dict_row

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

    def count_chunks(self) -> int:
        with self._conn.conn.cursor() as cur:
            cur.execute("select count(*) from public.chunks")
            row = cur.fetchone()
            assert row is not None, "count(*) returned no row"
            return int(row[0])
