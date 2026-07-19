"""Universal chunk CRUD — the ``public.chunks`` table.

Universal columns only. Domains add their own via SQL migrations and
extend the dataclass if they need extra fields stored at the DB layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID  # noqa: TC003

from psycopg.rows import dict_row

from lean.core.models import CHUNK_TYPE_TEXT, Chunk
from lean.core.store.base import StoreConnection

_INSERT_BATCH_SIZE = 1000


@dataclass
class ChunkRow:
    """Universal input shape for inserting a chunk with its embedding.

    Domains that need extra DB columns (e.g. ``image_hash`` for
    multimodal corpora) can subclass this dataclass and the matching
    ``ChunkRepo.replace_chunks`` (see lean-lss for the pattern).
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
    chunk_type: str = CHUNK_TYPE_TEXT


class ChunkRepo:
    """Read/write access to the universal columns of ``public.chunks``.

    Pass ``commit=False`` when the caller manages the transaction
    (e.g. to keep doc-upsert + chunk-replace atomic in a single
    transaction). The caller is then responsible for committing.
    """

    def __init__(self, conn: StoreConnection) -> None:
        self._conn = conn

    def replace_chunks(
        self,
        document_id: UUID,
        chunks: list[ChunkRow],
        *,
        commit: bool = True,
    ) -> None:
        with self._conn.conn.cursor() as cur:
            cur.execute(
                "delete from public.chunks where document_id = %s",
                (document_id,),
            )
            if chunks:
                rows = [
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
                        c.chunk_type,
                    )
                    for c in chunks
                ]
                insert_sql = """
                    insert into public.chunks
                        (document_id, chunk_index, section_path, heading_text,
                         page_start, page_end, token_count, content, embedding,
                         chunk_type)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                for i in range(0, len(rows), _INSERT_BATCH_SIZE):
                    cur.executemany(insert_sql, rows[i : i + _INSERT_BATCH_SIZE])
        if commit:
            self._conn.conn.commit()

    def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        with self._conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                select id, document_id, chunk_index, section_path, heading_text,
                       page_start, page_end, token_count, content, chunk_type
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
        with self._conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                select id, document_id, chunk_index, section_path, heading_text,
                       page_start, page_end, token_count, content, chunk_type
                from public.chunks
                where document_id = %s
                order by chunk_index
                limit %s
                """,
                (doc_id, limit),
            )
            rows = cur.fetchall()
        return [Chunk.from_row(r) for r in rows]


__all__ = ["ChunkRow", "ChunkRepo"]
