"""Chunk CRUD operations — the ``chunks`` table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID  # noqa: TC003

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from lean.models.schemas import CHUNK_TYPE_TEXT, Chunk
from lean.store.base import StoreConnection

# Postgres caps prepared statements at 65535 parameters. The chunks INSERT has 16
# columns per row, so a single executemany hits the ceiling at ~4096 rows. Batching
# at 1000 keeps a 4× headroom and tolerates future column additions up to ~24.
_INSERT_BATCH_SIZE = 1000


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
    chunk_type: str = CHUNK_TYPE_TEXT
    image_meta: dict[str, Any] | None = None
    bbox: dict[str, Any] | None = None
    image_hash: str | None = None
    provenance_model: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None


class ChunkRepo:
    """Read/write access to the ``public.chunks`` table."""

    def __init__(self, conn: StoreConnection) -> None:
        self._conn = conn

    def replace_chunks(
        self,
        document_id: UUID,
        chunks: list[ChunkRow],
        *,
        commit: bool = True,
    ) -> None:
        """Delete existing chunks for a document and insert new ones.

        Pass ``commit=False`` when the caller manages the transaction
        (e.g. to keep doc-upsert and chunk-replace atomic in a single
        transaction). The caller is then responsible for committing.
        """
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
                        Jsonb(c.image_meta) if c.image_meta else None,
                        Jsonb(c.bbox) if c.bbox else None,
                        c.image_hash,
                        c.provenance_model,
                        c.embedding_model,
                        c.embedding_dim,
                    )
                    for c in chunks
                ]
                insert_sql = """
                    insert into public.chunks
                        (document_id, chunk_index, section_path, heading_text,
                         page_start, page_end, token_count, content, embedding,
                         chunk_type, image_meta, bbox, image_hash,
                         provenance_model, embedding_model, embedding_dim)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                for i in range(0, len(rows), _INSERT_BATCH_SIZE):
                    cur.executemany(insert_sql, rows[i : i + _INSERT_BATCH_SIZE])
        if commit:
            self._conn.conn.commit()

    def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        """Fetch a single chunk by ID. Returns None if not found."""
        with self._conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                select id, document_id, chunk_index, section_path, heading_text,
                       page_start, page_end, token_count, content, chunk_type,
                       image_meta, bbox, image_hash, provenance_model,
                       embedding_model, embedding_dim
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
                       page_start, page_end, token_count, content, chunk_type,
                       image_meta, bbox, image_hash, provenance_model,
                       embedding_model, embedding_dim
                from public.chunks
                where document_id = %s
                order by chunk_index
                limit %s
                """,
                (doc_id, limit),
            )
            rows = cur.fetchall()
        return [Chunk.from_row(r) for r in rows]

    def find_duplicate_image_hashes(self) -> list[tuple[str, int]]:
        """Return ``(image_hash, count)`` pairs for hashes appearing in more than one chunk.

        Diagnostic only — surfaces duplicate images across the corpus. Future ingest
        optimization could skip VLM description for hashes already in the DB.
        """
        with self._conn.conn.cursor() as cur:
            cur.execute(
                """
                select image_hash, count(*) as cnt
                from public.chunks
                where image_hash is not null
                group by image_hash
                having count(*) > 1
                order by cnt desc
                """
            )
            return [(row[0], int(row[1])) for row in cur.fetchall()]
