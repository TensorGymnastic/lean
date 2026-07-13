"""Document CRUD operations — the ``documents`` table.

Holds upsert, delete, listing, and storage-path lookup. Takes a
``StoreConnection`` and never opens a connection itself.
"""

from __future__ import annotations

from uuid import UUID  # noqa: TC003

from lean.models.schemas import DocumentSummary
from lean.store.base import StoreConnection


class DocumentRepo:
    """Read/write access to the ``public.documents`` table."""

    def __init__(self, conn: StoreConnection) -> None:
        self._conn = conn

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
        with self._conn.conn.cursor() as cur:
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
            self._conn.conn.commit()
            assert row is not None, "upsert_document returned no row"
            return UUID(str(row[0]))

    def delete_document(self, document_id: UUID) -> None:
        """Delete a document and cascade-delete its chunks."""
        with self._conn.conn.cursor() as cur:
            cur.execute(
                "delete from public.documents where id = %s",
                (document_id,),
            )
        self._conn.conn.commit()

    def list_documents(self) -> list[DocumentSummary]:
        """List all documents in the corpus with chunk counts, newest first."""
        from psycopg.rows import dict_row

        with self._conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                select d.id, d.source_path, d.title, d.authors, d.page_count,
                       d.extraction_method, d.ingested_at,
                       count(c.id) as chunk_count
                from public.documents d
                left join public.chunks c on c.document_id = d.id
                group by d.id
                order by d.ingested_at desc
                """
            )
            rows = cur.fetchall()
        return [
            DocumentSummary(
                id=str(r["id"]),
                source_path=r["source_path"],
                title=r["title"],
                authors=r["authors"] or [],
                page_count=r["page_count"],
                extraction_method=r["extraction_method"],
                chunk_count=r["chunk_count"],
                ingested_at=r["ingested_at"],
            )
            for r in rows
        ]

    def get_document_storage_paths(self, document_id: UUID) -> tuple[str, str] | None:
        """Return ``(source_storage_path, markdown_storage_path)`` or None."""
        with self._conn.conn.cursor() as cur:
            cur.execute(
                "select source_storage_path, markdown_storage_path "
                "from public.documents where id = %s",
                (document_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return str(row[0]), str(row[1])
