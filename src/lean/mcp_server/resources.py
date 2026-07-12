"""MCP resources for the lean corpus — read-only data exposed to agents."""

from __future__ import annotations

import json

from lean.mcp_server.tools import mcp


@mcp.resource("lean://documents")
async def documents_resource() -> str:
    """List all documents in the corpus (JSON array)."""
    from lean.mcp_server.tools import list_documents

    docs = await list_documents()
    return json.dumps([d.model_dump(mode="json") for d in docs], indent=2)


@mcp.resource("lean://documents/{doc_id}/markdown")
async def markdown_resource(doc_id: str) -> str:
    """Full extracted markdown for a document."""
    from lean.mcp_server.tools import get_document_markdown

    return await get_document_markdown(doc_id)


@mcp.resource("lean://documents/{doc_id}/chunks")
async def chunks_resource(doc_id: str) -> str:
    """Chunk metadata for a document (JSON array, no embeddings)."""
    from uuid import UUID

    from psycopg.rows import dict_row

    from lean.store.pgvector import PgVectorStore

    store = PgVectorStore.from_env()
    try:
        with store._conn.cursor(row_factory=dict_row) as cur:  # noqa: SLF001
            cur.execute(
                "select id, chunk_index, section_path, heading_text, "
                "page_start, page_end, token_count "
                "from public.chunks where document_id = %s order by chunk_index",
                (UUID(doc_id),),
            )
            rows = cur.fetchall()
    finally:
        store.close()
    return json.dumps(
        [
            {
                "id": str(r["id"]),
                "chunk_index": r["chunk_index"],
                "section_path": r["section_path"],
                "heading_text": r["heading_text"],
                "page_start": r["page_start"],
                "page_end": r["page_end"],
                "token_count": r["token_count"],
            }
            for r in rows
        ],
        indent=2,
    )


@mcp.resource("lean://stats")
async def stats_resource() -> str:
    """Corpus statistics (JSON)."""
    from lean.mcp_server.tools import corpus_stats

    stats = await corpus_stats()
    return stats.model_dump_json(indent=2)
