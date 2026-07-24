"""Vector and BM25 search plus Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID  # noqa: TC003

from psycopg.rows import dict_row

from lean.core.models.schemas import Chunk
from lean.core.store.base import StoreConnection

_SELECT_COLUMNS = """
    c.id, c.document_id, c.chunk_index, c.section_path,
    c.heading_text, c.page_start, c.page_end, c.token_count,
    c.content, c.chunk_type, c.image_meta,
    c.bbox, c.image_hash, c.provenance_model,
    c.embedding_model, c.embedding_dim
"""


@dataclass
class SearchHit:
    """A search result: a Chunk plus its similarity score."""

    chunk: Chunk
    score: float


class SearchEngine:
    """Vector + full-text search over the ``public.chunks`` table."""

    def __init__(self, conn: StoreConnection) -> None:
        self._conn = conn

    def _build_metadata_filters(
        self,
        *,
        doc_id: UUID | None,
        section_substring: str | None,
        author: str | None,
        year_min: int | None,
        year_max: int | None,
        chunk_type: str | None = None,
    ) -> tuple[str, list[str], list[object]]:
        """Return (join_clause, conditions, params) for shared metadata filters."""
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
        if chunk_type is not None:
            conditions.append("c.chunk_type = %s")
            params.append(chunk_type)

        return join_clause, conditions, params

    def _search(
        self,
        *,
        score_sql: str,
        score_params: list[object],
        order_sql: str,
        order_params: list[object],
        extra_where: list[str] | None = None,
        extra_params: list[object] | None = None,
        doc_id: UUID | None = None,
        section_substring: str | None = None,
        author: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        chunk_type: str | None = None,
        k: int = 5,
    ) -> list[SearchHit]:
        join_clause, conditions, params = self._build_metadata_filters(
            doc_id=doc_id,
            section_substring=section_substring,
            author=author,
            year_min=year_min,
            year_max=year_max,
            chunk_type=chunk_type,
        )
        if extra_where:
            conditions[:0] = extra_where
            params[:0] = extra_params or []

        where_clause = " and ".join(conditions)
        sql = f"""
            select {_SELECT_COLUMNS}, {score_sql}
            from public.chunks c{join_clause}
            where {where_clause}
            order by {order_sql}
            limit %s
        """
        params_final = score_params + params + order_params + [k]
        with self._conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, tuple(params_final))
            rows = cur.fetchall()
        return self._rows_to_hits(rows)

    def vector_search(
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
        chunk_type: str | None = None,
    ) -> list[SearchHit]:
        """Cosine similarity search over chunks with metadata filters."""
        extra_where: list[str] = []
        extra_params: list[object] = []
        if min_score is not None:
            extra_where.append("1 - (c.embedding <=> %s::vector) >= %s")
            extra_params.extend([query_embedding, min_score])
        return self._search(
            score_sql="1 - (c.embedding <=> %s::vector) as score",
            score_params=[query_embedding],
            order_sql="c.embedding <=> %s::vector",
            order_params=[query_embedding],
            extra_where=extra_where or None,
            extra_params=extra_params or None,
            doc_id=doc_id,
            section_substring=section_substring,
            author=author,
            year_min=year_min,
            year_max=year_max,
            chunk_type=chunk_type,
            k=k,
        )

    def bm25_search(
        self,
        *,
        query_text: str,
        k: int = 5,
        doc_id: UUID | None = None,
        section_substring: str | None = None,
        author: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        chunk_type: str | None = None,
    ) -> list[SearchHit]:
        """Full-text search via tsvector + ts_rank (BM25-style ranking)."""
        return self._search(
            score_sql="ts_rank(c.tsv, plainto_tsquery('english', %s)) as score",
            score_params=[query_text],
            order_sql="score desc",
            order_params=[],
            extra_where=["c.tsv @@ plainto_tsquery('english', %s)"],
            extra_params=[query_text],
            doc_id=doc_id,
            section_substring=section_substring,
            author=author,
            year_min=year_min,
            year_max=year_max,
            chunk_type=chunk_type,
            k=k,
        )

    @staticmethod
    def _rows_to_hits(rows: list[dict[str, Any]]) -> list[SearchHit]:
        """Build ``SearchHit`` list from dict rows."""
        return [
            SearchHit(
                chunk=Chunk.from_row(r).model_copy(update={"score": float(r["score"])}),
                score=float(r["score"]),
            )
            for r in rows
        ]

    @staticmethod
    def reciprocal_rank_fusion(
        vector_hits: list[SearchHit],
        bm25_hits: list[SearchHit],
        *,
        k: int,
        rrf_k: int,
    ) -> list[SearchHit]:
        """Fuse two ranked lists using Reciprocal Rank Fusion.

        rrf_score = Σ 1/(rrf_k + rank) for each list the chunk appears in.
        Returns top-k by fused score. Score on returned hits is the RRF score.
        """
        scores: dict[str, float] = {}
        best_hit: dict[str, SearchHit] = {}

        for rank, hit in enumerate(vector_hits):
            cid = hit.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            best_hit[cid] = hit

        for rank, hit in enumerate(bm25_hits):
            cid = hit.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            if cid not in best_hit:
                best_hit[cid] = hit

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        result: list[SearchHit] = []
        for cid, score in ranked:
            hit = best_hit[cid]
            result.append(
                SearchHit(
                    chunk=hit.chunk.model_copy(update={"score": score}),
                    score=score,
                )
            )
        return result
