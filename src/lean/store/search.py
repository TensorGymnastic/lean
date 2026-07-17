"""Vector and BM25 search plus Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID  # noqa: TC003

from psycopg.rows import dict_row

from lean.models.schemas import Chunk
from lean.store.base import StoreConnection


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

        return join_clause, conditions, params

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
    ) -> list[SearchHit]:
        """Cosine similarity search over chunks with metadata filters."""
        join_clause, conditions, params = self._build_metadata_filters(
            doc_id=doc_id,
            section_substring=section_substring,
            author=author,
            year_min=year_min,
            year_max=year_max,
        )
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
        with self._conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, tuple(params_final))
            rows = cur.fetchall()
        return self._rows_to_hits(rows)

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
    ) -> list[SearchHit]:
        """Full-text search via tsvector + ts_rank (BM25-style ranking)."""
        join_clause, conditions, params = self._build_metadata_filters(
            doc_id=doc_id,
            section_substring=section_substring,
            author=author,
            year_min=year_min,
            year_max=year_max,
        )
        conditions.insert(0, "c.tsv @@ plainto_tsquery('english', %s)")
        params.insert(0, query_text)

        where_clause = " and ".join(conditions)
        sql = f"""
            select c.id, c.document_id, c.chunk_index, c.section_path,
                   c.heading_text, c.page_start, c.page_end, c.token_count,
                   c.content,
                   ts_rank(c.tsv, plainto_tsquery('english', %s)) as score
            from public.chunks c{join_clause}
            where {where_clause}
            order by score desc
            limit %s
        """
        params_final = [query_text] + params + [k]
        with self._conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, tuple(params_final))
            rows = cur.fetchall()
        return self._rows_to_hits(rows)

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
