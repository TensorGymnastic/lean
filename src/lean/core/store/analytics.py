"""Query logging and corpus statistics.

Holds query-log persistence, document/chunk counting, and aggregate corpus
stats. Takes a ``StoreConnection`` and never opens a connection itself.
"""

from __future__ import annotations

import json
from uuid import UUID  # noqa: TC003

from psycopg.rows import dict_row

from lean.core.models.schemas import CorpusStats
from lean.core.store.base import StoreConnection


class AnalyticsRepo:
    """Query logging and aggregate statistics for the corpus."""

    def __init__(self, conn: StoreConnection) -> None:
        self._conn = conn

    def log_query(
        self,
        *,
        query_text: str,
        k: int,
        filters: dict[str, object],
        hit_chunk_ids: list[UUID],
        hit_scores: list[float],
        latency_ms: int,
    ) -> None:
        """Persist a search query log entry for analytics and evaluation."""
        with self._conn.conn.cursor() as cur:
            cur.execute(
                """
                insert into public.query_logs
                    (query_text, k, filters, hit_chunk_ids, hit_scores, latency_ms)
                values (%s, %s, %s::jsonb, %s::uuid[], %s, %s)
                """,
                (
                    query_text,
                    k,
                    json.dumps(filters),
                    hit_chunk_ids,
                    hit_scores,
                    latency_ms,
                ),
            )
            self._conn.conn.commit()

    def save_eval_run(
        self,
        *,
        config: dict[str, object],
        hit_rate: float,
        mrr: float,
        ndcg: float = 0.0,
        recall: float = 0.0,
        mean_latency_ms: int,
        sample_count: int,
        k: int,
    ) -> None:
        """Persist a retrieval evaluation run for trending."""
        with self._conn.conn.cursor() as cur:
            cur.execute(
                """
                insert into public.eval_runs
                    (config, hit_rate, mrr, ndcg, recall, mean_latency_ms, sample_count, k)
                values (%s::jsonb, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    json.dumps(config),
                    hit_rate,
                    mrr,
                    ndcg,
                    recall,
                    mean_latency_ms,
                    sample_count,
                    k,
                ),
            )
            self._conn.conn.commit()

    def count_documents(self) -> int:
        with self._conn.conn.cursor() as cur:
            cur.execute("select count(*) from public.documents")
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("count(*) returned no row")
            return int(row[0])

    def corpus_stats(
        self,
        *,
        embedding_dim: int,
        embedding_model: str,
    ) -> CorpusStats:
        """Return corpus statistics: document/chunk counts, extraction breakdown."""
        with self._conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute("select count(*) from public.documents")
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("count(*) returned no row")
            doc_count: int = row["count"]
            cur.execute("select count(*) from public.chunks")
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("count(*) returned no row")
            chunk_count: int = row["count"]
            cur.execute("select coalesce(sum(token_count), 0) as total from public.chunks")
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("count(*) returned no row")
            total_tokens: int = row["total"]
            cur.execute(
                "select extraction_method, count(*) as cnt "
                "from public.documents group by extraction_method"
            )
            breakdown = {r["extraction_method"]: r["cnt"] for r in cur.fetchall()}
            cur.execute("select max(ingested_at) as last from public.documents")
            last_row = cur.fetchone()
            last_ingested = last_row["last"] if last_row else None

        return CorpusStats(
            document_count=doc_count,
            chunk_count=chunk_count,
            total_tokens=total_tokens,
            extraction_method_breakdown=breakdown,
            embedding_dim=embedding_dim,
            embedding_model=embedding_model,
            last_ingested_at=last_ingested,
        )
