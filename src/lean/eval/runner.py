"""Retrieval evaluation: hit_rate@k, MRR@k, NDCG@k, Recall@k.

Pure Python implementations of standard IR metrics (LlamaIndex-compatible
definitions). No LlamaIndex dependency.

The eval dataset is built by sampling chunks from the corpus and using
their heading text + content as pseudo-queries. For production eval,
replace with a hand-curated set of (query, expected_chunk_id) pairs.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from psycopg.rows import dict_row

from lean.infrastructure.embedder import get_embedder
from lean.store.base import StoreConnection
from lean.store.search import SearchEngine, SearchHit


@dataclass
class EvalSample:
    """One (query, expected_chunk_id) pair for evaluation."""

    query: str
    expected_chunk_id: str


@dataclass
class EvalResult:
    """Aggregated metrics across all eval samples."""

    hit_rate: float
    mrr: float
    ndcg: float
    recall: float
    mean_latency_ms: float
    sample_count: int
    k: int


def build_eval_dataset(
    store: StoreConnection,
    *,
    sample_size: int,
    seed: int,
) -> list[EvalSample]:
    """Build an eval dataset by sampling chunks from the corpus.

    Uses heading_text + first 100 chars of content as the pseudo-query.
    For production eval, replace with hand-curated (query, expected) pairs.
    """
    with store.conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            select id, heading_text, content
            from public.chunks
            where heading_text is not null
              and length(heading_text) > 5
            order by random()
            limit %s
            """,
            (sample_size,),
        )
        rows = cur.fetchall()

    rng = random.Random(seed)
    rng.shuffle(rows)

    samples: list[EvalSample] = []
    for r in rows:
        heading = r["heading_text"].strip()
        content_preview = r["content"][:100].strip().replace("\n", " ")
        query = f"{heading}: {content_preview}"
        samples.append(EvalSample(query=query, expected_chunk_id=str(r["id"])))
    return samples


def evaluate(
    store: StoreConnection,
    samples: list[EvalSample],
    *,
    k: int,
) -> EvalResult:
    """Run retrieval evaluation: compute hit_rate@k, MRR@k, NDCG@k, Recall@k.

    For each sample, embed the query, search top-k, check if the expected
    chunk appears in the results, and compute rank-aware metrics.
    """
    embedder = get_embedder()
    engine = SearchEngine(store)

    hits = 0
    reciprocal_ranks: list[float] = []
    dcg_values: list[float] = []
    recall_values: list[float] = []
    latencies: list[int] = []

    for sample in samples:
        start = time.monotonic()
        query_vec = embedder.embed_query(sample.query)

        results: list[SearchHit] = engine.vector_search(
            query_embedding=query_vec,
            k=k,
        )
        latencies.append(int((time.monotonic() - start) * 1000))

        chunk_ids = [h.chunk.id for h in results]
        found_rank: int | None = None
        if sample.expected_chunk_id in chunk_ids:
            hits += 1
            found_rank = chunk_ids.index(sample.expected_chunk_id) + 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

        dcg_values.append(_dcg_at_k(found_rank, k))
        recall_values.append(1.0 if found_rank is not None else 0.0)

    n = len(samples) if samples else 1
    return EvalResult(
        hit_rate=hits / n,
        mrr=sum(reciprocal_ranks) / n,
        ndcg=sum(dcg_values) / n,
        recall=sum(recall_values) / n,
        mean_latency_ms=sum(latencies) / n,
        sample_count=len(samples),
        k=k,
    )


def _dcg_at_k(rank: int | None, k: int) -> float:
    """Discounted Cumulative Gain at k for binary relevance.

    If the relevant item is found at position ``rank`` (1-based), DCG = 1/log2(rank+1).
    IDCG (ideal) = 1/log2(2) = 1. NDCG = DCG/IDCG.
    If not found, NDCG = 0.
    """
    if rank is None or rank > k:
        return 0.0
    dcg = 1.0 / math.log2(rank + 1)
    idcg = 1.0 / math.log2(2)
    return dcg / idcg
