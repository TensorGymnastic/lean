"""Retrieval evaluation: hit_rate@k, MRR@k, and eval dataset generation.

Pure Python implementations of standard IR metrics. No LlamaIndex dependency.

The eval dataset is built by sampling chunks from the corpus and using
their heading text + content as pseudo-queries. When an LLM endpoint is
available, richer question generation can be added.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from psycopg.rows import dict_row

from lean.store.pgvector import PgVectorStore, SearchHit


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
    mean_latency_ms: float
    sample_count: int
    k: int


def build_eval_dataset(
    store: PgVectorStore,
    *,
    sample_size: int = 50,
    seed: int = 42,
) -> list[EvalSample]:
    """Build an eval dataset by sampling chunks from the corpus.

    Uses heading_text + first 100 chars of content as the pseudo-query.
    """
    with store._conn.cursor(row_factory=dict_row) as cur:
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
    store: PgVectorStore,
    samples: list[EvalSample],
    *,
    k: int = 5,
) -> EvalResult:
    """Run retrieval evaluation: compute hit_rate@k and MRR@k.

    For each sample, embed the query, search top-k, check if the expected
    chunk appears in the results.
    """
    from lean.embeddings.liquid_lmf import LiquidLMFEmbedder
    from lean.settings import Settings

    settings = Settings()
    embedder = LiquidLMFEmbedder(
        model=settings.embedding_model,
        hf_token=settings.hf_token,
        device=settings.embedding_device,
        dim=settings.embedding_dim,
    )

    hits = 0
    reciprocal_ranks: list[float] = []
    latencies: list[int] = []

    for sample in samples:
        start = time.monotonic()
        query_vec = embedder.embed_query(sample.query)

        results: list[SearchHit] = store.search(
            query_embedding=query_vec,
            k=k,
        )
        latencies.append(int((time.monotonic() - start) * 1000))

        chunk_ids = [h.chunk.id for h in results]
        if sample.expected_chunk_id in chunk_ids:
            hits += 1
            rank = chunk_ids.index(sample.expected_chunk_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(samples) if samples else 1
    return EvalResult(
        hit_rate=hits / n,
        mrr=sum(reciprocal_ranks) / n,
        mean_latency_ms=sum(latencies) / n,
        sample_count=len(samples),
        k=k,
    )
