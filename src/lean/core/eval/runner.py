"""Retrieval evaluation: hit_rate@k, MRR@k, NDCG@k, Recall@k.

Pure Python implementations of standard IR metrics (LlamaIndex-compatible
definitions). No LlamaIndex dependency.

The default eval dataset is built by sampling chunks from the corpus and using
their heading text + content as pseudo-queries. For semi-curated eval, pass a
JSON file of (query, expected_chunk_id) pairs to ``load_curated_dataset`` and
run ``evaluate`` on the result — see ``data/curated_eval_dataset.json`` for the
format and ``scripts/build_eval_dataset.py`` for the builder.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from psycopg.rows import dict_row

from lean.core.infrastructure.embedder import get_embedder
from lean.core.store.base import StoreConnection
from lean.core.store.search import SearchEngine, SearchHit


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


def load_curated_dataset(path: Path) -> list[EvalSample]:
    """Load a curated eval dataset from a JSON file.

    Each entry must be a JSON object with non-empty ``query`` (str) and
    ``expected_chunk_id`` (str) fields. Extra fields (``heading``, ``section``,
    ``doc``) are silently dropped — they are diagnostic context written by
    ``scripts/build_eval_dataset.py`` and are not used by ``evaluate``.

    Args:
        path: Filesystem path to a JSON file whose root value is a list of
            the entry shapes described above.

    Returns:
        List of ``EvalSample`` (in JSON order).

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: JSON is malformed, root is not a list, or any entry is
            missing/has empty ``query`` or ``expected_chunk_id``, or any entry
            is not a JSON object.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        msg = f"invalid json in {path}: {e.msg} (line {e.lineno}, col {e.colno})"
        raise ValueError(msg) from e

    if not isinstance(data, list):
        msg = (
            f"curated dataset root must be a JSON list of objects, "
            f"got {type(data).__name__} in {path}"
        )
        raise ValueError(msg)

    samples: list[EvalSample] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            msg = f"entry {i} in {path} must be a JSON object, got {type(entry).__name__}"
            raise ValueError(msg)
        query = entry.get("query")
        if not isinstance(query, str) or not query.strip():
            msg = f"entry {i} in {path} is missing non-empty 'query'"
            raise ValueError(msg)
        chunk_id = entry.get("expected_chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            msg = f"entry {i} in {path} is missing non-empty 'expected_chunk_id'"
            raise ValueError(msg)
        samples.append(EvalSample(query=query.strip(), expected_chunk_id=chunk_id.strip()))
    return samples


def build_eval_dataset(
    store: StoreConnection,
    *,
    sample_size: int,
    seed: int,
) -> list[EvalSample]:
    """Build an eval dataset by sampling chunks from the corpus.

    Uses heading_text + first 100 chars of content as the pseudo-query.
    For production eval, replace with hand-curated (query, expected) pairs.

    Sampling is deterministic: the same seed always selects the same chunks,
    portable across Postgres versions (Python Mersenne Twister, not SQL random()).
    """
    with store.conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            select id, heading_text, content
            from public.chunks
            where heading_text is not null
              and length(heading_text) > 5
            """,
        )
        rows = cur.fetchall()

    rng = random.Random(seed)
    n = min(sample_size, len(rows))
    sampled = rng.sample(rows, n)

    samples: list[EvalSample] = []
    for r in sampled:
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
    If not found or beyond top-k, returns 0.0.

    The ``rank > k`` branch is unreachable from ``evaluate()`` (which calls
    ``vector_search(..., k=k)`` and so ``rank`` is always ``<= k``), but is
    retained as a library-grade invariant for any future caller that passes
    a rank from a wider candidate set.
    """
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)
