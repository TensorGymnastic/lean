# Evaluation

`lean eval` computes hit_rate@k, MRR@k, NDCG@k, and Recall@k against the
ingested corpus. The implementation is in `src/lean/core/eval/runner.py`. This
document describes what it actually measures — and what it **doesn't**
measure that is easy to misread from the metrics.

---

## What `lean eval` does

There are two modes:

**Default (pseudo-eval):**

1. **Build an eval dataset** by sampling `eval.sample_size` chunks from
   `public.chunks` (filtered to `heading_text` longer than 5 chars).
2. **Generate a pseudo-query** from each sampled chunk: the heading plus
   the first 100 chars of content. The chunk's own ID is the expected
   hit.
3. **For each sample**: embed the pseudo-query, call
   `SearchEngine.vector_search`, and check whether the expected chunk
   appears in the top-k.
4. **Aggregate** hit_rate, MRR, NDCG, and Recall into `EvalResult`,
   persisted to `public.eval_runs`.

**Curated (`--dataset <path>`):**

```bash
lean --config configs/lean-pdf-lss.yaml eval --dataset data/curated_eval_dataset.json --k 5
```

1. **Load** the JSON list of `{"query": str, "expected_chunk_id": str}`
   entries via `runner.load_curated_dataset(path)`. Extra fields
   (`heading`, `section`, `doc`) are silently dropped.
2. **For each sample**: same as default — embed the query, vector_search,
   check top-k, aggregate metrics.
3. **Aggregate** identical to default.

The curated mode uses real queries (paraphrased from chunk headings by
`scripts/build_eval_dataset.py`) instead of self-similar pseudo-queries.
It still calls `engine.vector_search` directly (not the full pipeline),
but its numbers are **closer to real-world retrieval quality** because
the query no longer matches the target chunk's own text.

---

## What the metrics actually measure

Because the query is **derived from the answer itself** (heading +
content preview), `lean eval` measures:

- **Can the embedder map a chunk back to itself?** (self-similarity)
- **Does the vector index return that chunk in top-k?** (recall of the
  ivfflat index, given a query nearly identical to the chunk text)

It does **not** measure:

| Gap | Why it matters |
|---|---|
| **Real query reformulation** | A user typing "what is DMAIC?" doesn't paste the chunk's heading. Real retrieval has a vocabulary mismatch that pseudo-eval hides. |
| **The full search pipeline** | `evaluate()` calls `engine.vector_search` directly — **no BM25, no RRF fusion, no reranker, no postprocessors**. This is not what `lean search` returns. See "Pipeline gap" below. |
| **Reranker contribution** | With `rerank.enabled: true`, real queries get reranked. Pseudo-queries (heading + content) are already so similar to the target that rerank barely moves them. Eval numbers don't reflect rerank benefit. |

### Recall@k is identical to HitRate@k in this eval

Each sample carries exactly **one** relevant item
(`expected_chunk_id`). Binary Recall with a single relevant item reduces
to "did the relevant item appear in top-k?" — which is also the
definition of HitRate@k. As a result `result.recall == result.hit_rate`
always, and only **3 of the 4 metrics are independent**
(`hit_rate==recall`, `mrr`, `ndcg`). To get a meaningful Recall that
differs from HitRate, build a curated dataset where each query has
multiple graded-relevant chunks and upgrade `EvalSample` to carry them.

---

## Pipeline gap (HIGH)

`SearchEngine.vector_search` is the **vector-only** path. The production
`services/search.py` does:

```
embed → vector + BM25 → RRF fusion → rerank → similarity filter → reorder → truncate
```

`evaluate()` calls only the first stage. **An eval improvement can mask a
regression in fusion/rerank, and vice versa.**

If you want to eval the full pipeline, you need to either:

- Build a curated dataset (real queries + expected chunk IDs) and call
  `services.search.search()` instead of `engine.vector_search`
- Add a second eval mode that exercises the full pipeline

---

## Sampling determinism (resolved)

The query now fetches all eligible chunks and samples in Python:

```python
rng = random.Random(seed)
n = min(sample_size, len(rows))
sampled = rng.sample(rows, n)
```

Python's Mersenne Twister is stable across versions and platforms, so the
same `seed` always selects the same chunks — portable across Postgres
versions, machines, and environments. Two consecutive runs on the same
corpus with the same seed produce identical hit_rate / MRR / NDCG / Recall.

Previously, SQL `ORDER BY random()` was unseeded and the Python seed only
shuffled presentation order — fixed in commit that introduced
`random.sample()` in `build_eval_dataset`.

---

## Pseudo-query construction

```python
heading = r["heading_text"].strip()
content_preview = r["content"][:100].strip().replace("\n", " ")
query = f"{heading}: {content_preview}"
```

- Only chunks with `heading_text > 5 chars` are eligible. Chunks without
  headings (e.g. "Front Matter" or short single-paragraph sections) are
  silently excluded — they don't appear in eval at all.
- The first 100 chars of content is appended. Truncation is at byte
  boundary, so multi-byte characters may split.
- Newlines are flattened to spaces.

---

## For production eval

The module docstring (`src/lean/core/eval/runner.py:6-9`) is explicit:

> The eval dataset is built by sampling chunks from the corpus and using
> their heading text + content as pseudo-queries. **For production eval,
> replace with a hand-curated set of (query, expected_chunk_id) pairs.**

A curated dataset should have:

- Real user queries (paraphrased, not copy-pasted from chunks)
- Multiple relevant chunks per query (graded relevance, not just binary)
- Coverage of queries that span multiple sections / multiple docs
- Coverage of queries that should NOT match anything (negative cases)

Then run eval against `services.search.search()` to exercise the full
pipeline.

---

## Persisted eval results

Eval results are written to `public.eval_runs` (`hit_rate`, `mean_latency_ms`,
`sample_count`, `k`, `mrr`, `ndcg`, `recall`, `created_at`). The CLI
prints the latest run; trend analysis requires direct SQL:

```sql
select created_at, hit_rate, mrr, ndcg, recall, sample_count, k
from public.eval_runs
order by created_at desc
limit 20;
```

See [`limitations.md`](limitations.md) for known eval caveats.
