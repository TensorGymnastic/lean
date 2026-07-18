# pgvector

Vector similarity search for Postgres — used by lean via raw SQL on a
`vector(1024)` column with cosine distance.

## Indexes lean uses

| Index | Type | Use case | Tuning |
|---|---|---|---|
| `chunks_embedding_idx` | ivfflat | Approximate NN search | `lists=100` (up to ~100k chunks) |
| `chunks_content_trgm` | pg_trgm GIN | Fuzzy heading/author text search | (default) |

ivfflat builds K-means clusters at creation. After bulk-ingest into a
populated table, **run `REINDEX INDEX chunks_embedding_idx`** — see
[`operations.md`](../operations.md#post-ingest-reindex).

## Distance operators

```sql
-- Cosine distance (1 - cosine_similarity). Range: 0 (identical) to 2 (opposite).
-- Requires normalized embeddings (L2 norm = 1) for ranking to be correct.
embedding <=> query_vector

-- L2 (Euclidean) distance. Range: 0 (identical) to ∞.
-- Use for non-normalized vectors.
embedding <-> query_vector

-- Inner product (negative). Range: -∞ to ∞. Faster than cosine.
-- Higher = more similar.
embedding <#> query_vector
```

Lean uses `<=>` (cosine) because `normalize_embeddings=True` in the
embedder makes dot product equivalent to cosine similarity.

## Search tuning

```sql
-- Default: search 1 list. Fast but low recall.
SET ivfflat.probes = 1;

-- Higher recall: search more lists. Lean doesn't set this globally —
-- instead we use RRF fusion over vector + BM25 + rerank for recall.
SET ivfflat.probes = 10;
```

## Query pattern (what lean does)

```sql
SELECT id, document_id, chunk_index, content, embedding <=> %s AS distance
FROM public.chunks
WHERE embedding <=> %s < %max_distance>
ORDER BY distance
LIMIT <k>;
```

See `src/lean/store/search.py` for the actual implementation.

## Lists parameter rule of thumb

| Rows | Recommended `lists` |
|---|---|
| < 10k | 10 |
| 10k–100k | 100 (lean's choice) |
| 100k–1M | 1000 |
| > 1M | sqrt(rows) or use HNSW instead |

## Gotchas

- **ivfflat recall degrades after bulk ingest** without `REINDEX`
  (clusters go stale). See [`operations.md`](../operations.md#post-ingest-reindex).
- **`executemany` hits the 65535-param limit** at ~7281 chunks per batch
  (9 columns). See [`limitations.md`](../limitations.md).
- **HNSW is generally better for high-recall needs** but uses more
  memory and slower builds. ivfflat is fine for lean's corpus size.

## Resources

- Docs: <https://github.com/pgvector/pgvector>
- lean schema: `db/schemas/003_chunks.sql`
