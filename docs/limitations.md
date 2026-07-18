# Limitations

Known caveats — things that work but behave differently than you might
expect from the surface API. For the **why**, see `docs/decisions/` and
the inline code comments.

---

## Data correctness

### `chunks.page_start` / `chunks.page_end`

Populated by `_enrich_chunks_with_block_meta` in `services/ingestion.py` when
marker's local extraction path is used. The enrichment matches chunks to
marker blocks by word overlap and derives page numbers from
`FlatBlockOutput.page`. The **remote marker server path returns empty
block_metas** (the server hasn't been updated to return structured block data
yet), so page fields will be NULL for remote-extracted documents until
`scripts/marker_server.py` is updated and redeployed.

### Contextual Retrieval is irreversible

When `llm.contextual_retrieval: true` is set and a document is ingested,
the LLM-generated context (up to 8000 chars of doc + 1000 chars of chunk)
is **fused into the chunk content** before embedding and storage. The
original chunk text is **not preserved anywhere**.

- Disabling `contextual_retrieval` does NOT restore original text
- To undo: `lean reingest` re-runs extraction, which re-generates context
  if still enabled, or re-runs without context if disabled — but only
  if the source PDF bytes are unchanged (see dedup caveat below)
- The DB column has no flag to distinguish contextualized vs raw chunks

If you turn this on for a production corpus, plan a reingest
strategy in advance.

### Dedup is by SHA-256, not by path

`upsert_document` matches on `source_sha256`. Editing a PDF on disk
changes the SHA, so a `lean reingest` after the edit creates a **new
`document_id`** and leaves the old document row + chunks orphaned.

- Old chunks are still searchable (orphaned docs are not filtered out)
- They consume vector index slots and ivfflat cluster space
- To clean up: `lean delete <old_doc_id>`, then optionally `REINDEX`
  (see [`operations.md`](operations.md))

---

## Search & evaluation

### `lean eval` is pseudo-eval

By default, the eval dataset is built from heading + content preview, so
the query is derived from the answer. It measures self-similarity, not
real user queries. **Default eval numbers are not comparable to external
benchmarks.**

Pass `--dataset <path>` to a JSON list of `{"query": str,
"expected_chunk_id": str}` pairs (see
[`data/curated_eval_dataset.json`](../data/curated_eval_dataset.json) or
`scripts/build_eval_dataset.py`) to run evaluation against real queries
instead. Curated datasets still bypass the full search pipeline (see
below) unless the eval is rerouted through `services.search.search()` —
that is left as a project-specific caller concern.

`lean eval` (in either mode) bypasses the full search pipeline — it calls
`engine.vector_search` directly (no BM25, no RRF, no rerank, no
postprocessors). It does not measure what `lean search` returns.

See [`evaluation.md`](evaluation.md) for the full methodology.

### Eval sampling is deterministic given a fixed seed

`build_eval_dataset` fetches eligible chunks via SQL, then samples in
Python using `random.Random(seed).sample(rows, n)` (see
`src/lean/eval/runner.py:71`). Python's Mersenne Twister is stable
across versions and platforms, so the same `seed` always selects the
same chunks. Two consecutive runs on the same corpus with the same seed
produce identical hit_rate / MRR / NDCG / Recall numbers.

(Previously used unseeded SQL `ORDER BY random()`; replaced by Python
sampling.) See [`evaluation.md`](evaluation.md#sampling-determinism-resolved).

---

## Operational

### REST API now mirrors all MCP tools (full parity)

| MCP tool | REST endpoint |
|---|---|
| `ingest_pdf` | `POST /ingest` |
| `search` | `GET /search` |
| `list_documents` | `GET /documents` |
| `get_chunk` | `GET /chunks/{chunk_id}` |
| `get_document_markdown` | `GET /documents/{doc_id}/markdown` |
| `delete_document` | `DELETE /documents/{doc_id}` |
| `reingest` | ❌ not exposed (use CLI `lean reingest`) |
| `corpus_stats` | `GET /stats` |

7 of 8 MCP tools are exposed over REST. `reingest` is intentionally CLI-only
(long-running batch operation; better suited to CLI than synchronous HTTP).

### Docker compose exposes only the MCP port

`docker compose up` exposes port 8765 (MCP HTTP). The FastAPI REST API
(port 8766) is **not** mapped to the host. To use REST inside the
container, add `8766:8766` to `docker-compose.yml` or run the API on
the host (`make api-serve`).

### Docker healthcheck only verifies port open

The healthcheck creates a TCP socket to port 8765 and closes it. It
does not exercise the MCP protocol. A hung server (event-loop deadlock,
exhausted DB pool) still passes the healthcheck.

### `lean health` always exits 0

`make health` (alias `make smoke`) runs `lean health`, which checks
OCR, DB, and Ollama. It prints `[FAIL]` per-component but **never
raises `typer.Exit(1)`**. Do not use as a CI gate — wrap with grep or
parse the output.

### `lean mcp-serve` lacks `--host` flag

The CLI `lean mcp-serve --transport http` binds to
`settings.mcp_http_host` (default `127.0.0.1`). The module entrypoint
`python -m lean.mcp_server --host 0.0.0.0` does support `--host`. To
expose MCP over HTTP to non-localhost without bypassing the CLI, either:

- Set `transport.mcp_host: 0.0.0.0` in `config.yaml`, or
- Use `python -m lean.mcp_server --transport http --host 0.0.0.0`

### `lean db-init` uses `PGPASSWORD` env (not argv)

`PGPASSWORD` is set as an env var inside the Makefile to avoid
leaking the DB password to the process listing. If you call the CLI
directly, pass the password in `.env` (via `SUPABASE_DB_URL`) — never
as a CLI flag.

---

## Resource limits

### Ingest streams the PDF for SHA-256

`_sha256_streaming` (`src/lean/services/ingestion.py:39-45`) reads the
PDF in 1 MiB blocks. Memory footprint during hashing is bounded by the
block size, not the file size. The earlier "loads full PDF twice"
behavior was replaced by streaming; `max_pdf_mb` is now a size guard
checked via `pdf_path.stat().st_size` before any read.

### `executemany` for chunk inserts can hit the 65535-param limit

Postgres caps prepared statements at 65535 parameters. With 16 columns
per chunk (see `store/chunks.py:63-93`), this is hit at ~4096 chunks
per insert batch. For very large books (rare in Lean Six Sigma corpora)
this errors mid-ingest.

---

## Configuration drift

These have been observed in the codebase and may resurface:

- README previously described `make smoke` and `make health` as
  different — they are identical (`smoke: health` in the Makefile).
- README previously marketed the REST API as a "mirror" — it is partial.

Historical drift that has been resolved:
- `reranker.py` module docstring previously claimed rerank was
  "disabled by default" while `config.yaml` enabled it — the docstring
  now correctly references `retrieval.rerank.enabled: true`.
- Eval sampling previously used unseeded SQL `ORDER BY random()` —
  replaced by seeded Python `random.Random(seed).sample(...)`.
