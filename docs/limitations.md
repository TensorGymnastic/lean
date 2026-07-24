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
instead. Curated datasets are closer to real-world retrieval quality
because the query no longer matches the target chunk's own text — but
`lean eval` (in either mode) bypasses the full search pipeline,
calling `engine.vector_search` directly (no BM25, no RRF, no rerank, no
postprocessors). It does not measure what `lean search` returns.

See [`evaluation.md`](evaluation.md) for the full methodology.

### Eval sampling is deterministic given a fixed seed

`build_eval_dataset` fetches eligible chunks via SQL, then samples in
Python using `random.Random(seed).sample(rows, n)` (see
`src/lean/core/eval/runner.py:71`). Python's Mersenne Twister is stable
across versions and platforms, so the same `seed` always selects the
same chunks. Two consecutive runs on the same corpus with the same seed
produce identical hit_rate / MRR / NDCG / Recall numbers.

(Previously used unseeded SQL `ORDER BY random()`; replaced by Python
sampling.) See [`evaluation.md`](evaluation.md#sampling-determinism-resolved).

---

## Operational

### REST API now mirrors 7 of 8 MCP tools (near-full parity)

| MCP tool | REST endpoint |
|---|---|
| `ingest_pdf` | `POST /ingest` |
| `search` | `GET /search` (supports `chunk_type` filter) |
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

### `lean health` exits 1 on component failure

`make health` (alias `make smoke`) runs `lean health`, which checks
OCR server, database, and Ollama. It prints `[OK]` / `[--]` / `[FAIL]`
per component, and exits with code 1 if **any** component reports
`status: "error"` (see `src/lean/cli.py`'s `health` command). Safe to
use as a CI gate or Docker `HEALTHCHECK` precondition — but note that
the per-component check only verifies connectivity/HTTP 200, not full
pipeline correctness (e.g. a hung event loop with an open port still
passes).

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

`_sha256_streaming` (`src/lean/core/services/ingestion.py:39-45`) reads the
PDF in 1 MiB blocks. Memory footprint during hashing is bounded by the
block size, not the file size. The earlier "loads full PDF twice"
behavior was replaced by streaming; `max_pdf_mb` is now a size guard
checked via `pdf_path.stat().st_size` before any read.

### `executemany` chunk inserts are batched at 1000 rows

Postgres caps prepared statements at 65535 parameters. With 16 columns
per chunk (see `store/chunks.py:63-93`), a single `executemany` call
would hit the ceiling at ~4096 rows. This is mitigated by
`_INSERT_BATCH_SIZE = 1000` (see `store/chunks.py:15-18`), which keeps
a 4× headroom and tolerates future column additions up to ~24 columns.
Very large books in a single document still batch correctly — no
mid-ingest `prepared statement "..." has too many parameters` errors
in practice.

---

## Configuration drift

These have been observed in the codebase and may resurface:

- README previously described `make smoke` and `make health` as
  different — they are identical (`smoke: health` in the Makefile).

---

## Active backlog

Known follow-up items tracked in this section.

- **BLG-002** — `cli.py` `_output()` consistency: four commands
  (`get_markdown`, `delete`, `reingest_all`, `health`) bypass the
  unified `_output()` helper and call `typer.echo(json.dumps(...))`
  directly. Cosmetic JSON-envelope drift, not a bug.
- **BLG-007** — `embedding_model` mismatch warning: stored per-chunk
  `embedding_model` is never compared with `Settings.embedding_model`,
  so corpus model drift has no visible signal at `corpus_stats` or at
  reingest. On the AGENTS.md roadmap; warn, do not fail.
- **BLG-009** — Embedding-dim mismatch validation: query embeddings
  whose `dim` differs from stored `embedding_dim` produce nonsense
  results with no runtime guard. Log + warn + degrade gracefully.
- **BLG-010** — More per-command CLI tests to push `cli.py` ≥ 90%
  coverage (currently 85%; mostly `db_init` happy path and a handful
  of error edge cases remaining).

## Closed in this session

- **BLG-004 (2026-07-19) ✅** — M10: split `services/ingestion.py:ingest_pdf` per
  AGENTS.md:77 SOLID-S roadmap. Extracted 5 module-private helpers
  (`_validate_pdf_path`, `_extract_markdown`, `_chunk_sections`,
  `_maybe_contextualize`, `_embed_chunks`); `ingest_pdf` reduced to a
  34-line orchestrator. Five companion tests added to
  `tests/test_services_ingestion.py`; the 12 pre-existing tests
  continue to pass. `make verify`: 349 passed, 59 deselected,
  91.80% coverage. Commit `b19ed8e`.


Last reconciled against `master` HEAD `2420b58` during BLG-001.

**Audit follow-ups (2026-07-19):** the architecture validation at
`docs/audit-architecture-2026-07-19.md` produced 5 findings; remediation
tracked here as M1–M5.

- **M1 (audit 4.1) ✅** — `_find_best_block_match` Jaccard threshold
  lifted to `Settings.block_match_min_overlap` (default 0.15). Tunable
  per-corpus without code changes. Validated `(0, 1]`.
- **M2 (audit 2.1 + 2.2) ✅** — `LiquidLMFEmbedder.__init__` now
  requires `model` and `revision` (no class-level defaults).
  `unlimited_ocr.extract_markdown` now requires `model`, `dpi`,
  `timeout`, `max_tokens`, `batch_size` (no function-level defaults).
  The singleton factory in `infrastructure/embedder.py` and the
  pipeline orchestrator in `extraction/pipeline.py` were the only
  production callers and already thread from `Settings`. Two new
  contract tests assert the new "no defaults" invariant; six
  existing tests updated to pass values explicitly.
- **M3 (audit 4.3) ✅** — `vlm/client.py:107` now POSTs to
  `/v1/chat/completions` (was `/chat/completions` without `/v1`),
  matching the OCR and LLM clients. The YAML `vlm.base_url` examples
  no longer bake `/v1` into the URL (the client appends it). One new
  test `test_posts_to_v1_chat_completions` asserts the path; the
  class docstring was updated to document the new convention.
- **M4 (audit 3.1) ✅** — `CHUNK_TYPE_TEXT = "text"` and
  `CHUNK_TYPE_IMAGE = "image"` constants introduced in
  `lean/models/schemas.py` as the single source of truth for
  `chunks.chunk_type`. Replaced 4 string literal sites: `Chunk` field
  default, `Chunk.from_row` fallback, `ChunkRow` dataclass default,
  `_build_chunk_rows` image assignment. The DB-level CHECK constraint
  in migration `012` continues to enforce the same two values at the
  database layer. One new test asserts the constants and the `Chunk`
  default.
- **M5 (audit 2.3) ✅** — `Settings.mcp_http_port` and
  `Settings.api_port` no longer carry Python-side defaults (8765 and
  8766 respectively); both come exclusively from
  `config.yaml:transport.mcp_port` and `api_port`, overridable via
  `MCP_HTTP_PORT` / `API_PORT` env vars. The 22 pre-existing settings
  tests still pass (they were already exercising the YAML path, not
  the Python default). One new test asserts both the YAML default and
  the env-var override work.
