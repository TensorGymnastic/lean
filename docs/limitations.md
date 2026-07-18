# Limitations

Known caveats — things that work but behave differently than you might
expect from the surface API. For the **why**, see `docs/decisions/` and
the inline code comments.

---

## Data correctness

### `chunks.page_start` / `chunks.page_end` are always NULL

The columns exist in the schema (`db/schemas/003_chunks.sql:10-11`) and
the MCP resource `lean://documents/{id}/chunks` exposes them. **Ingestion
hardcodes both to `None`** in `src/lean/services/ingestion.py:155-156`.
The OCR client returns per-page markdown but we do not currently track
which chunks came from which pages.

If you need page-anchored citation, this needs implementation work
(upstream OCR client would need to return page boundaries, and the
chunker would need to preserve them).

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

### `agent_id` analytics is dormant

`public.query_logs.agent_id` exists, `services/search.search(agent_id=...)`
accepts it, and `AnalyticsRepo.log_query` writes it. **No caller (CLI,
MCP, REST) populates it.** Every analytics row has `agent_id = NULL`.

To wire it: each transport needs to accept an `agent_id` arg (from CLI
flag, MCP client ID, or REST header) and pass it through. Not implemented.

---

## Search & evaluation

### `lean eval` is pseudo-eval

The eval dataset is built from heading + content preview, so the query
is derived from the answer. It measures self-similarity, not real user
queries. **Eval numbers are not comparable to external benchmarks.**

`lean eval` also bypasses the full search pipeline — it calls
`engine.vector_search` directly (no BM25, no RRF, no rerank, no
postprocessors). It does not measure what `lean search` returns.

See [`evaluation.md`](evaluation.md) for the full methodology.

### Eval sampling is non-deterministic

`build_eval_dataset` uses `ORDER BY random()` in SQL, which is
unseeded. The `seed=42` only shuffles the already-fetched rows. Two
consecutive runs sample different chunks and produce different
hit_rate / MRR / NDCG / Recall numbers, even on the same corpus.

---

## Operational

### REST API is a partial mirror, not full parity

| MCP tool | REST endpoint |
|---|---|
| `ingest_pdf` | `POST /ingest` |
| `search` | `GET /search` |
| `list_documents` | `GET /documents` |
| `get_chunk` | ❌ not exposed |
| `get_document_markdown` | ❌ not exposed |
| `delete_document` | ❌ not exposed |
| `reingest` | ❌ not exposed |
| `corpus_stats` | `GET /stats` (subset) |

This is intentional scope — REST is for HTTP monitoring / scripting,
not a full client surface. To add an endpoint, see
[`architecture.md`](architecture.md) on the transport tier pattern.

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

### Ingest loads the entire PDF into memory

`pdf_path.read_bytes()` is called once for SHA-256 hashing and then
again for extraction. A `max_pdf_mb=200` PDF occupies ~400 MB during
ingest. Concurrent ingests multiply this linearly.

### `executemany` for chunk inserts can hit the 65535-param limit

Postgres caps prepared statements at 65535 parameters. With 9 columns
per chunk, this is hit at ~7281 chunks per insert batch. For very large
books (rare in Lean Six Sigma corpora) this errors mid-ingest.

---

## Configuration drift

These have been observed in the codebase and may resurface:

- `reranker.py` module docstring claims rerank is "disabled by default",
  but `config.yaml:38` enables it. Trust the config, not the docstring.
- README previously described `make smoke` and `make health` as
  different — they are identical (`smoke: health` in the Makefile).
- README previously marketed the REST API as a "mirror" — it is partial.
