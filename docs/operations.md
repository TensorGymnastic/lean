# Operations

Operational reference: Docker, monitoring, reindex, reingest semantics,
and resource planning. For configuration knobs, see
[`configuration.md`](configuration.md).

---

## Docker

### Ports

`docker-compose.yml` exposes only one port on the host:

| Service | Host port | Container port | Protocol |
|---|---|---|---|
| `lean-app` (MCP HTTP) | `8765` | `8765` | HTTP/SSE |

**The FastAPI REST API (port 8766) is NOT exposed by `docker compose up`.**
To use the REST API inside the container, either:

- Run `lean api-serve` locally (`make api-serve`) on the host, or
- Add `8766:8766` to `docker-compose.yml` under the `lean-app` service ports.

### Mounts

The `lean-app` service bind-mounts (read-only):

| Host path | Container path | Purpose |
|---|---|---|
| `./data` | `/app/data` | Source PDFs (must be inside `storage.corpus_root`) |
| `./src/lean/config/config.yaml` | `/app/src/lean/config/config.yaml` | App config (no hot-reload) |

**New PDFs added to host `data/` after `docker compose up` require a container
restart** to be visible, because the mount is read-only and not auto-refreshed
for already-mounted directories.

### Healthcheck

```yaml
healthcheck:
  test: [CMD-SHELL, /app/.venv/bin/python -c 'import socket; s=socket.create_connection(("127.0.0.1",8765),timeout=5); s.close()' || exit 1']
  interval: 30s
  timeout: 5s
  retries: 3
```

**The healthcheck only verifies the port is open.** It does not verify
that the MCP server is responding to requests. A hung server (event-loop
deadlock, exhausted DB pool) still passes.

### Migrations

`db/schemas/` is bind-mounted to `docker-entrypoint-initdb.d` and runs
**only on first DB init**. For subsequent schema changes, run
`make db-init` explicitly.

---

## `lean health`

`make health` (alias: `make smoke`) runs `lean health`, which checks
OCR server, database, and Ollama.

**`lean health` always exits 0**, even when all three checks fail. It
echoes `[FAIL]` per-component but never raises `typer.Exit(1)`. **Do not
use `lean health` as a CI gate** — it will silently pass on a broken
deployment. Wrap it with explicit grep or parse the output if you need
a real exit code.

---

## Post-ingest reindex

The `chunks_embedding_idx` ivfflat index builds K-means clusters at
creation time (`lists=100`). After bulk-ingesting new chunks into a
table that already has many rows, recall degrades because new vectors
fall into stale clusters. Run this after large ingest batches:

```sql
REINDEX INDEX chunks_embedding_idx;
```

This is **not** automatic. The `db/schemas/003_chunks.sql` migration has
the same advisory in a comment but no automation.

**Cost:** a 100k-chunk table takes ~30s to reindex. Run during a
maintenance window or schedule after-hours.

---

## Reingest semantics

Ingest dedups by SHA-256 of the source PDF bytes. The behavior:

| Scenario | Outcome |
|---|---|
| Same path, same bytes | No-op (returns existing `document_id`) |
| Same path, **different bytes** | New `document_id` created; **old document row and chunks remain orphaned** |
| New path, any bytes | New document |
| `lean reingest <doc_id>` | Re-reads `source_storage_path` from the existing row; if bytes have changed, new `document_id`, old doc becomes orphan |

**To delete an orphan:** `lean delete <old_doc_id>`. Use
`lean corpus-stats` to find orphans after a content edit.

This is intentional (per `src/lean/store/documents.py:upsert_document`).
Content-addressable dedup is the source of truth; the path is incidental.

---

## Silent OCR fallback

`OCRBackendUnavailable` causes the pipeline to fall back to markitdown
with `extraction_method=markitdown` in `IngestResult`. **Ingest still
returns success.** Operators watching for ingest failures will miss the
degradation.

To detect it:

- Check `IngestResult.extraction_method` after each ingest
- Monitor `IngestResult.warnings` for the fallback marker
- In CI, ingest a known PDF and assert `extraction_method` matches
  the expected backend

---

## Resource planning

| Operation | Memory | Notes |
|---|---|---|
| Ingest | `max_pdf_mb` (default 200 MB) | `pdf_path.read_bytes()` loads the full PDF for SHA-256 |
| Concurrent ingest | `n_concurrent × max_pdf_mb` | Each ingest holds the bytes in memory until the hash is computed |
| Embedding (CPU) | ~500 MB | sentence-transformers + torch |
| Embedding (GPU) | model + batch | Ollama-side; not in this process |
| DB connections | pool size = `psycopg` default | Each MCP tool call opens + closes a `StoreConnection` |

For a 200 MB PDF with concurrent ingests, plan for `n × 200 MB` of
spare RAM.

---

## Ingest → Search → Eval order

1. `make db-init` — apply migrations
2. `make ingest-all` — ingest all `data/*.pdf`
3. `REINDEX INDEX chunks_embedding_idx;` — after large ingest (see above)
4. `make search QUERY="..."` — smoke-test retrieval
5. `make eval` — measure retrieval quality (see [`evaluation.md`](evaluation.md))
