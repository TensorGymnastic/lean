# AGENTS.md — lean

## Purpose

Repository-local guidance for agents working in `lean`.

`lean` is a dockerized MCP server for Lean Six Sigma PDF corpus ingestion.
It extracts markdown from PDFs (marker-pdf primary, Unlimited-OCR secondary,
markitdown fallback), optionally enriches charts/images with a Vision-Language
Model (MiniMax M3 or local Ollama), chunks section-aware, embeds with
Liquid LMF2.5-Embedding-350M, stores in Supabase pgvector with full provenance
metadata, and exposes the corpus via fastmcp tools/resources/prompts.

## Architecture

**Layering (enforced):** transport (mcp_server / api / cli) → services → store → infrastructure.
No business logic in transports. Each store repo is CRUD-narrow.

- `src/lean/config/settings.py` — pydantic-settings: `.env` (secrets) + `config.yaml` (app config)
- `src/lean/models/schemas.py` — Pydantic types shared across modules
- `src/lean/extraction/` — PDF→markdown (`pipeline.py` orchestrator with 3-stage fallback):
  - `marker_converter.py` — **primary**: `datalab-to/marker` (surya OCR + texify), local CPU or remote GPU via HTTP
  - `unlimited_ocr.py` — **secondary**: `baidu/Unlimited-OCR` via remote transformers server (returns page images, no figures)
  - `markitdown_converter.py` — **tertiary**: pure-python fallback when both above unavailable
  - `metadata.py` — filename → `{title, authors, year, doi}` parser
  - `contextual.py` — optional LLM-based contextual retrieval augmentation
- `src/lean/vlm/` — Vision-Language Model client for chart/image description
  - `client.py` — httpx-based, OpenAI-compatible (`/v1/chat/completions`), provider-agnostic transport
  - `prompts.py` — `CHART_EXTRACTION_PROMPT` + `parse_description` JSON parser with markdown-fence handling
- `src/lean/chunker/` — `markdown_ast.py` (mistune section parser) + `recursive.py` (tiktoken splitter)
- `src/lean/embeddings/` — `liquid_lmf.py` (local CPU) + `remote_ollama.py` (remote GPU, retries 3×)
- `src/lean/infrastructure/embedder.py` — singleton factory: remote Ollama when configured, else local CPU
- `src/lean/store/` — focused repos: `base`, `documents`, `chunks`, `search`, `analytics`
- `src/lean/retrieval/` — cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) + postprocessors
- `src/lean/llm/` — OpenAI-compatible LLM client (MiniMax/Ollama), optional sidecar for HyDE/multi-query/contextual retrieval
- `src/lean/services/` — business logic: `ingestion` (extract+VLM+embed+store), `search` (hybrid+rerank), `corpus` (CRUD)
- `src/lean/mcp_server/` — 8 tools, 4 resources, 3 prompts, stdio/http entrypoint
- `src/lean/api/routes.py` — FastAPI mirror (bearer-authed REST, 5 endpoints)
- `src/lean/auth/bearer.py` — ASGI middleware, `hmac.compare_digest` token check
- `src/lean/cli.py` — Typer CLI (full parity with MCP tools)
- `src/lean/eval/` — retrieval evaluation harness (hit_rate@k, MRR@k, NDCG@k, Recall@k)
- `db/schemas/` — SQL migrations (001-012). Key migrations:
  - `010_add_chunk_types.sql` — `chunk_type` (`text`/`image`) + `image_meta` JSONB
  - `011_add_provenance_metadata.sql` — `bbox`, `image_hash`, `provenance_model`, `embedding_model`, `embedding_dim`
  - `012_add_chunk_type_check.sql` — CHECK constraint enforcing `chunk_type IN ('text','image')`
- `scripts/` — `canonical-queries.json` (eval fixture), `docs_lint.py` (drift detector)
- `docs/` — reference docs (configuration, operations, architecture, evaluation, limitations, decisions)

## Engineering Rules

- Python 3.12+, uv-managed
- Strict mypy, ruff (line-length 100, includes `S` bandit rules)
- TDD: write test first, watch it fail, implement, watch it pass
- Coverage gate: `fail_under = 80` enforced in pyproject.toml
- `from __future__ import annotations` in all modules
- Each module has one responsibility and is independently testable
- **File-size ceiling ~500 LOC, function-size ceiling ~50 LOC** — split when approaching (currently `cli.py` at 435 is the only remaining oversized file; `ingest_pdf` and `search` were split to 130/99 LOC)
- Heavy operations (DB, extraction, embedding, VLM) run via `anyio.to_thread.run_sync`
  so the MCP event loop stays responsive
- All singletons use `@lru_cache` (`get_settings`, `get_embedder`, `get_llm`,
  `_get_reranker`) — thread-safe, no mutable module globals
- Config validation at startup via pydantic `field_validator` + `model_validator`
  (API key min 16 chars, ports 1–65535, `chunk_hard_cap > chunk_target_max`)
- Model revisions pinned to SHA hashes in `config.yaml` for `trust_remote_code`
  safety (`embedding.model_revision`, `retrieval.rerank.model_revision`)
- Ingest paths confined to `settings.corpus_root` (default: `data/`) — prevents
  path traversal via authenticated API/MCP clients
- Optional ML deps isolated via `uv sync --extra`:
  - `--extra local-models` — torch/transformers/sentence-transformers (local CPU embeddings + reranker)
  - `--extra marker` — marker-pdf + surya (high-quality extraction)

## Design Discipline (KISS / YAGNI / DRY / SOLID)

- **KISS** — prefer a 30-line function over a 300-line "flexible" one. Don't add config knobs for hypothetical future needs.
- **YAGNI** — every column, config field, and parameter must have a reader within 1 commit of being written. Documented exceptions: `bbox` (wiring pending), `image_hash` (dedup query pending).
- **DRY** — shared helpers live in one place (`_build_metadata_filters`, `infrastructure/embedder.py`). No copy-paste between `vector_search` and `bm25_search`.
- **SOLID-S** — services own one orchestration concern; stores own CRUD; transports own I/O. `ingest_pdf` currently violates this (extraction+VLM+embed+store in one function) — split is on the roadmap.
- **SOLID-O** — extraction backends are pluggable via `pipeline.py`. VLM is provider-agnostic at the transport layer (any OpenAI-compatible endpoint).
- **SOLID-D** — embedder uses Protocol + factory. VLM is currently concrete (`VLMClient`); add a Protocol when a second provider shape lands.

## Validation

- `make verify` — ruff + mypy + pytest (unit only, excludes integration/slow/e2e)
  + coverage gate at 80%
- `make verify-all` — all tests
- `uv run python scripts/docs_lint.py` — detects drift between
  `config.yaml` / CLI commands / MCP tools and `docs/` + `README.md`
- 253 unit tests + integration tests + e2e tests (coverage 84.01%)
- Integration tests need `SUPABASE_DB_URL` env var set
- E2E tests need full stack running (Supabase + GPU servers)
- CI: 3 jobs (verify matrix Python 3.12+3.13, security pip-audit+trivy+CodeQL,
  integration with Postgres service container)

## Documentation

- `README.md` — onboarding entry point (~170 lines)
- `docs/configuration.md` — every `Settings` field, defaults, validators
- `docs/operations.md` — Docker, healthcheck, post-ingest reindex, reingest semantics
- `docs/architecture.md` — pipeline + directory layout + transport tier pattern
- `docs/evaluation.md` — `lean eval` methodology + caveats
- `docs/limitations.md` — known caveats (page fields NULL, eval pseudo-queries, etc.)
- `docs/decisions/` — ADR scaffolding (write new ADRs here when needed)

**Rule:** if you add a field to `config.yaml`, a CLI command, or an MCP
tool, update the corresponding doc page in the same commit. Run
`scripts/docs_lint.py` before pushing to catch drift.

## Key Decisions

- **Extraction fallback chain**: marker-pdf (local CPU or remote GPU via HTTP) → Unlimited-OCR (remote transformers) → markitdown (pure Python). Marker returns figures; OCR/markitdown do not.
- **Remote GPU marker server** — pure-Python HTTP wrapper around marker's `PdfConverter` (singleton in GPU memory). 44× faster than local CPU (14s vs 626s for a 29-page PDF). Configured via `marker.remote_url` in `config.yaml`. **Operational note:** the server script currently lives at `/tmp/marker_server.py` on the GPU host — vendor into `scripts/` before relying on it in production.
- **VLM enrichment (default: MiniMax M3)** — chart/image descriptions structured as `{title, chart_type, axis_labels, key_data_points, description, source_text}`. MiniMax M3 via API (~3.5s/image, ~$0.004/image, `disable_thinking: true` for clean JSON). Local Ollama Qwen3.5 fallback documented in `config.yaml` comments. **Compliance:** VLM sends chart images to a third-party (Shanghai) — disable for corpora with PII/trade-secret concerns; see `docs/limitations.md`.
- **Provenance metadata** — migration 011 added `bbox`, `image_hash`, `provenance_model`, `embedding_model`, `embedding_dim`. Currently written on every chunk/image but **not yet read in any WHERE clause** — `image_hash` dedup query and `embedding_model` mismatch warning are on the roadmap. `bbox` is always NULL (marker block-level polygon wiring pending).
- **Chunk types** — `text` (default) and `image` (VLM-described). Search supports `chunk_type` filter end-to-end (MCP/API/CLI → service → store). No SQL CHECK constraint yet — validation is a roadmap item.
- **LiquidAI/LFM2.5-Embedding-350M** (1024-dim), remote Ollama when configured, local CPU fallback. Pinned to SHA `f35ae2c91d687658dbf1f2b449382f0b019b9808`.
- Local Supabase via Docker (zero cloud cost)
- fastmcp v3.4.4 standalone
- Bearer token auth on HTTP transport (`hmac.compare_digest`); stdio is local-trusted
- Multi-stage Docker build with non-root user (uid=1000), `.dockerignore` excludes
  `.git/`, `.venv/`, caches, PDFs
- `lean db-init` uses `PGPASSWORD` env (not argv) — Makefile delegates to CLI

## Known Structural Debt (review-derived)

Documented honestly so future agents don't re-derive. Items here are **accepted**, not blocking. Last verified 2026-07-18 against source.

1. **`page_start` / `page_end` are hardcoded `None`** at `services/ingestion.py:231-232` (text chunks) and `:251-252` (image chunks). Chunker doesn't preserve page boundaries; cleanup needs upstream marker/OCR changes.
2. **`bbox` column is wired through but always NULL** — defined in `models/schemas.py:50`, `store/chunks.py:31`, written at `chunks.py:85`, read at `store/search.py:97,140`; never populated with non-NULL because `ChunkRow.bbox` defaults to `None` and no caller sets it. Awaits marker block-level polygon wiring.
3. **`image_hash` is written but never used in a WHERE clause** — set at `services/ingestion.py:178,258`, read into the `Chunk` model at `models/schemas.py:73`, but no dedup query exists. Future: SELECT … WHERE image_hash = … to skip duplicate figures.

**Previously listed and since FIXED (do not re-litigate):**
- `ingest_pdf` was a 244-LOC god function → split into `_describe_images` (74 LOC) + `_build_chunk_rows` (52 LOC) + `_persist_ingest` (48 LOC); `ingest_pdf` is now 130 LOC orchestration.
- `search` was a 188-LOC function mixing orchestration + business logic → split into `_validate_search_inputs` + `_expand_queries` + `_compute_fetch_k` + `_fetch_one_query` + `_rerank_hits` + `_postprocess_hits` + `SearchRequest` dataclass; `search` is now 99 LOC orchestration.
- Transaction boundary split → fixed by single-transaction pattern at `services/ingestion.py:204-271` (`commit=False` + final `conn.conn.commit()`).
- `vlm_max_concurrency` unused → wired via `anyio.Semaphore(settings.vlm_max_concurrency)` at `services/ingestion.py:155`.
- Postgres bound to `0.0.0.0:54322` → bound to `127.0.0.1:54322:5432` at `docker-compose.yml:5`.
- `chunk_type` had no CHECK constraint → added in `db/schemas/012_add_chunk_type_check.sql`; app-side validation at `services/search.py:24` (`VALID_CHUNK_TYPES`).
- `marker_server.py` lived in `/tmp/` → vendored at `scripts/marker_server.py`; deployment guide at `docs/marker-server-deployment.md`.
- `_extract_local` / `_get_converter` 0% coverage → covered by mock-module injection tests in `tests/test_marker_converter.py` (`test_get_converter_constructs_pdf_converter_when_installed`, `test_get_converter_force_ocr_true_uses_separate_cache_entry`). Coverage of `marker_converter.py`: 88% → 98%.
- `executemany` for chunk inserts could hit the 65535-param Postgres limit → batched at `_INSERT_BATCH_SIZE = 1000` in `store/chunks.py`; invariant enforced by `test_replace_chunks_batch_size_keeps_params_under_postgres_limit`.
- `vlm_provider` field was dead config → removed from `settings.py`, `config.yaml` examples, and `docs/configuration.md` table. All VLM providers use the same OpenAI-compatible transport.

See `docs/limitations.md` for runtime caveats (eval pseudo-queries, dedup orphans, partial REST mirror).
