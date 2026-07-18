# AGENTS.md — lean

## Purpose

Repository-local guidance for agents working in `lean`.

`lean` is a dockerized MCP server for Lean Six Sigma PDF corpus ingestion.
It extracts markdown from PDFs (Unlimited-OCR primary, markitdown fallback),
chunks it section-aware, embeds with Liquid LMF2.5-Embedding-350M, stores in
Supabase pgvector, and exposes the corpus via fastmcp tools/resources/prompts.

## Architecture

- `src/lean/config/settings.py` — pydantic-settings: `.env` (secrets) + `config.yaml` (app config)
- `src/lean/models/schemas.py` — Pydantic types shared across modules
- `src/lean/extraction/` — PDF→markdown (marker primary, Unlimited-OCR secondary, markitdown fallback, pipeline orchestrator)
- `src/lean/chunker/` — markdown AST section parser (mistune) + recursive token splitter (tiktoken)
- `src/lean/embeddings/` — `liquid_lmf.py` (local CPU) + `remote_ollama.py` (remote GPU)
- `src/lean/infrastructure/embedder.py` — singleton factory: remote Ollama when configured, else local CPU
- `src/lean/store/` — focused repos: `base`, `documents`, `chunks`, `search`, `analytics`
- `src/lean/retrieval/` — cross-encoder reranker + postprocessors
- `src/lean/llm/` — OpenAI-compatible LLM client (MiniMax/Ollama), optional sidecar
- `src/lean/vlm/` — Vision-Language Model client for chart/image description (Ollama/vLLM/MiniMax/DashScope)
- `src/lean/services/` — business logic: `ingestion`, `search`, `corpus`
- `src/lean/mcp_server/` — 8 tools, 4 resources, 3 prompts, stdio/http entrypoint
- `src/lean/api/routes.py` — FastAPI mirror (bearer-authed REST)
- `src/lean/auth/bearer.py` — ASGI middleware for bearer token auth
- `src/lean/cli.py` — Typer CLI (full parity with MCP tools)
- `src/lean/eval/` — retrieval evaluation harness (hit_rate@k, MRR@k)
- `db/schemas/` — SQL migrations (001-011)
- `scripts/` — canonical-queries.json (eval fixture) + `docs_lint.py` (drift detector)
- `docs/` — reference docs (configuration, operations, architecture, evaluation, limitations, decisions)

## Engineering Rules

- Python 3.12+, uv-managed
- Strict mypy, ruff (line-length 100, includes `S` bandit rules)
- TDD: write test first, watch it fail, implement, watch it pass
- Coverage gate: `fail_under = 80` enforced in pyproject.toml
- `from __future__ import annotations` in all modules
- Each module has one responsibility and is independently testable
- Heavy operations (DB, extraction, embedding) run via `anyio.to_thread.run_sync`
  so the MCP event loop stays responsive
- All singletons use `@lru_cache` (`get_settings`, `get_embedder`, `get_llm`,
  `_get_reranker`) — thread-safe, no mutable module globals
- Config validation at startup via pydantic `field_validator` + `model_validator`
  (API key min 16 chars, ports 1–65535, `chunk_hard_cap > chunk_target_max`)
- Model revisions pinned to SHA hashes in `config.yaml` for `trust_remote_code`
  safety (`embedding.model_revision`, `retrieval.rerank.model_revision`)
- Ingest paths confined to `settings.corpus_root` (default: `data/`) — prevents
  path traversal via authenticated API/MCP clients
- torch/transformers/sentence-transformers are optional deps
  (`uv sync --extra local-models`); remote-only deployments skip them.
  marker-pdf is a separate optional dep (`uv sync --extra marker`).

## Validation

- `make verify` — ruff + mypy + pytest (unit only, excludes integration/slow/e2e)
  + coverage gate at 80%
- `make verify-all` — all tests
- `uv run python scripts/docs_lint.py` — detects drift between
  `config.yaml` / CLI commands / MCP tools and `docs/` + `README.md`
- 206 unit tests + integration tests + e2e tests (coverage 82.35%)
- Integration tests need `SUPABASE_DB_URL` env var set
- E2E tests need full stack running (Supabase + OCR server + models)
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

- Unlimited-OCR primary (remote transformers server, NOT vLLM), markitdown fallback
- LiquidAI/LFM2.5-Embedding-350M (1024-dim), remote Ollama when configured, local CPU fallback
- Local Supabase via Docker (zero cloud cost)
- fastmcp v3.4.4 standalone
- Bearer token auth on HTTP transport (`hmac.compare_digest`); stdio is local-trusted
- Multi-stage Docker build with non-root user (uid=1000), `.dockerignore` excludes
  `.git/`, `.venv/`, caches, PDFs
- `lean db-init` uses `PGPASSWORD` env (not argv) — Makefile delegates to CLI
