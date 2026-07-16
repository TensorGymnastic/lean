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
- `src/lean/extraction/` — PDF→markdown (Unlimited-OCR remote server, markitdown fallback, pipeline orchestrator)
- `src/lean/chunker/` — markdown AST section parser (mistune) + recursive token splitter (tiktoken)
- `src/lean/embeddings/` — `liquid_lmf.py` (local CPU) + `remote_ollama.py` (remote GPU)
- `src/lean/infrastructure/embedder.py` — singleton factory: remote Ollama when configured, else local CPU
- `src/lean/store/` — focused repos: `base`, `documents`, `chunks`, `search`, `analytics`
- `src/lean/retrieval/` — cross-encoder reranker + postprocessors
- `src/lean/services/` — business logic: `ingestion`, `search`, `corpus`
- `src/lean/mcp_server/` — 8 tools, 4 resources, 3 prompts, stdio/http entrypoint
- `src/lean/api/routes.py` — FastAPI mirror (bearer-authed REST)
- `src/lean/auth/bearer.py` — ASGI middleware for bearer token auth
- `src/lean/cli.py` — Typer CLI (full parity with MCP tools)
- `src/lean/eval/` — retrieval evaluation harness (hit_rate@k, MRR@k)
- `db/schemas/` — SQL migrations (001-006)
- `scripts/` — reingest-all, smoke checks, GPU server setup

## Engineering Rules

- Python 3.12+, uv-managed
- Strict mypy, ruff (line-length 100)
- TDD: write test first, watch it fail, implement, watch it pass
- `from __future__ import annotations` in all modules
- Each module has one responsibility and is independently testable
- Heavy operations (DB, extraction, embedding) run via `anyio.to_thread.run_sync`
  so the MCP event loop stays responsive

## Validation

- `make verify` — ruff + mypy + pytest (unit only, excludes integration/slow/e2e)
- `make verify-all` — all tests
- Integration tests need `SUPABASE_DB_URL` env var set
- E2E tests need full stack running (Supabase + OCR server + models)

## Key Decisions

- Unlimited-OCR primary (remote transformers server, NOT vLLM), markitdown fallback
- LiquidAI/LFM2.5-Embedding-350M (1024-dim), remote Ollama when configured, local CPU fallback
- Local Supabase via Docker (zero cloud cost)
- fastmcp v3.4.4 standalone
- Bearer token auth on HTTP transport; stdio is local-trusted
