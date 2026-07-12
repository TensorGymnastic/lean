# AGENTS.md — lean

## Purpose

Repository-local guidance for agents working in `lean`.

`lean` is a dockerized MCP server for Lean Six Sigma PDF corpus ingestion.
It extracts markdown from PDFs (Unlimited-OCR primary, markitdown fallback),
chunks it section-aware, embeds with Liquid LMF2.5-Embedding-350M, stores in
Supabase pgvector, and exposes the corpus via fastmcp tools/resources/prompts.

## Architecture

- `src/lean/settings.py` — pydantic-settings, all env-driven config
- `src/lean/models/schemas.py` — Pydantic types shared across modules
- `src/lean/extraction/` — PDF→markdown (markitdown fallback, Unlimited-OCR via vLLM, pipeline orchestrator)
- `src/lean/chunker/` — markdown AST section parser (mistune) + recursive token splitter (tiktoken)
- `src/lean/embeddings/liquid_lmf.py` — sentence-transformers wrapper for LFM2.5-Embedding-350M
- `src/lean/store/pgvector.py` — direct Postgres+pgvector CRUD and cosine search
- `src/lean/retrieval/search.py` — embed query → pgvector top-k
- `src/lean/mcp_server/` — 8 tools, 4 resources, 3 prompts, stdio/http entrypoint
- `src/lean/api/routes.py` — FastAPI mirror (bearer-authed REST)
- `src/lean/auth/bearer.py` — ASGI middleware for bearer token auth
- `src/lean/cli.py` — Typer CLI (ingest, search, mcp-serve, db-init)
- `db/schemas/` — SQL migrations (extensions, documents, chunks)
- `supabase/` — Supabase CLI config + seed (storage buckets)
- `scripts/` — smoke checks, 3080 Ti setup guide

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
- E2E tests need full stack running (Supabase + vLLM)

## Key Decisions

See `docs/superpowers/specs/2026-07-12-lean-mcp-pipeline-design.md` §9 for the
full decision log. Critical ones:

- Unlimited-OCR primary (vLLM on remote 3080 Ti), markitdown fallback
- LiquidAI/LFM2.5-Embedding-350M (1024-dim, asymmetric prompts)
- Local Supabase via CLI (zero cloud cost)
- fastmcp v3.4.4 standalone
- Bearer token auth on HTTP transport; stdio is local-trusted
