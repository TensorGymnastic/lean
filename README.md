# lean

An MCP server for ingesting Lean Six Sigma PDFs into a searchable knowledge base.
Uses vision-based OCR for high-quality extraction, section-aware chunking,
GPU-accelerated embeddings, and hybrid BM25 + vector search via pgvector.

## Features

- **Vision-based PDF extraction** — `datalab-to/marker` (surya OCR + texify) as primary backend with proper table/equation/heading formatting, with `markitdown` fallback when marker is not installed
- **Section-aware chunking** — mistune AST parser splits markdown by headings, then a recursive tiktoken-based splitter bounds chunks to a target token window
- **GPU-accelerated embeddings** — LiquidAI/LFM2.5-Embedding-350M (1024-dim) served via Ollama on the GPU server, with automatic local CPU fallback
- **Hybrid search** — BM25 full-text (PostgreSQL tsvector) fused with pgvector cosine similarity via Reciprocal Rank Fusion (RRF, k=60)
- **Cross-encoder reranking** — fetch wide candidate set, rerank with `ms-marco-MiniLM-L-6-v2`, return top-k
- **MCP server** — 8 tools, 4 resources, 3 prompts exposed over stdio or HTTP, plus a partial REST mirror (5 endpoints)
- **Retrieval evaluation** — `lean eval` command computing hit_rate@k, MRR@k, NDCG@k, Recall@k (see [`docs/evaluation.md`](docs/evaluation.md) for caveats)
- **Optional LLM sidecar** — Contextual Retrieval, HyDE, multi-query generation — all opt-in, pipeline works without LLM
- **Security hardening** — corpus-root path confinement, API key validation (min 16 chars, `change-me` rejected), HuggingFace model revisions pinned to SHA hashes, non-root Docker user, multi-stage build
- **Optional ML deps** — torch/transformers/sentence-transformers only when local CPU embeddings are needed (`uv sync --extra local-models`); marker-pdf for high-quality extraction (`uv sync --extra marker`)

## Tech Stack

| Layer | Technology |
|---|---|
| MCP | fastmcp v3.4.4 |
| OCR | `baidu/Unlimited-OCR` via transformers (remote GPU) |
| Embeddings | LiquidAI/LFM2.5-Embedding-350M via Ollama (remote GPU) |
| Search | pgvector cosine + PostgreSQL tsvector BM25 + RRF fusion |
| Storage | Supabase (Postgres 15 + pgvector) via Docker |
| Framework | Python 3.12+, uv-managed, strict mypy + ruff |

## Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/)
- **Docker** (for local Supabase)
- **Remote GPU server** (NVIDIA, 12GB+ VRAM) running:
  - OCR server: `baidu/Unlimited-OCR` served via transformers on port 8000
  - Ollama: `lfm2.5-embed-32k` model (LFM2.5-Embedding-350M, 32K context) on port 11434

> The OCR and embedding servers are optional — `lean` falls back to markitdown extraction and local CPU embeddings when they are not configured.

## Quick Start

```bash
git clone <repo-url> && cd lean

# 1. Install dependencies
uv sync --all-groups
# (optional) Local CPU embeddings + reranker (~2GB torch):
uv sync --extra local-models

# 2. Install git hooks
make hooks-install

# 3. Configure secrets
cp .env.example .env  # set SUPABASE_DB_URL, LEAN_MCP_API_KEY (min 16 chars, not 'change-me')

# 4. Start local database
docker compose up -d supabase-db
make db-init

# 5. Ingest PDFs
make ingest-all

# 6. Search
make search QUERY="What is DMAIC?"
```

For the remote GPU server (OCR + Ollama setup), see
[`docs/ocr-server-deployment.md`](docs/ocr-server-deployment.md).

## Commands

### CLI (`lean`)

All commands support `--json` for structured output. Use `-v` / `--verbose` for debug logging.

| Command | Description |
|---|---|
| `lean ingest <path>` | Ingest a PDF into the corpus |
| `lean search "<query>"` | Semantic + hybrid search (`--k`, `--doc-id`, `--section`, `--author`, `--year-min`, `--year-max`, `--min-score`) |
| `lean list-documents` | List all documents in the corpus |
| `lean get-chunk <chunk_id>` | Retrieve a single chunk by UUID |
| `lean get-markdown <doc_id>` | Get extracted markdown for a document |
| `lean delete <doc_id>` | Delete a document and all its chunks |
| `lean reingest <doc_id>` | Re-extract a document with current settings |
| `lean reingest-all` | Batch reingest all documents (`--force` to re-extract OCR'd docs) |
| `lean eval` | Run retrieval evaluation (`--sample-size`, `--k`) |
| `lean health` | Check OCR server, database, and Ollama connectivity |
| `lean mcp-serve` | Start the MCP server (`--transport stdio\|http`, `--port`) |
| `lean api-serve` | Start the FastAPI REST API server (`--reload` for dev) |
| `lean db-init` | Apply all SQL migrations to the database |

### Makefile

| Target | Description |
|---|---|
| `make verify` | format-check + lint + typecheck + unit tests |
| `make verify-all` | all tests (incl. integration/e2e) |
| `make format` / `make format-check` / `make lint` / `make typecheck` | individual checks |
| `make db-init` | apply all SQL migrations |
| `make db-reset` | `supabase db reset` (destroys data) |
| `make ingest-all` / `make ingest-one FILE=…` | ingest PDFs |
| `make search QUERY="…"` | search from the command line |
| `make mcp-serve` / `make mcp-serve-http` | start MCP server (stdio / HTTP) |
| `make api-serve` | start FastAPI REST mirror (port 8766) |
| `make health` / `make smoke` | check OCR server, database, Ollama (`smoke` is an alias for `health`) |
| `make build` / `make up` / `make down` | Docker lifecycle |

## Usage

### MCP Client (Claude Desktop, opencode, etc.)

```json
{
  "mcpServers": {
    "lean": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/lean", "lean", "mcp-serve"]
    }
  }
}
```

**8 tools:** `ingest_pdf`, `search`, `get_chunk`, `list_documents`,
`get_document_markdown`, `delete_document`, `reingest`, `corpus_stats`

**4 resources:** `lean://documents`, `lean://documents/{id}/markdown`,
`lean://documents/{id}/chunks`, `lean://stats`

**3 prompts:** `lean_qa`, `lean_glossary`, `lean_compare_concepts`

### REST API (partial mirror)

```bash
make api-serve   # http://localhost:8766

curl -H "Authorization: Bearer $LEAN_MCP_API_KEY" \
     "http://localhost:8766/search?query=What+is+DMAIC%3F&k=5"
```

Endpoints (5/8 MCP tools — see [`docs/limitations.md`](docs/limitations.md#rest-api-is-a-partial-mirror-not-full-parity)):

| Endpoint | MCP equivalent |
|---|---|
| `GET /health` | (no auth, always 200) |
| `GET /search` | `search` |
| `GET /documents` | `list_documents` |
| `GET /stats` | `corpus_stats` (subset) |
| `POST /ingest` | `ingest_pdf` |

Domain errors map to HTTP codes: `ValueError` → 400, `PermissionError` → 403,
`FileNotFoundError` → 404.

## Documentation

- **[`docs/`](docs/)** — reference docs (configuration, operations, architecture, evaluation, limitations)
- **[`docs/configuration.md`](docs/configuration.md)** — every `Settings` field, defaults, validators, gotchas
- **[`docs/operations.md`](docs/operations.md)** — Docker ports, healthcheck semantics, post-ingest reindex, reingest semantics
- **[`docs/architecture.md`](docs/architecture.md)** — pipeline, directory layout, transport tier pattern
- **[`docs/limitations.md`](docs/limitations.md)** — known caveats (page fields NULL, eval pseudo-queries, dedup orphans, REST partial mirror)
- **[`docs/decisions/`](docs/decisions/)** — Architecture Decision Records
- **[`AGENTS.md`](AGENTS.md)** — operational rules for agents

## License

MIT for project code. See model licenses for third-party weights (`baidu/Unlimited-OCR`, `LiquidAI/LFM2.5-Embedding-350M`).
