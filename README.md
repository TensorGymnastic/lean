# lean

An MCP server for ingesting Lean Six Sigma PDFs into a searchable multimodal
knowledge base. Uses marker-pdf for high-quality extraction (with optional
remote GPU acceleration), Vision-Language Model enrichment for charts and
figures, section-aware chunking, GPU-accelerated embeddings, provenance-tracked
storage, and hybrid BM25 + vector search via pgvector.

## Features

- **Marker-pdf extraction (primary)** — `datalab-to/marker` (surya OCR + texify) with proper table/equation/heading formatting and figure extraction. Runs locally on CPU or on a remote GPU server via HTTP (44× faster — 14s vs 626s for a 29-page PDF). Falls back to Unlimited-OCR (remote transformers) then markitdown (pure Python) when unavailable.
- **VLM chart/image enrichment** — Vision-Language Model (MiniMax M3 default, Ollama Qwen3.5 fallback) describes every chart, diagram, and figure at ingest time. Descriptions are structured (`title`, `chart_type`, `axis_labels`, `key_data_points`, `description`) and embedded alongside text, making visual content searchable. **~3.5s/image, ~$0.004/image via MiniMax M3 API.**
- **Provenance metadata** — every chunk tracks `embedding_model`, `embedding_dim`; every image chunk tracks `image_hash`, `provenance_model` (which VLM described it). Enables model-version auditing and future dedup.
- **Chunk-type filter** — search returns `text` and `image` chunks; filter by `chunk_type="image"` to surface only charts/figures.
- **Section-aware chunking** — mistune AST parser splits markdown by headings, then a recursive tiktoken-based splitter bounds chunks to a target token window
- **GPU-accelerated embeddings** — LiquidAI/LFM2.5-Embedding-350M (1024-dim) served via Ollama on the GPU server, with automatic local CPU fallback
- **Hybrid search** — BM25 full-text (PostgreSQL tsvector) fused with pgvector cosine similarity via Reciprocal Rank Fusion (RRF, k=60)
- **Cross-encoder reranking** — fetch wide candidate set, rerank with `ms-marco-MiniLM-L-6-v2`, return top-k
- **MCP server** — 8 tools, 4 resources, 3 prompts exposed over stdio or HTTP, plus a REST mirror (7 endpoints)
- **Retrieval evaluation** — `lean eval` command computing hit_rate@k, MRR@k, NDCG@k, Recall@k (see [`docs/evaluation.md`](docs/evaluation.md) for caveats)
- **Optional LLM sidecar** — Contextual Retrieval, HyDE, multi-query generation — all opt-in, pipeline works without LLM
- **Security hardening** — corpus-root path confinement, API key validation (min 16 chars, `change-me` rejected), HuggingFace model revisions pinned to SHA hashes, non-root Docker user, multi-stage build
- **Optional ML deps** — torch/transformers/sentence-transformers only when local CPU embeddings are needed (`uv sync --extra local-models`); marker-pdf for high-quality extraction (`uv sync --extra marker`)

## Tech Stack

| Layer | Technology |
|---|---|
| MCP | fastmcp v3.4.4 |
| Extraction (primary) | `datalab-to/marker` (surya OCR + texify) — local CPU or remote GPU via HTTP |
| Extraction (fallback) | `baidu/Unlimited-OCR` via transformers (remote GPU), then `markitdown` |
| VLM | MiniMax M3 via API (default), or Ollama Qwen3.5 (local fallback) |
| Embeddings | LiquidAI/LFM2.5-Embedding-350M (1024-dim) via Ollama (remote GPU) |
| Search | pgvector cosine + PostgreSQL tsvector BM25 + RRF fusion |
| Storage | Supabase (Postgres 15 + pgvector) via Docker |
| Framework | Python 3.12+, uv-managed, strict mypy + ruff |

## Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/)
- **Docker** (for local Supabase)
- **Remote GPU server** (NVIDIA, 12GB+ VRAM) running:
  - **Marker server**: pure-Python HTTP wrapper around marker's `PdfConverter` on port 8000 (44× faster than CPU)
  - **Ollama**: `lfm2.5-embed-32k` model (LFM2.5-Embedding-350M, 32K context) on port 11434
  - *(optional)* **Unlimited-OCR server**: `baidu/Unlimited-OCR` via transformers on port 8001 (secondary extraction fallback)

> All GPU services are optional — `lean` falls back to local marker-pdf (CPU), local CPU embeddings, and markitdown when remote servers are not configured. VLM enrichment is also optional (disable via `vlm.enabled: false`).

## Quick Start

```bash
git clone <repo-url> && cd lean

# 1. Install dependencies
uv sync --all-groups
uv sync --extra marker          # marker-pdf for high-quality extraction
# (optional) Local CPU embeddings + reranker (~2GB torch):
uv sync --extra local-models

# 2. Install git hooks
make hooks-install

# 3. Configure secrets
cp .env.example .env  # set SUPABASE_DB_URL, LEAN_MCP_API_KEY (min 16 chars, not 'change-me')
# (optional) VLM_API_KEY for MiniMax, MARKER_REMOTE_URL for GPU marker server

# 4. Start local database
docker compose up -d supabase-db
make db-init

# 5. Ingest PDFs
make ingest-all

# 6. Search (text + image chunks)
make search QUERY="What is DMAIC?"
make search QUERY="Pareto chart of defects"   # surfaces VLM-described images
```

For the remote GPU server (marker + Ollama + optional OCR setup), see
[`docs/ocr-server-deployment.md`](docs/ocr-server-deployment.md).

## Commands

### CLI (`lean`)

All commands support `--json` for structured output. Use `-v` / `--verbose` for debug logging.

| Command | Description |
|---|---|
| `lean ingest <path>` | Ingest a PDF into the corpus (uses marker → OCR → markitdown fallback chain, optional VLM enrichment) |
| `lean search "<query>"` | Semantic + hybrid search (`--k`, `--doc-id`, `--section`, `--author`, `--year-min`, `--year-max`, `--min-score`, `--chunk-type text\|image`) |
| `lean list-documents` | List all documents in the corpus |
| `lean get-chunk <chunk_id>` | Retrieve a single chunk by UUID |
| `lean get-markdown <doc_id>` | Get extracted markdown for a document |
| `lean delete <doc_id>` | Delete a document and all its chunks |
| `lean reingest <doc_id>` | Re-extract a document with current settings |
| `lean reingest-all` | Batch reingest all documents (`--force` to re-extract OCR'd docs) |
| `lean eval` | Run retrieval evaluation (`--sample-size`, `--k`, `--dataset <path>` for curated queries) |
| `lean health` | Check marker server, OCR server, database, and Ollama connectivity |
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
| `make health` / `make smoke` | check marker server, OCR server, database, Ollama (`smoke` is an alias for `health`) |
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

### REST API (near-full mirror)

```bash
make api-serve   # http://localhost:8766

curl -H "Authorization: Bearer $LEAN_MCP_API_KEY" \
     "http://localhost:8766/search?query=What+is+DMAIC%3F&k=5"
```

Endpoints (7/8 MCP tools — `reingest` is CLI-only; see [`docs/limitations.md`](docs/limitations.md#rest-api-now-mirrors-7-of-8-mcp-tools-near-full-parity)):

| Endpoint | MCP equivalent |
|---|---|
| `GET /health` | (no auth, always 200) |
| `GET /search` | `search` (supports `chunk_type` filter) |
| `GET /documents` | `list_documents` |
| `GET /documents/{id}/markdown` | `get_document_markdown` |
| `DELETE /documents/{id}` | `delete_document` |
| `GET /chunks/{chunk_id}` | `get_chunk` |
| `GET /stats` | `corpus_stats` |
| `POST /ingest` | `ingest_pdf` |

Domain errors map to HTTP codes: `ValueError` → 400, `PermissionError` → 403,
`FileNotFoundError` → 404.

## Documentation

- **[`docs/`](docs/)** — reference docs (configuration, operations, architecture, evaluation, limitations)
- **[`docs/configuration.md`](docs/configuration.md)** — every `Settings` field, defaults, validators, gotchas
- **[`docs/operations.md`](docs/operations.md)** — Docker ports, healthcheck semantics, post-ingest reindex, reingest semantics
- **[`docs/architecture.md`](docs/architecture.md)** — pipeline, directory layout, transport tier pattern
- **[`docs/limitations.md`](docs/limitations.md)** — known caveats (page fields on remote path, eval pseudo-queries, SHA-256 dedup orphans, REST near-full mirror)
- **[`docs/decisions/`](docs/decisions/)** — Architecture Decision Records
- **[`AGENTS.md`](AGENTS.md)** — operational rules for agents

## License

MIT for project code. See model licenses for third-party weights (`datalab-to/marker`, `baidu/Unlimited-OCR`, `LiquidAI/LFM2.5-Embedding-350M`, MiniMax M3).

For known caveats and runtime behavior, see [`docs/limitations.md`](docs/limitations.md) and [`AGENTS.md`](AGENTS.md).
