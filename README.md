# lean

An MCP server for ingesting Lean Six Sigma PDFs into a searchable knowledge base.
Uses vision-based OCR for high-quality extraction, section-aware chunking,
GPU-accelerated embeddings, and hybrid BM25 + vector search via pgvector.

## Features

- **Vision-based PDF extraction** — `baidu/Unlimited-OCR` via a remote GPU server, with `markitdown` fallback when the OCR server is unavailable
- **Section-aware chunking** — mistune AST parser splits markdown by headings, then a recursive tiktoken-based splitter bounds chunks to a target token window
- **GPU-accelerated embeddings** — LiquidAI/LFM2.5-Embedding-350M (1024-dim) served via Ollama on the GPU server, with automatic local CPU fallback
- **Hybrid search** — BM25 full-text (PostgreSQL tsvector) fused with pgvector cosine similarity via Reciprocal Rank Fusion (RRF, k=60)
- **Cross-encoder reranking** — fetch wide candidate set (fetch_k = 8×top_k), rerank with `ms-marco-MiniLM-L-6-v2`, return top-k
- **MCP server** — 8 tools, 4 resources, 3 prompts exposed over stdio or HTTP (FastAPI REST mirror included)
- **Retrieval evaluation** — built-in `lean eval` command computing hit_rate@k, MRR@k, NDCG@k, and Recall@k, with results persisted for trending
- **Optional LLM sidecar** — Contextual Retrieval (Anthropic technique), HyDE, and multi-query generation via MiniMax or local Ollama — all opt-in, pipeline works without LLM
- **Thin transport layers** — CLI, MCP server, and REST API are all thin delegates to a shared services layer; no business logic in the transport tier

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

## Installation

```bash
git clone <repo-url> && cd lean

# 1. Install dependencies
uv sync --all-groups

# 2. Install git hooks
make hooks-install

# 3. Configure secrets
cp .env.example .env
# Edit .env: set SUPABASE keys, HF_TOKEN, LEAN_MCP_API_KEY

# 4. Configure remote servers (optional — edit if using remote GPU)
#    Edit src/lean/config/config.yaml:
#      ocr.base_url: http://<gpu-host>:8000
#      embedding.remote_url: http://<gpu-host>:11434

# 5. Start local database
docker compose up -d supabase-db

# 6. Apply database migrations
make db-init

# 7. Ingest PDFs
uv run lean ingest data/*.pdf
# Or: make ingest-all
```

### Remote GPU Server Setup

See `docs/ocr-server-deployment.md` for the full guide. Summary:

```bash
# OCR server (Unlimited-OCR via transformers)
# On the GPU server — requires no_repeat_ngram_size=3 in model.infer_multi()
mkdir -p ~/ocr-serve && cd ~/ocr-serve
uv init && uv add "transformers>=4.57.1,<5" torch accelerate PyMuPDF \
  fastapi uvicorn pillow torchvision addict matplotlib easydict einops
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('baidu/Unlimited-OCR')"
# Create server.py — see docs/ocr-server-deployment.md
sudo systemd-run uv run python server.py  # serves on :8000

# Ollama (LFM2.5 embeddings with 32K context)
ollama create lfm2.5-embed-32k -f - << 'EOF'
FROM hf.co/LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0
PARAMETER num_ctx 32768
EOF
```

## File and Folder Structure

```
lean/
├── src/lean/
│   ├── config/
│   │   ├── settings.py          # pydantic-settings: .env (secrets) + config.yaml (app config)
│   │   └── config.yaml          # app config (OCR/embedding URLs, chunk sizes, search params)
│   ├── infrastructure/
│   │   └── embedder.py          # singleton factory: remote Ollama when configured, else local CPU
│   ├── services/                # business logic (thin transport delegates here)
│   │   ├── ingestion.py         # PDF → extract → chunk → embed → store
│   │   ├── search.py            # hybrid BM25 + vector search with RRF fusion
│   │   └── corpus.py            # list, get, delete, stats
│   ├── store/                   # focused Postgres repositories
│   │   ├── base.py              # StoreConnection wrapper
│   │   ├── documents.py         # DocumentRepo (CRUD)
│   │   ├── chunks.py            # ChunkRepo (upsert + replace)
│   │   ├── search.py            # SearchEngine (vector + hybrid)
│   │   └── analytics.py         # AnalyticsRepo (query logs, eval runs, corpus stats)
│   ├── extraction/              # PDF → markdown
│   │   ├── unlimited_ocr.py     # OCR client (batched, OpenAI-compatible API)
│   │   ├── ocr_postprocess.py   # annotation stripper + empty paragraph filter
│   │   ├── markitdown_fallback.py
│   │   ├── metadata.py          # PDF metadata extraction (title, authors, year)
│   │   └── pipeline.py          # orchestrator: OCR → fallback
│   ├── chunker/                 # markdown → token-bounded chunks
│   │   ├── markdown_ast.py      # mistune section parser
│   │   └── recursive.py         # tiktoken recursive splitter
│   ├── embeddings/
│   │   ├── liquid_lmf.py        # local CPU embedder (sentence-transformers)
│   │   └── remote_ollama.py     # remote Ollama embedder (GPU)
│   ├── retrieval/
│   │   ├── reranker.py          # cross-encoder reranker (optional)
│   │   └── postprocessors.py    # similarity cutoff, long-context reorder
│   ├── eval/
│   │   └── runner.py            # hit_rate@k, MRR@k harness
│   ├── mcp_server/              # MCP transport (8 tools, 4 resources, 3 prompts)
│   ├── api/                     # FastAPI REST mirror (bearer-authed)
│   ├── auth/bearer.py           # ASGI bearer token middleware
│   ├── models/schemas.py        # shared Pydantic types
│   └── cli.py                   # Typer CLI (full parity with MCP tools)
├── db/schemas/                  # SQL migrations (001-006)
├── scripts/                     # canonical-queries.json (eval data)
├── docker-compose.yml           # Supabase DB + lean-app
├── Makefile                     # dev commands
└── pyproject.toml               # uv project config
```

## Commands

### CLI (`lean`)

All commands support `--json` for structured output.

| Command | Description |
|---|---|
| `lean ingest <path>` | Ingest a PDF into the corpus |
| `lean search "<query>"` | Semantic + hybrid search (`--k`, `--doc-id`, `--section`, `--author`, `--year-min`, `--year-max`, `--min-score`) |
| `lean list-documents` | List all documents in the corpus |
| `lean corpus-stats` | Show document/chunk counts and extraction breakdown |
| `lean get-chunk <chunk_id>` | Retrieve a single chunk by UUID |
| `lean get-markdown <doc_id>` | Get extracted markdown for a document |
| `lean delete <doc_id>` | Delete a document and all its chunks |
| `lean reingest <doc_id>` | Re-extract a document with current settings |
| `lean reingest-all` | Batch reingest all documents (`--force` to re-extract OCR'd docs) |
| `lean eval` | Run retrieval evaluation (`--sample-size`, `--k`) |
| `lean health` | Check OCR server, database, and Ollama connectivity |
| `lean mcp-serve` | Start the MCP server (`--transport stdio\|http`, `--port`) |
| `lean db-init` | Apply all SQL migrations to the database |

### Makefile

| Target | Description |
|---|---|
| `make verify` | format-check + lint + typecheck + unit tests |
| `make verify-all` | format-check + lint + typecheck + all tests (incl. integration/e2e) |
| `make format` / `make format-check` / `make lint` / `make typecheck` | individual checks |
| `make db-init` | apply all SQL migrations |
| `make ingest-all` | ingest all `data/*.pdf` |
| `make search QUERY="..."` | search from the command line |
| `make mcp-serve` | start MCP server (stdio) |
| `make api-serve` | start FastAPI REST mirror (port 8766) |
| `make smoke` | health checks: OCR server + pgvector |
| `make health` | check OCR server, database, and Ollama (`lean health`) |
| `make build` / `make up` / `make down` | Docker lifecycle |

> GPU server setup is documented in `docs/ocr-server-deployment.md`.

## Usage

### MCP Client (Claude Desktop, opencode, etc.)

Add to your MCP client config:

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

All CLI operations are Python — no bash scripts for core functionality.

**8 tools:** `ingest_pdf`, `search`, `get_chunk`, `list_documents`,
`get_document_markdown`, `delete_document`, `reingest`, `corpus_stats`

**Search filters:** `query`, `k`, `doc_id`, `section`, `author`,
`year_min`, `year_max`, `min_score`

**4 resources:** `lean://documents`, `lean://documents/{id}/markdown`,
`lean://documents/{id}/chunks`, `lean://stats`

**3 prompts:** `lean_qa`, `lean_glossary`, `lean_compare_concepts`

### REST API

```bash
# Start the API server
make api-serve   # http://localhost:8766

# Search (bearer-authed, GET with query params)
curl -H "Authorization: Bearer $LEAN_MCP_API_KEY" \
     "http://localhost:8766/search?query=What+is+DMAIC%3F&k=5"

# List documents
curl -H "Authorization: Bearer $LEAN_MCP_API_KEY" \
     http://localhost:8766/documents
```

### CLI Quick Start

```bash
# Ingest a PDF
lean ingest "data/My Lean Six Sigma Book.pdf"

# Search the corpus
lean search "What is DMAIC?" --k 5

# Reingest everything with OCR
lean reingest-all --force

# Evaluate retrieval quality
lean eval --sample-size 50 --k 5
```

## Configuration

Secrets go in `.env`, app config goes in `src/lean/config/config.yaml`. Environment variables override YAML values. The `get_settings()` factory caches a single Settings instance (`@lru_cache`) — all consumers share it.

### `.env` (secrets — not committed)

```
SUPABASE_DB_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres
HF_TOKEN=<your-token>
LEAN_MCP_API_KEY=<your-key>
MINIMAX_API_KEY=<your-key>
```

### `config.yaml` (app config — committed, override URLs for your setup)

```yaml
ocr:
  base_url: http://<gpu-host>:8000    # empty = markitdown only
embedding:
  remote_url: http://<gpu-host>:11434  # empty = local CPU
  remote_model: lfm2.5-embed-32k
  num_ctx: 32768
chunking:
  target_max: 450
  hard_cap: 500
  overlap: 50
retrieval:
  top_k: 5
  hybrid_search: true
  fetch_multiplier: 8
  fetch_k_floor: 40         # minimum candidate set size
  rrf_k: 60                 # Reciprocal Rank Fusion constant
  rerank:
    enabled: true
eval:
  sample_size: 50
  k: 5
  seed: 42
storage:
  source_prefix: "sources/"
  markdown_prefix: "markdown/"
health:
  http_timeout: 10
logging:
  level: INFO
llm:
  generate_max_tokens: 500
  generate_temperature: 0.0
  contextual_retrieval: false  # set true + add MINIMAX_API_KEY to .env
  multi_query: false           # set true for multi-query generation
  hyde: false                  # set true for HyDE
```

## Database Migrations

| # | File | Purpose |
|---|---|---|
| 001 | `extensions.sql` | vector, uuid-ossp, pg_trgm |
| 002 | `documents.sql` | documents table |
| 003 | `chunks.sql` | chunks + ivfflat index + trigram |
| 004 | `query_logs.sql` | search query analytics |
| 005 | `tsvector.sql` | BM25 full-text GIN index |
| 006 | `eval_runs.sql` | eval metrics tracking |
| 007 | `drop_markdown_content.sql` | drop unused column |

## License

MIT for project code. See model licenses for third-party weights (`baidu/Unlimited-OCR`, `LiquidAI/LFM2.5-Embedding-350M`).
