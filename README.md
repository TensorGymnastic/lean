# lean

Dockerized MCP server for Lean Six Sigma PDF corpus ingestion and retrieval.

## What it does

Ingests PDFs into a queryable knowledge base and exposes it to AI agents over
the Model Context Protocol (MCP). Pipeline: PDF → Unlimited-OCR → markdown →
section-aware chunks → Liquid LMF embeddings → Supabase pgvector → hybrid search.

## Architecture

| Service | Location | Purpose |
|---|---|---|
| `supabase-db` | Local docker | Postgres 15 + pgvector, port 54322 |
| `lean-app` | Local | fastmcp server (stdio/http:8765) + FastAPI mirror (8766) |
| `ocr-serve` | Remote 3080 Ti (:8000) | `baidu/Unlimited-OCR` via transformers (batched, 20 pages/request) |
| `ollama` | Remote 3080 Ti (:11434) | LFM2.5-Embedding-350M (32K context) for GPU-accelerated embeddings |

### Retrieval pipeline

```
Query → embed (remote Ollama) → hybrid search (BM25 tsvector + pgvector cosine)
      → RRF fusion → similarity cutoff → long-context reorder → top-k results
```

### Ingestion pipeline

```
PDF → render pages (fitz, DPI=300) → batch 20 pages → OCR server →
      clean annotations → section-aware chunk (mistune + tiktoken) →
      embed (remote Ollama) → pgvector upsert
```

## Quick start

### 1. Prerequisites

- Python 3.12+, uv-managed
- Docker (for Supabase local)
- Remote GPU server with:
  - `baidu/Unlimited-OCR` served via transformers on port 8000
  - Ollama with `lfm2.5-embed-32k` model on port 11434

### 2. Setup

```bash
uv sync --all-groups
make hooks-install
cp .env.example .env  # edit: Supabase keys, HF token, MCP API key
```

### 3. Configure remote servers

Edit `src/lean/config/config.yaml`:

```yaml
vllm:
  base_url: http://<gpu-server-ip>:8000    # OCR server

embedding:
  remote_url: http://<gpu-server-ip>:11434  # Ollama
  remote_model: lfm2.5-embed-32k            # 32K context model
```

### 4. Start

```bash
supabase start
make db-init          # apply migrations (001-006)
make ingest-all       # ingest data/*.pdf
make search QUERY="What is DMAIC?"
make mcp-serve        # start MCP server
```

### 5. Connect MCP client

Add to `~/.config/opencode/opencode.json` (or Claude Desktop config):

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

## Remote GPU server setup

### OCR server (Unlimited-OCR via transformers)

```bash
# On the GPU server:
mkdir -p ~/ocr-serve && cd ~/ocr-serve
uv init && uv add "transformers>=4.57.1,<5" torch accelerate PyMuPDF \
  fastapi uvicorn pillow torchvision addict matplotlib easydict einops

# Download model weights
uv run python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('baidu/Unlimited-OCR')"

# Create server.py (see scripts/setup-3080ti.sh for template)
sudo systemd-run uv run python server.py  # serves on :8000
```

### Ollama (LFM2.5 embeddings with 32K context)

```bash
# On the GPU server:
ollama create lfm2.5-embed-32k -f - << 'EOF'
FROM hf.co/LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0
PARAMETER num_ctx 32768
EOF
```

## MCP surface

**8 tools:** `ingest_pdf`, `search`, `get_chunk`, `list_documents`,
`get_document_markdown`, `delete_document`, `reingest`, `corpus_stats`

**Search filters:** `query`, `k`, `doc_id`, `section`, `author`,
`year_min`, `year_max`, `min_score`

**4 resources:** `lean://documents`, `lean://documents/{id}/markdown`,
`lean://documents/{id}/chunks`, `lean://stats`

**3 prompts:** `lean_qa`, `lean_glossary`, `lean_compare_concepts`

## CLI

```bash
lean ingest <path>              # ingest a PDF
lean search "<query>" --k 5     # hybrid search
lean list-documents             # list corpus
lean corpus-stats               # stats
lean eval --sample-size 50      # retrieval eval (hit_rate, MRR)
lean reingest <doc_id>          # re-extract with current settings
lean delete <doc_id>            # remove a document
lean mcp-serve                  # start MCP server
lean db-init                    # apply SQL migrations
```

All commands support `--json` for structured output.

## Development

```bash
make verify      # ruff + mypy + pytest (unit only)
make verify-all  # all tests including integration
make format      # ruff format
make lint        # ruff check
```

## Project structure

```
src/lean/
  config/           .env (secrets) + config.yaml (app config)
  infrastructure/   Singleton factories (embedder)
  services/         Business logic (ingestion, search, corpus)
  store/            Focused repos (base, documents, chunks, search, analytics)
  extraction/       PDF→markdown (OCR, markitdown, metadata, post-process)
  chunker/          Section parser + recursive token splitter
  embeddings/       LiquidLMF + remote Ollama
  retrieval/        Cross-encoder reranker + postprocessors
  eval/             hit_rate/MRR harness
  mcp_server/       Thin MCP tool delegates
  api/              Thin REST delegates
  cli.py            Thin CLI delegates
```

## Database migrations

| # | File | Purpose |
|---|---|---|
| 001 | extensions.sql | vector, uuid-ossp, pg_trgm |
| 002 | documents.sql | documents table |
| 003 | chunks.sql | chunks + ivfflat + trigram |
| 004 | query_logs.sql | search analytics |
| 005 | tsvector.sql | BM25 full-text index |
| 006 | eval_runs.sql | eval metrics tracking |

## Tech stack

- **MCP:** fastmcp v3.4.4
- **OCR:** baidu/Unlimited-OCR (transformers-based, batched 20p/request)
- **Embeddings:** LiquidAI/LFM2.5-Embedding-350M (1024-dim, via Ollama GPU)
- **Search:** pgvector cosine + PostgreSQL tsvector BM25 + RRF fusion
- **Storage:** Supabase (Postgres 15 + pgvector)
- **Python:** 3.12, uv-managed, strict mypy

## License

MIT for project code. See `NOTICE` for third-party model licenses.
