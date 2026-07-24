# lean

A dockerized MCP server driven by YAML domain manifests. Built-in domains
ship out of the box: PDF (marker-pdf + OCR + markitdown + optional VLM),
markdown/source-file corpus, and URL corpus. Adding a fourth domain = drop
a YAML in `configs/` + a `tools.py` in `src/lean/domains/`. PDF-like
domains also need `adapters.py`, `metadata.py`, and optionally `parser.py`
+ a prompt file — see "Adding a new domain" below.

Uses marker-pdf for high-quality PDF extraction (with optional remote GPU
acceleration), Vision-Language Model enrichment for charts and figures,
section-aware chunking, GPU-accelerated embeddings, provenance-tracked
storage, and hybrid BM25 + vector search via pgvector.

## Features

- **YAML-driven domains** — every command (`ingest`, `search`, `reingest`,
  `eval`, `health`, `db-init`, `mcp-serve`, `api-serve`) is selected via
  `--config <yaml>`. Built-in domains: `lean-pdf-lss`, `lean-code`,
  `lean-web`. Domain-specific tools register via shared `@mcp_tool` /
  `@rest_route` / `@cli_command` decorators in `lean.core.adapters`.
- **Marker-pdf extraction (primary)** — `datalab-to/marker` (surya OCR +
  texify) with proper table/equation/heading formatting and figure
  extraction. Runs locally on CPU or on a remote GPU server via HTTP
  (up to 44× faster on a 29-page PDF; single-machine benchmark — see
  `docs/marker-server-deployment.md` for the rig). Falls back to Unlimited-OCR
  (remote transformers) then markitdown (pure Python) when unavailable.
- **VLM chart/image enrichment** — Vision-Language Model (MiniMax M3
  default, Ollama Qwen3.5 fallback) describes every chart, diagram, and
  figure at ingest time. Descriptions are structured
  (`title`, `chart_type`, `axis_labels`, `key_data_points`, `description`)
  and embedded alongside text, making visual content searchable.
  **~3.5s/image, ~$0.004/image via MiniMax M3 API.** Ships
  disabled-by-default (`vlm.enabled: false`) in all 3 example configs.
- **Provenance metadata** — every chunk tracks `embedding_model`,
  `embedding_dim`; every image chunk tracks `image_hash`,
  `provenance_model` (which VLM described it). Enables model-version
  auditing and future dedup.
- **Chunk-type filter** — search returns `text` and `image` chunks;
  filter by `chunk_type="image"` to surface only charts/figures.
- **Section-aware chunking** — mistune AST parser splits markdown by
  headings, then a recursive tiktoken-based splitter bounds chunks to a
  target token window
- **GPU-accelerated embeddings** — LiquidAI/LFM2.5-Embedding-350M
  (1024-dim) served via Ollama on the GPU server, with automatic local
  CPU fallback
- **Hybrid search** — BM25 full-text (PostgreSQL tsvector) fused with
  pgvector cosine similarity via Reciprocal Rank Fusion (RRF, k=60)
- **Cross-encoder reranking** — fetch wide candidate set, rerank with
  `ms-marco-MiniLM-L-6-v2`, return top-k
- **MCP server** — tools exposed over stdio or HTTP, configured per YAML
  manifest. Built-in domains ship tools + matching REST routes via shared
  decorators.
- **Retrieval evaluation** — `lean eval` command computing hit_rate@k,
  MRR@k, NDCG@k, Recall@k. Default mode measures self-similarity, not
  real-world retrieval — see [`docs/evaluation.md`](docs/evaluation.md).
  Use `--dataset` for curated mode or `--mode full` for the full search
  pipeline.
- **Optional LLM sidecar** — Contextual Retrieval, HyDE, multi-query
  generation — all opt-in, pipeline works without LLM
- **Security hardening** — corpus-root path confinement, API key
  validation (min 16 chars, `change-me` rejected), HuggingFace model
  revisions pinned to SHA hashes, non-root Docker user, multi-stage
  build
- **Optional ML deps** — torch/transformers/sentence-transformers only
  when local CPU embeddings are needed (`uv sync --extra local-models`);
  marker-pdf for high-quality extraction (`uv sync --extra marker`);
  trafilatura for web extraction (`uv sync --extra web`)

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
  - **Marker server**: pure-Python HTTP wrapper around marker's
    `PdfConverter` on port 8000 (44× faster than CPU). Configure at
    `settings.marker.remote_url` in the YAML, or `MARKER_REMOTE_URL` env.
  - **Ollama**: `lfm2.5-embed-32k` model (LFM2.5-Embedding-350M, 32K
    context) on port 11434
  - *(optional)* **Unlimited-OCR server**: `baidu/Unlimited-OCR` via
    transformers on port 8001 (secondary extraction fallback)

> All GPU services are optional — `lean` falls back to local marker-pdf
> (CPU), local CPU embeddings, and markitdown when remote servers are
> not configured. VLM enrichment is also optional (disable via
> `vlm.enabled: false` at the top level of the YAML).

## Quick Start

```bash
git clone <repo-url> && cd lean

# 1. Install dependencies
uv sync --all-groups
uv sync --extra marker          # marker-pdf for high-quality extraction
# (optional) Local CPU embeddings + reranker (~2GB torch):
uv sync --extra local-models
# (optional) URL extraction for the lean-web domain:
uv sync --extra web

# 2. Install git hooks
make hooks-install

# 3. Configure secrets
cp .env.example .env  # set SUPABASE_DB_URL, LEAN_MCP_API_KEY (min 16 chars, not 'change-me')
# (optional) VLM_API_KEY for MiniMax, MINIMAX_API_KEY for LLM features

# 4. Start local database
docker compose up -d supabase-db
make db-init

# 5. Pick a domain and ingest
make CONFIG=configs/lean-pdf-lss.yaml ingest-all   # PDFs
make CONFIG=configs/lean-code.yaml ingest-one FILE=README.md   # markdown
make CONFIG=configs/lean-web.yaml ingest https://example.com    # URL

# 6. Search (text + image chunks)
make search QUERY="What is DMAIC?"
make search QUERY="Pareto chart of defects"   # surfaces VLM-described images
```

For the remote GPU server (marker + Ollama + optional OCR setup), see
[`docs/ocr-server-deployment.md`](docs/ocr-server-deployment.md).

## Commands

### CLI (`lean`)

All commands support `--json` for structured output. Use `-v` / `--verbose`
for debug logging.

**Every command requires `--config <yaml>` (or `LEAN_CONFIG` env var).**

| Command | Description |
|---|---|
| `lean --config <yaml> db-init` | Apply SQL migrations from `db/schemas/` |
| `lean --config <yaml> health` | Check DB + embedder + optional VLM/LLM connectivity |
| `lean --config <yaml> mcp-serve` | Start MCP server (`--transport stdio\|http`, `--port`) |
| `lean --config <yaml> api-serve` | Start FastAPI REST API server (`--reload` for dev) |
| `lean --config <yaml> eval` | Run retrieval evaluation (`--sample-size`, `--k`, `--dataset <path>`) |

**Domain-specific commands** (auto-registered from the domain's
`tools.py`):

| Domain | Commands |
|---|---|
| `lean-pdf-lss` | `ingest <pdf>`, `search "<query>"`, `list-documents`, `get-chunk <chunk_id>`, `get-markdown <doc_id>`, `delete <doc_id>`, `reingest <doc_id>`, `reingest-all [--force]` |
| `lean-code` | `ingest <file>`, `ingest-directory --dir <dir> [--pattern <glob>]`, `search`, `list-documents`, `corpus-stats`, `get-chunk`, `get-markdown`, `delete` |
| `lean-web` | `ingest <url>`, `ingest-list --file <urls.txt>`, `search`, `list-documents`, `corpus-stats` |

Search options (all domains): `--k`, `--doc-id`, `--section`, `--author`,
`--year-min`, `--year-max`, `--min-score`, `--chunk-type text\|image`.

### Makefile

| Target | Description |
|---|---|
| `make verify` | format-check + lint + typecheck + unit tests |
| `make verify-all` | all tests (incl. integration/e2e) |
| `make format` / `make format-check` / `make lint` / `make typecheck` | individual checks |
| `make db-init` | apply all SQL migrations |
| `make db-reset` | `supabase db reset` (destroys data) |
| `make ingest-all` / `make ingest-one FILE=…` | ingest PDFs (uses `CONFIG`) |
| `make search QUERY="…"` | search from the command line |
| `make mcp-serve` / `make mcp-serve-http` | start MCP server (stdio / HTTP) |
| `make api-serve` | start FastAPI REST mirror |
| `make health` / `make smoke` | check marker server, OCR server, database, Ollama (`smoke` is an alias for `health`) |
| `make build` / `make up` / `make down` | Docker lifecycle |

## Usage

### MCP Client (Claude Desktop, opencode, etc.)

```json
{
  "mcpServers": {
    "lean": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/lean", "lean", "--config",
               "configs/lean-pdf-lss.yaml", "mcp-serve"]
    }
  }
}
```

Tools exposed by each domain (via shared `@mcp_tool` /
`@rest_route` / `@cli_command` decorators in `lean.core.adapters`):

- `lean-pdf-lss` — `ingest_pdf`, `search`, `get_chunk`, `list_documents`,
  `get_document_markdown`, `delete_document`, `reingest`, `corpus_stats`
- `lean-code` — `ingest_file`, `search`, `get_chunk`, `list_documents`,
  `corpus_stats`, `get_document_markdown`, `delete_document`
- `lean-web` — `ingest_url`, `search`, `list_documents`, `corpus_stats`

Resources and prompts are **not** auto-shipped. Add them by writing
`@mcp.resource` / `@mcp.prompt` functions in your domain's `tools.py`.

### REST API

```bash
make api-serve   # http://localhost:8766

curl -H "Authorization: Bearer $LEAN_MCP_API_KEY" \
     "http://localhost:8766/search?query=What+is+DMAIC%3F&k=5"
```

Domain errors map to HTTP codes: `ValueError` → 400, `PermissionError` →
403, `FileNotFoundError` → 404. See `docs/architecture.md` for the
universal error-handling contract.

## Adding a new domain

1. Create `src/lean/domains/<my_domain>/` with at minimum `tools.py`
   (MCP/REST/CLI tool declarations). PDF-like domains also need
   `adapters.py` (extraction pipeline wiring), `metadata.py` (custom
   metadata extraction), and optionally `parser.py` + a prompt file
   (e.g. VLM chart-extraction prompts).
2. Drop a YAML manifest in `configs/<my-domain>.yaml`.
3. Run `uv run lean --config configs/<my-domain>.yaml --help` to verify
   the surface.

The shared decorators in `lean.core.adapters` register the same function
across MCP, REST, and CLI surfaces — write once, expose everywhere.

## Documentation

- **[`docs/`](docs/)** — reference docs (configuration, operations,
  architecture, evaluation, limitations)
- **[`docs/configuration.md`](docs/configuration.md)** — every
  `Settings` field, defaults, validators, gotchas
- **[`docs/operations.md`](docs/operations.md)** — Docker ports,
  healthcheck semantics, post-ingest reindex, reingest semantics
- **[`docs/architecture.md`](docs/architecture.md)** — pipeline,
  directory layout, transport tier pattern
- **[`docs/limitations.md`](docs/limitations.md)** — known caveats
- **[`docs/decisions/`](docs/decisions/)** — Architecture Decision Records
- **[`AGENTS.md`](AGENTS.md)** — operational rules for agents

## License

MIT for project code. See model licenses for third-party weights
(`datalab-to/marker`, `baidu/Unlimited-OCR`,
`LiquidAI/LFM2.5-Embedding-350M`, MiniMax M3).

For known caveats and runtime behavior, see
[`docs/limitations.md`](docs/limitations.md) and [`AGENTS.md`](AGENTS.md).
