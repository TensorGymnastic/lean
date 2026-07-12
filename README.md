# lean

Dockerized MCP server for Lean Six Sigma PDF corpus ingestion and retrieval.

## What it does

Ingests PDFs into a queryable knowledge base and exposes it to AI agents over
the Model Context Protocol (MCP). Pipeline: PDF → Unlimited-OCR (vLLM) →
markdown → section-aware chunks → Liquid LMF embeddings → Supabase pgvector.

## Architecture

| Service | Location | Purpose |
|---|---|---|
| `supabase-db` | Local docker | Postgres 15 + pgvector, port 54322 |
| `lean-app` | Local docker | fastmcp server (stdio/http:8765) + FastAPI mirror (8766) |
| `vllm-unlimited` | Remote 3080 Ti | Serves `baidu/Unlimited-OCR` via OpenAI-compatible HTTP |

## Quick start

```bash
uv sync --all-groups
make hooks-install
cp .env.example .env  # edit with your keys
supabase start
make db-init
make vllm-health      # verify 3080 Ti is reachable
make ingest-all       # ingest data/*.pdf
make search QUERY="What is DMAIC?"
make mcp-serve        # start MCP server for Claude Desktop / opencode
```

## MCP surface

**8 tools:** `ingest_pdf`, `search`, `get_chunk`, `list_documents`,
`get_document_markdown`, `delete_document`, `reingest`, `corpus_stats`

**4 resources:** `lean://documents`, `lean://documents/{id}/markdown`,
`lean://documents/{id}/chunks`, `lean://stats`

**3 prompts:** `lean_qa`, `lean_glossary`, `lean_compare_concepts`

## Development

```bash
make verify      # ruff + mypy + pytest (unit only)
make format      # ruff format
make lint        # ruff check
```

## Tech stack

- **MCP:** fastmcp v3.4.4
- **OCR:** baidu/Unlimited-OCR via vLLM (fallback: Microsoft MarkItDown)
- **Embeddings:** LiquidAI/LFM2.5-Embedding-350M (1024-dim)
- **Storage:** Supabase (Postgres 15 + pgvector)
- **Python:** 3.12, uv-managed, strict mypy

## License

MIT for project code. See `NOTICE` for third-party model licenses.
