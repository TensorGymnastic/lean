# Architecture

`lean` is a dockerized MCP server for Lean Six Sigma PDF corpus ingestion.
This document describes the **runtime data flow** and the **directory
layout**. For configuration, see [`configuration.md`](configuration.md).

---

## Pipeline

### Ingest

```
PDF
  └─► corpus_root check (PermissionError if outside)
      └─► max_pdf_mb check (ValueError if too large)
          └─► SHA-256 (streamed, 1 MiB blocks)
              └─► extraction.pipeline.extract_pdf_markdown()
                  ├─► marker_converter (primary)
                  │     ├─► _extract_remote — POST to marker.remote_url (GPU server)
                  │     └─► _extract_local — in-process PdfConverter (CPU)
                  ├─► unlimited_ocr (secondary) — if OCR_BASE_URL set
                  └─► markitdown_fallback (tertiary) — pure Python, always available
              └─► (optional) VLM enrichment — describe each chart/image (anyio.Semaphore-bounded)
                  └─► chunker.markdown_ast + recursive.split
                      └─► embedder.embed_documents (local CPU or remote Ollama)
                          └─► atomic single-txn: documents.upsert (commit=False) + chunks.replace (commit=False) + conn.commit()
                              └─► IngestResult
```

All heavy operations (extraction, embedding, VLM, DB) run via
`anyio.to_thread.run_sync` so the MCP event loop stays responsive.

### Search

```
query
 └─► validation (empty / oversize / k clamp)
     └─► optional: multi-query expand (LLM)
         └─► optional: HyDE transform (LLM)
             └─► embed query
                 └─► SearchEngine:
                     ├─► vector_search (pgvector cosine)
                     └─► bm25_search (PostgreSQL tsvector)
                         └─► RRF fusion (settings.rrf_k)
                             └─► reranker (cross-encoder, if enabled)
                                 └─► min_similarity filter
                                     └─► long-context reorder
                                         └─► truncate to k
                                             └─► list[Chunk]
```

The order matters. Similarity filter runs **after rerank but before**
long-context reorder — so `min_score` cuts happen first, then the
interleave for lost-in-the-middle mitigation.

---

## Directory layout

```
src/lean/
├── cli.py                          # universal entry: lean --config <yaml> <command>
├── core/                           # universal framework
│   ├── adapters.py                 # @mcp_tool / @rest_route / @cli_command + discover_tools
│   ├── config/
│   │   ├── settings.py             # CoreSettings: pydantic-settings (.env + YAML overlay)
│   │   └── domain_config.py        # DomainConfig: YAML manifest schema (extractors, vlm, tools, metadata)
│   ├── models/
│   │   └── schemas.py              # Chunk, DocumentSummary, IngestResult, CorpusStats, ExtractionMethod
│   ├── extraction/
│   │   ├── base.py                 # Pipeline + Extractor Protocol + BackendUnavailable
│   │   ├── marker.py               # MarkerExtractor (remote URL or in-process marker)
│   │   ├── ocr.py                  # UnlimitedOCRExtractor
│   │   ├── markitdown.py           # MarkitdownExtractor (fallback)
│   │   ├── metadata.py             # extract_metadata, PdfMetadata
│   │   └── pipeline_helpers.py     # find_best_block_match, hash_image, build_chunk_rows, describe_images
│   ├── chunker/
│   │   ├── markdown_ast.py         # mistune section parser
│   │   └── recursive.py            # tiktoken recursive splitter
│   ├── embeddings/
│   │   ├── liquid_lmf.py           # local CPU (lazy-imports sentence-transformers)
│   │   └── remote_ollama.py        # remote Ollama (HTTP)
│   ├── infrastructure/
│   │   └── embedder.py             # get_embedder() factory: remote Ollama → local CPU
│   ├── store/
│   │   ├── base.py                 # StoreConnection wrapper
│   │   ├── documents.py            # upsert by SHA-256
│   │   ├── chunks.py               # ChunkRepo + ChunkRow
│   │   ├── search.py               # SearchEngine + SearchHit
│   │   └── analytics.py            # query logs + eval runs
│   ├── retrieval/
│   │   ├── reranker.py             # cross-encoder
│   │   ├── postprocessors.py       # similarity cutoff + long-context reorder
│   │   └── query_transform.py      # HyDE / multi-query helpers
│   ├── llm/                        # optional sidecar
│   │   ├── base.py                 # get_llm() factory
│   │   └── openai_compatible.py    # OpenAI-compatible HTTP client
│   ├── vlm/
│   │   ├── client.py               # OpenAICompatibleVLM (Protocol VLMClient)
│   │   └── prompts.py              # default chart-extraction prompt + JSON parser
│   ├── services/                   # business logic
│   │   ├── ingestion.py            # ingest_pdf, reingest
│   │   ├── search.py               # search (orchestrates the pipeline above)
│   │   └── corpus.py               # list, get, delete, stats
│   ├── transports/                 # universal transport factories
│   │   ├── cli.py                  # universal Typer CLI (db_init, health, mcp-serve, api-serve, eval)
│   │   ├── api.py                  # build_api() FastAPI factory (bearer auth)
│   │   ├── mcp.py                  # build_mcp() FastMCP factory
│   │   ├── yaml_loader.py          # build_from_yaml() — full domain wiring
│   │   ├── builder.py              # TransportBuilder (legacy class-based path)
│   │   ├── registration.py         # DomainRegistration Protocol
│   │   └── health.py               # check_health() + _check_database/_check_ocr/_check_ollama
│   ├── auth/
│   │   └── bearer.py               # ASGI bearer-token middleware
│   ├── eval/                       # retrieval evaluation
│   │   └── runner.py               # hit_rate@k, MRR@k, NDCG@k, Recall@k
│   └── __init__.py
├── domains/                        # built-in domain adapters
│   ├── pdf_lss/                    # Lean Six Sigma PDF corpus
│   │   ├── adapters.py             # MarkerAdapter / OCRAdapter / MarkitdownAdapter
│   │   ├── metadata.py             # PDF metadata extraction (overrides core)
│   │   ├── parser.py               # JSON + fence parser
│   │   ├── prompts.py              # domain-specific prompt extensions
│   │   ├── chart_extraction_prompt.txt
│   │   └── tools.py                # 8 MCP + 7 REST + 9 CLI (registers via lean.core.adapters)
│   ├── code/                       # markdown / source-file corpus
│   │   ├── adapters.py             # MarkdownFileExtractor
│   │   ├── metadata.py             # file-stats metadata
│   │   └── tools.py                # 7 MCP + 6 REST + 8 CLI
│   └── web/                        # URL corpus
│       ├── adapters.py             # WebPageExtractor (httpx + trafilatura)
│       ├── metadata.py             # URL metadata
│       └── tools.py                # 7 MCP + 4 REST + 6 CLI
configs/                            # three example domain manifests
├── lean-pdf-lss.yaml               # LSS corpus config (24 inline comments)
├── lean-code.yaml                  # code/markdown corpus config
└── lean-web.yaml                   # URL corpus config
tests/                              # unit tests + integration + e2e (see Makefile targets)
db/schemas/                         # SQL migrations (universal + LSS-extended)
docs/                               # reference docs (this directory)
└── decisions/                       # ADR scaffolding
```

---

## Transport tier pattern

Three transports all delegate to the same service layer. The YAML loader
plugs the domain's tools into each:

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  CLI (Typer)│  │ MCP server  │  │  REST API   │
│  cli.py     │  │ mcp.py      │  │ api.py      │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
              ┌──────────────────┐
              │ core/services/   │   ← business logic lives here, only here
              │  ingestion.py    │
              │  search.py       │
              │  corpus.py       │
              └──────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ core/store/ +    │   ← infrastructure
              │ embedder + llm + │
              │ retrieval        │
              └──────────────────┘
```

The universal CLI in `core/transports/cli.py` registers 5 commands
(`db_init`, `health`, `mcp-serve`, `api-serve`, `eval`) and merges the
domain's `register_cli(...)` commands on top. The same pattern applies
to MCP (`build_mcp` + `register_mcp`) and REST (`build_api` +
`register_api`).

**Rule:** transport modules (CLI / MCP / REST) contain no business
logic. They parse args, call a service, format the response. If you find
yourself adding a branch in a transport file, push it into a service
instead. This is the enforced pattern for adding new operations.

---

## Singletons

Three factories are cached via `@lru_cache(maxsize=1)`:

| Factory | Lifetime | Why |
|---|---|---|
| `get_settings()` | process | Cheap re-reads if `cache_clear()` is called (tests) |
| `get_embedder()` | process | Avoids re-loading ~500 MB torch model per call |
| `get_llm()` | process | HTTP client reuse; connection pool warm |
| `_get_reranker()` | process (maxsize=4) | CrossEncoder model load is expensive |

All consumers share a single instance. Call `cache_clear()` to force
re-creation (used in tests; never in production paths).

---

## DB schema (overview)

```
documents ───┬── source_sha256 (unique) — dedup key
              ├── extraction_method (marker | unlimited_ocr | markitdown)
              └── metadata (jsonb)

chunks ──────┴── document_id (FK cascade)
              ├── chunk_index >= 0, token_count > 0 (CHECK)
              ├── chunk_type IN ('text','image') (CHECK — migration 012)
              ├── section_path, heading_text
              ├── page_start, page_end  (populated by _enrich_chunks_with_block_meta on local marker path; NULL on remote marker path — see limitations.md)
              ├── content (text)
              ├── embedding vector(N) — ivfflat index, lists=100
              ├── image_meta (jsonb, image chunks only)
              ├── bbox (jsonb, populated on local marker path; NULL on remote)
              ├── image_hash (text, surfaced via corpus_stats find_duplicate_image_hashes)
              └── provenance: provenance_model, embedding_model, embedding_dim

query_logs   ─── query_text, k > 0, filters, hit_chunk_ids, latency_ms >= 0

eval_runs    ─── mrr, ndcg, recall, mean_latency_ms >= 0, sample_count > 0, k > 0
```

See `db/schemas/` for full DDL. Migrations run on first DB init via
`docker-entrypoint-initdb.d`, or manually via `make db-init`.
