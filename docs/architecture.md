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
├── config/
│   ├── settings.py          # pydantic-settings (.env + config.yaml, validated)
│   └── config.yaml          # committed app config (URLs, thresholds, revisions)
├── models/
│   └── schemas.py           # shared Pydantic types (Chunk, Document, SearchHit…)
├── extraction/              # PDF → markdown
│   ├── pipeline.py          # marker → OCR → markitdown orchestrator
│   ├── marker_converter.py  # PRIMARY: datalab-to/marker (surya + texify), local + remote GPU
│   ├── unlimited_ocr.py     # SECONDARY: baidu/Unlimited-OCR via remote transformers
│   ├── markitdown_fallback.py  # TERTIARY: pure-Python fallback
│   ├── ocr_postprocess.py   # annotation stripper + empty-paragraph filter
│   ├── metadata.py          # title / authors / year / publisher
│   └── contextual.py        # optional Contextual Retrieval (LLM)
├── chunker/
│   ├── markdown_ast.py      # mistune section parser
│   └── recursive.py         # tiktoken recursive splitter
├── embeddings/
│   ├── liquid_lmf.py        # local CPU embedder (sentence-transformers)
│   └── remote_ollama.py     # remote Ollama embedder (HTTP)
├── infrastructure/
│   └── embedder.py          # singleton factory: remote Ollama when set, else local CPU
├── store/                   # Postgres repositories
│   ├── base.py              # StoreConnection wrapper
│   ├── documents.py         # upsert by SHA-256
│   ├── chunks.py            # batch upsert + replace
│   ├── search.py            # vector + BM25 + RRF
│   └── analytics.py         # query logs + eval runs + corpus stats
├── retrieval/
│   ├── reranker.py          # cross-encoder (ms-marco-MiniLM-L-6-v2)
│   └── postprocessors.py    # similarity cutoff + long-context reorder
├── llm/                     # optional sidecar
│   ├── base.py              # MiniMax / Ollama factory
│   └── openai_compatible.py # OpenAI-compatible HTTP client
├── services/                # business logic (the only place logic lives)
│   ├── ingestion.py         # ingest_pdf
│   ├── search.py            # search (orchestrates the pipeline above)
│   └── corpus.py            # list, get, delete, stats
├── mcp_server/              # MCP transport (8 tools, 4 resources, 3 prompts)
├── api/routes.py            # FastAPI REST mirror (7/8 endpoints — see limitations.md)
├── auth/bearer.py           # ASGI bearer-token middleware
├── cli.py                   # Typer CLI (full parity with MCP tools)
└── eval/runner.py           # retrieval evaluation (see evaluation.md)
```

---

## Transport tier pattern

Three transports all delegate to the same service layer:

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  CLI (Typer)│  │ MCP server  │  │  REST API   │
│  cli.py     │  │ mcp_server/ │  │ api/routes  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
              ┌──────────────────┐
              │ services/        │   ← business logic lives here, only here
              │  ingestion.py    │
              │  search.py       │
              │  corpus.py       │
              └──────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ store/ + embedder│   ← infrastructure
              │ + llm + retrieval│
              └──────────────────┘
```

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
