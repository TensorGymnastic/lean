# SOLID Restructuring — lean codebase

**Date:** 2026-07-12
**Status:** Approved
**Approach:** Pragmatic SOLID (split God modules, add service layer, centralize config)

---

## Problem

The codebase has grown to 2,484 LOC across 32 Python files with clear SOLID violations:

1. **`pgvector.py` (417 LOC)** — God Class: document CRUD + chunk CRUD + vector search + BM25 search + RRF fusion + query logging + corpus stats. Violates SRP.
2. **`tools.py` (337 LOC)** — God Module: 8 MCP tools with business logic inline. CLI and REST both import from `mcp_server.tools` — transport layer calling transport layer.
3. **DRY**: `_get_embedder()` duplicated in `tools.py` and `search.py`. `Settings()` instantiated ad-hoc everywhere.
4. **Config mixing**: Secrets and app config both in `settings.py`.

## Solution

### 1. Config layer: `.env` + `config.yaml`

**`.env`** (gitignored, secrets only):
```
SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_DB_URL, HF_TOKEN, LEAN_MCP_API_KEY
```

**`config.yaml`** (committed, application defaults):
```yaml
vllm:
  base_url: http://localhost:8000

embedding:
  model: LiquidAI/LFM2.5-Embedding-350M
  dim: 1024
  device: cpu

ocr:
  model: baidu/Unlimited-OCR
  dpi: 300
  timeout_s: 600.0
  max_tokens: 32768

chunking:
  target_min: 350
  target_max: 450
  hard_cap: 500
  max_heading_level: 4
  token_encoding: cl100k_base

retrieval:
  top_k: 5
  min_similarity: 0.0
  hybrid_search: true
  rerank:
    enabled: false
    model: cross-encoder/ms-marco-MiniLM-L-6-v2
    top_n: 5

storage:
  sources_bucket: sources
  markdown_bucket: markdown

transport:
  mcp_host: 127.0.0.1
  mcp_port: 8765
  api_port: 8766
```

`config/settings.py` loads both into a unified pydantic Settings object. YAML values are defaults overridable by env vars (env wins for deployment flexibility).

### 2. Service layer (NEW: `services/`)

Business logic extracted from `tools.py` into three focused service modules:

- **`services/ingestion.py`** — `ingest_pdf(path)`, `reingest(doc_id)`. Orchestrates extraction → metadata → chunking → embedding → storage. Currently inlined in `tools.py:ingest_pdf` and `tools.py:reingest`.

- **`services/search.py`** — `search(query, **filters)`. Orchestrates embed → hybrid search → rerank → postprocess → log. Currently in `retrieval/search.py` but with module-level singletons.

- **`services/corpus.py`** — `list_documents()`, `delete_document(id)`, `corpus_stats()`, `get_chunk(id)`, `get_document_markdown(id)`. Currently thin wrappers in `tools.py`.

Services depend on store interfaces and infrastructure, not on transport layer.

### 3. Store split (from `pgvector.py` → `store/` package)

417 LOC God Class → 5 focused modules:

- **`store/base.py`** — `StoreConnection`: connection management, `from_env()`, `close()`. ~30 LOC.
- **`store/documents.py`** — `DocumentRepo`: `upsert_document()`, `delete_document()`, `list_documents()`, `get_document()`. ~80 LOC.
- **`store/chunks.py`** — `ChunkRepo`: `replace_chunks()`, `get_chunk()`, `count_chunks()`. ~60 LOC.
- **`store/search.py`** — `SearchEngine`: `vector_search()`, `bm25_search()`, `reciprocal_rank_fusion()`. ~120 LOC.
- **`store/analytics.py`** — `AnalyticsRepo`: `log_query()`, `corpus_stats()`, `count_documents()`. ~60 LOC.

### 4. Infrastructure factory (NEW: `infrastructure/embedder.py`)

Single embedder factory replacing the duplicated `_get_embedder()` pattern:

```python
# infrastructure/embedder.py
_embedder: LiquidLMFEmbedder | None = None

def get_embedder() -> LiquidLMFEmbedder:
    global _embedder
    if _embedder is None:
        settings = get_settings()
        _embedder = LiquidLMFEmbedder(...)
    return _embedder
```

Called by `services/ingestion.py`, `services/search.py`, and `eval/runner.py`.

### 5. Transport layer (thin)

After extracting business logic to services:

- **`mcp_server/tools.py`** — Thin tool definitions that delegate to services. Each tool is 3-5 lines: parse args → call service → return result.
- **`api/routes.py`** — Thin REST routes that delegate to services. Same pattern.
- **`cli.py`** — Thin CLI commands that delegate to services. Same pattern.

All three transport layers import from `services/`, not from each other.

### 6. Dependency direction (corrected)

```
Before:  CLI → mcp_server.tools → store/embeddings/extraction
         REST → mcp_server.tools → store/embeddings/extraction

After:   CLI → services → store/embeddings/extraction
         REST → services → store/embeddings/extraction
         MCP → services → store/embeddings/extraction
```

No transport layer imports from another transport layer.

## Files Created

| File | LOC est. | Responsibility |
|---|---|---|
| `config/settings.py` | 80 | Load .env + config.yaml → Settings |
| `config/config.yaml` | 40 | App config defaults |
| `services/ingestion.py` | 100 | Ingest pipeline orchestration |
| `services/search.py` | 80 | Search orchestration |
| `services/corpus.py` | 60 | Corpus management |
| `store/base.py` | 30 | DB connection |
| `store/documents.py` | 80 | Document CRUD |
| `store/chunks.py` | 60 | Chunk CRUD |
| `store/search.py` | 120 | Vector + BM25 + RRF |
| `store/analytics.py` | 60 | Query logs + stats |
| `infrastructure/embedder.py` | 25 | Singleton embedder factory |

## Files Modified

| File | Change |
|---|---|
| `mcp_server/tools.py` | 337→100 LOC (thin delegates to services) |
| `api/routes.py` | 85→60 LOC (thin delegates to services) |
| `cli.py` | 241→200 LOC (delegates to services instead of tools) |
| `.env.example` | Update (remove VLLM_BASE_URL, it's in config.yaml now) |

## Files Deleted

| File | Reason |
|---|---|
| `settings.py` (root) | Replaced by `config/settings.py` |

## SOLID Principles Applied

- **S**RP: Each store module has one responsibility. Each service has one responsibility.
- **O**CP: New search strategies (e.g., semantic chunking) can be added to `store/search.py` without modifying services.
- **L**SP: Store modules share `StoreConnection` base — substitutable.
- **I**SP: Services depend on focused interfaces (DocumentRepo, ChunkRepo), not a God Class.
- **D**IP: Services depend on store abstractions, not concrete pgvector implementation details.

## KISS/DRY/YAGNI Applied

- **KISS**: No Clean Architecture layering. No abstract base classes. Concrete modules with clear boundaries.
- **DRY**: One embedder factory. One settings loader. One connection class.
- **YAGNI**: No repository interfaces (no ABCs). No dependency injection container. No event bus. Just clean modules.
