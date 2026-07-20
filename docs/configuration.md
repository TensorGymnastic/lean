# Configuration reference

`lean` uses `pydantic-settings` (`Settings` class) with two sources:

| Source | Purpose | In repo? |
|---|---|---|
| `.env` | Secrets (DB DSN, HF token, API key, LLM key, VLM key) | gitignored |
| `configs/<domain>.yaml` | App config (model revisions, ports, chunking, retrieval, VLM transport, domain manifest) | committed |

**Loader contract:** `CoreSettings.from_yaml(path)` reads every field's
env var first, then overlays the YAML's `settings:` block. `validation_alias`
maps each Settings field to its env var name (e.g. `LEAN_MCP_API_KEY`,
`MCP_HTTP_PORT`, `MINIMAX_API_KEY`).

The singleton `get_settings()` honours the YAML path passed via
`set_active_yaml_path()` (set by `build_from_yaml`), so a CLI invocation
like `lean --config configs/lean-pdf-lss.yaml mcp-serve` sees env + YAML
merged before validation runs.

**No Python defaults for transport ports** (M5 audit fix): `mcp_http_port`
and `api_port` must come from `settings.transport` in YAML or from
`MCP_HTTP_PORT` / `API_PORT` env vars. This makes `config.yaml` the
single source of truth for port assignment.

For the **quick-start** example, see the main `README.md` Configuration
section. This document is the **single source of truth** for every field
and its gotchas.

---

## Secrets (`.env` only)

| Env var | Required | Validation | Notes |
|---|---|---|---|
| `SUPABASE_DB_URL` | yes | — | Direct Postgres DSN (not pooler) for pgvector |
| `HF_TOKEN` | no | — | HF token for gated models; empty disables HuggingFace auth |
| `LEAN_MCP_API_KEY` | yes | `>= 16` chars; **rejects literal `"change-me"`** | Bearer token for MCP HTTP + REST transport. `change-me` is blocked even if it meets the length requirement. |
| `MINIMAX_API_KEY` | no | — | Required only if `llm.contextual_retrieval`, `llm.multi_query`, or `llm.hyde` is true. Aliased to `Settings.llm_api_key`. |
| `OCR_BASE_URL` | no | — | Optional override of `ocr.base_url` from YAML |
| `MCP_HTTP_PORT` | no | `1..65535` | Optional override of `transport.mcp_http_port` from YAML |
| `API_PORT` | no | `1..65535` | Optional override of `transport.api_port` from YAML |
| `VLM_API_KEY` | no | — | Aliased to `Settings.vlm_api_key` |

---

## App config (`config.yaml`)

### `embedding`

| Field | Default | Notes |
|---|---|---|
| `model` | `LiquidAI/LFM2.5-Embedding-350M` | Model identifier for HF / sentence-transformers |
| `model_revision` | `f35ae2c91d687658dbf1f2b449382f0b019b9808` | Pinned SHA for `trust_remote_code` safety |
| `dim` | `1024` | Must match the `vector(N)` column in `db/schemas/003_chunks.sql` |
| `device` | `cpu` | Also used by reranker (no separate `rerank.device`) |
| `remote_url` | `""` | If set, uses remote Ollama; empty = local CPU |
| `remote_model` | `lfm2.5-embed-32k` | Ollama model name |
| `num_ctx` | `32768` | **Must match Ollama `PARAMETER num_ctx`** — mismatch silently truncates long chunks |
| `http_timeout` | `30.0` | Seconds |
| `max_retries` | `3` | Exponential backoff `2 ** attempt` between retries |

**Asymmetric prompts (load-bearing):** `src/lean/core/embeddings/liquid_lmf.py` uses
`prompt_name="query"` for queries and `"document"` for passages, with
`normalize_embeddings=True`. Swapping the embedder without preserving this
will silently destroy retrieval. See ADR-0001 (forthcoming).

### `marker` (primary PDF extractor)

Settings fields flow into the `MarkerAdapter` constructor (see `_merge_extractor_defaults`
in `yaml_loader`). The YAML's `marker:` block is the single source of truth.

| Field | Default | Notes |
|---|---|---|
| `force_ocr` | `false` | Force OCR on all pages (set `true` for scanned PDFs) |
| `remote_url` | `""` | If set, `MarkerAdapter.remote_url` will pick it up automatically. Empty = local CPU marker. |

When `remote_url` is non-empty, the extractor POSTs the PDF to the remote
GPU server and returns the result. The remote server is expected to be
a pure-Python wrapper around marker's `PdfConverter` (see
`scripts/marker_server.py`).

### `ocr` (secondary PDF extractor)

Settings fields flow into the `UnlimitedOCRAdapter` constructor.

| Field | Default | Notes |
|---|---|---|
| `base_url` | `""` | Empty = markitdown-only fallback. No OCR. |
| `model` | `baidu/Unlimited-OCR` | Model name (informational; client is OpenAI-compatible) |
| `dpi` | `300` | Page render DPI for OCR; validated `[50, 1200]` |
| `timeout_s` | `1800.0` | 30 min for large books |
| `max_tokens` | `32768` | OCR output max tokens per page batch; validated `>= 256` |
| `batch_size` | `20` | Pages per OCR request; validated `>= 1` |

**Silent fallback:** `OCRBackendUnavailable` causes the pipeline to fall back
to markitdown with `extraction_method=markitdown` and ingest **still returns
success**. Watch `IngestResult.warnings` or `extraction_method` to detect this.

### `chunking`

| Field | Default | Notes |
|---|---|---|
| `target_max` | `450` | Soft target tokens per chunk |
| `hard_cap` | `500` | Hard upper bound. Validated: `hard_cap > target_max` |
| `overlap` | `50` | Token overlap between adjacent chunks (boundary context) |
| `max_section_heading_level` | `4` | Headings deeper than this are flattened into the parent section |
| `token_counter_encoding` | `cl100k_base` | tiktoken encoding name |

### `retrieval`

| Field | Default | Notes |
|---|---|---|
| `search_top_k` | `5` | Default `k` when caller doesn't supply one |
| `search_max_k` | `100` | Hard ceiling on user-supplied `k` (DoS prevention). Validated `>= 1`. |
| `search_max_query_len` | `2000` | Reject queries longer than this. Validated `>= 1`. |
| `search_fetch_k_cap` | `500` | Hard ceiling on `fetch_k` regardless of `fetch_multiplier` |
| `min_similarity` | `0.0` | Drop hits below this cosine similarity after rerank |
| `hybrid_search_enabled` | `true` | Enable BM25 + vector fusion via RRF |
| `fetch_multiplier` | `8` | `fetch_k = max(k * fetch_multiplier, fetch_k_floor)`, then clamped by `search_fetch_k_cap` |
| `fetch_k_floor` | `40` | Minimum candidate set even when `k` is small |
| `rrf_k` | `60` | Reciprocal Rank Fusion constant. Validated `> 0`. |
| `rerank_enabled` | `false` | Python-side default is off; the YAML may override |
| `rerank_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `rerank_model_revision` | `c5ee24cb16019beea0893ab7796b1df96625c6b8` | Pinned SHA for reproducibility |
| `rerank_top_n` | `5` | Top N after rerank. If `top_k > rerank.top_n`, rerank discards candidates the user asked for. |

### `eval`

| Field | Default | Notes |
|---|---|---|
| `eval_sample_size` | `50` | Chunks to sample. Validated `> 0`. |
| `eval_k` | `5` | Top-k for hit_rate / MRR / NDCG / Recall |
| `eval_seed` | `42` | Seeds the Python `random.Random(seed).sample(rows, n)` call that selects chunks — stable across Postgres versions and platforms. |

### `transport`

| Field | Default | Notes |
|---|---|---|
| `mcp_http_host` | `127.0.0.1` | MCP HTTP bind host. Env override: `MCP_HTTP_HOST`. |
| `mcp_http_port` | **(required)** | MCP HTTP port. Source: `settings.transport.mcp_http_port` (YAML) or `MCP_HTTP_PORT` env. Validated `1..65535`. **No Python default** (M5). |
| `api_port` | **(required)** | FastAPI REST port. Source: `settings.transport.api_port` or `API_PORT` env. Validated `1..65535`. **No Python default** (M5). |

### `storage`

| Field | Default | Notes |
|---|---|---|
| `corpus_root` | `data` | Ingest paths must resolve inside this directory. Outside → `PermissionError` (→ HTTP 403 on REST). |
| `max_pdf_mb` | `200` | Reject PDFs larger than this before `read_bytes`. Validated `>= 1`. |

### `health`

| Field | Default | Notes |
|---|---|---|
| `health_http_timeout` | `10.0` | OCR / Ollama / pgvector health-check request timeout (seconds) |

### `logging`

| Field | Default | Notes |
|---|---|---|
| `log_level` | `INFO` | Python logging level (`DEBUG`/`INFO`/`WARNING`/`ERROR`) |

### `ingestion`

| Field | Default | Notes |
|---|---|---|
| `block_match_min_overlap` | `0.15` | Minimum Jaccard-like word-overlap score (in `(0, 1]`) for `_find_best_block_match` to attach a chunk to a marker block. Lower values yield more (noisier) `bbox`/`page_start`/`page_end` matches; higher values yield fewer but cleaner matches. Set higher for low-OCR-quality PDFs. See `services/ingestion.py:_find_best_block_match` and `tests/test_services_ingestion_block_match.py`. |

### `llm` (optional sidecar)

| Field | Default | Notes |
|---|---|---|
| `llm_base_url` | `https://api.minimax.io` | MiniMax OpenAI-compatible endpoint |
| `llm_model` | `MiniMax-Text-01` | Model name |
| `llm_ollama_url` | `""` | Alternative LLM endpoint (e.g. local Qwen) |
| `llm_ollama_model` | `qwen3.5:9b` | Local Ollama model |
| `llm_timeout_s` | `60.0` | LLM HTTP timeout |
| `llm_generate_max_tokens` | `500` | Default max_tokens for generation |
| `llm_generate_temperature` | `0.0` | Default temperature |
| `llm_contextual_retrieval` | `false` | **Irreversible once ingested** — see [`limitations.md`](limitations.md) |
| `llm_multi_query` | `false` | Generate `llm_multi_query_count` paraphrases |
| `llm_hyde` | `false` | HyDE: embed a hypothetical doc instead of the query |
| `llm_multi_query_count` | `4` | Number of paraphrased queries to generate |

The `llm_api_key` field is read from `MINIMAX_API_KEY` in `.env`.

**LLM provider precedence:** if `MINIMAX_API_KEY` is set, MiniMax is used
regardless of `llm_ollama_url`. If you want local Ollama for LLM features,
leave `MINIMAX_API_KEY` empty.

### `vlm` (optional vision-language model for chart/image description)

VLM is configured in **two places** that must agree:

- `settings.vlm.*` (CoreSettings transport-level: `base_url`, `model`, `api_key`, `detail`, `timeout_s`, `max_concurrency`, `max_tokens`, `disable_thinking`)
- `vlm:` at the top of the YAML (DomainConfig domain-level: `prompt_file` / `prompt_inline`, `parser`, `image_heading_format`, **`enabled`**)

The single `enabled` flag lives under top-level `vlm:`; CoreSettings derives
it from `domain_config.vlm.enabled` for the startup warning. Don't declare
`enabled` in both places — declare it once, top-level.

| Field | Default | Notes |
|---|---|---|
| `enabled` | `false` | Master switch (top-level `vlm:`). When false, images are captured but not described. |
| `base_url` | `""` | Required when enabled. e.g. `http://gpu:11434/v1` (Ollama), `https://api.minimax.io/v1` |
| `model` | `""` | Required when enabled. e.g. `gemma3:27b`, `qwen2.5-vl:7b`, `MiniMax-M3` |
| `api_key` | `""` | From `.env` (`VLM_API_KEY`). Empty for local Ollama/vLLM. |
| `detail` | `default` | Image resolution tier: `low`, `default`, or `high`. Higher = better quality, more tokens. |
| `timeout_s` | `120.0` | VLM HTTP timeout (first model load can be slow) |
| `max_concurrency` | `4` | Concurrency cap for parallel VLM image description (enforced via `anyio.Semaphore`). Validated `>= 1`. |
| `max_tokens` | `1000` | Max response tokens per image description. YAML override recommended for chart prompts (`8192` in `lean-pdf-lss.yaml`). |
| `disable_thinking` | `true` | MiniMax-M3: skip `<think>` reasoning for faster structured JSON output. |

**Validation:** if `vlm.enabled=true`, then `vlm.base_url` and `vlm.model`
must be non-empty. `vlm.detail` must be one of `low`/`default`/`high`.

**VLM failure handling:** if a VLM call fails (timeout, network error, invalid
response), a warning is appended to `IngestResult.warnings` and the image is
skipped — ingest continues successfully.

**Search:** use `chunk_type="image"` on the search tool to search only
chart/figure descriptions, or `chunk_type="text"` for text-only chunks.

---

## Validation summary

All constraints below are enforced at startup via pydantic
`field_validator` + `model_validator`. Invalid config fails fast.

| Field | Constraint |
|---|---|
| `LEAN_MCP_API_KEY` | `len >= 16` AND not equal to `"change-me"` |
| `mcp_http_port`, `api_port` | `1 <= port <= 65535` |
| `chunk_hard_cap` | `> chunk_target_max` |
| `rrf_k` | `> 0` |
| `fetch_multiplier` | `>= 1` |
| `eval_k` | `> 0` |
| `search_max_k`, `search_max_query_len`, `search_fetch_k_cap`, `max_pdf_mb` | `>= 1` |
| `vlm.enabled` | If true, `vlm.base_url` and `vlm.model` must be non-empty |
| `vlm.detail` | One of `low`/`default`/`high` |
| `vlm.max_concurrency` | `>= 1` |
| `ocr.dpi` | `[50, 1200]` |
| `ocr.timeout_s` | `> 0` |
| `ocr.max_tokens` | `>= 256` |
| `ocr.batch_size` | `>= 1` |
| `block_match_min_overlap` | `(0, 1]` |