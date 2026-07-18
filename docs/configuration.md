# Configuration reference

`lean` uses `pydantic-settings` (`Settings` class) with two sources:

| Source | Purpose | In repo? |
|---|---|---|
| `.env` | Secrets (DB DSN, HF token, API key) | gitignored |
| `src/lean/config/config.yaml` | App config (URLs, thresholds, model revisions) | committed |

Environment variables override YAML values. `get_settings()` is an
`@lru_cache` singleton — all consumers share one instance. Call
`get_settings.cache_clear()` to force re-read of env (used in tests).

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
| `MINIMAX_API_KEY` | no | — | Required only if `llm.contextual_retrieval`, `llm.multi_query`, or `llm.hyde` is true |
| `OCR_BASE_URL` | no | — | Optional override of `ocr.base_url` from YAML |

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

**Asymmetric prompts (load-bearing):** `src/lean/embeddings/liquid_lmf.py` uses
`prompt_name="query"` for queries and `"document"` for passages, with
`normalize_embeddings=True`. Swapping the embedder without preserving this
will silently destroy retrieval. See ADR-0001 (forthcoming).

### `marker`

Marker (datalab-to/marker) is the primary extraction backend when installed
(`uv sync --extra marker`). Uses surya OCR + texify for high-quality tables,
equations, and layout. Works on CPU natively; GPU auto-detected.

| Field | Default | Notes |
|---|---|---|
| `force_ocr` | `false` | Force OCR on all pages (set `true` for scanned PDFs) |

When marker is not installed, the pipeline falls through to Unlimited-OCR
(if `ocr.base_url` is set) then markitdown.

### `ocr`

Unlimited-OCR via a remote GPU server. Secondary backend (used when marker
is not installed and `base_url` is set).

| Field | Default | Notes |
|---|---|---|
| `base_url` | `""` | Empty = markitdown-only fallback. No OCR. |
| `model` | `baidu/Unlimited-OCR` | Model name (informational; client is OpenAI-compatible) |
| `dpi` | `300` | Page render DPI for OCR |
| `timeout_s` | `1800.0` | 30 min for large books |
| `max_tokens` | `32768` | OCR output max tokens per page batch |
| `batch_size` | `20` | Pages per OCR request |

**Silent fallback:** `OCRBackendUnavailable` causes the pipeline to fall back
to markitdown with `extraction_method=markitdown` and ingest **still returns
success**. Watch `IngestResult.warnings` or `extraction_method` to detect this.

### `chunking`

| Field | Default | Notes |
|---|---|---|
| `target_max` | `450` | Soft target tokens per chunk |
| `hard_cap` | `500` | Hard upper bound. Validated: `hard_cap > target_max` |
| `overlap` | `50` | Token overlap between adjacent chunks (boundary context) |
| `max_heading_level` | `4` | Headings deeper than this are flattened into the parent section |
| `token_encoding` | `cl100k_base` | tiktoken encoding name |

### `retrieval`

| Field | Default | Notes |
|---|---|---|
| `top_k` | `5` | Default `k` when caller doesn't supply one |
| `max_k` | `100` | Hard ceiling on user-supplied `k` (DoS prevention). Validated `>= 1`. |
| `max_query_len` | `2000` | Reject queries longer than this. Validated `>= 1`. |
| `fetch_k_cap` | `500` | Hard ceiling on `fetch_k` regardless of `fetch_multiplier` |
| `min_similarity` | `0.0` | Drop hits below this cosine similarity after rerank |
| `hybrid_search` | `true` | Enable BM25 + vector fusion via RRF |
| `fetch_multiplier` | `8` | `fetch_k = max(k * fetch_multiplier, fetch_k_floor)`, then clamped by `fetch_k_cap` |
| `fetch_k_floor` | `40` | Minimum candidate set even when `k` is small |
| `rrf_k` | `60` | Reciprocal Rank Fusion constant. Validated `> 0`. |
| `rerank.enabled` | `true` (this repo's `config.yaml`) / `False` (Python default) | **Drift:** module docstring says "disabled by default"; actual config enables it |
| `rerank.model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `rerank.model_revision` | `c5ee24cb16019beea0893ab7796b1df96625c6b8` | Pinned SHA for reproducibility |
| `rerank.top_n` | `5` | Top N after rerank. If `top_k > rerank.top_n`, rerank discards candidates the user asked for. |

### `eval`

| Field | Default | Notes |
|---|---|---|
| `sample_size` | `50` | Chunks to sample. Validated `> 0`. |
| `k` | `5` | Top-k for hit_rate / MRR / NDCG / Recall |
| `seed` | `42` | **Only the post-fetch Python shuffle is seeded.** SQL `ORDER BY random()` is unseeded → each run samples different chunks. See [`evaluation.md`](evaluation.md). |

### `transport`

| Field | Default | Notes |
|---|---|---|
| `mcp_host` | `127.0.0.1` | MCP HTTP bind host |
| `mcp_port` | `8765` | MCP HTTP port. Validated 1–65535. |
| `api_port` | `8766` | FastAPI REST port. Validated 1–65535. |

### `storage`

| Field | Default | Notes |
|---|---|---|
| `corpus_root` | `data` | Ingest paths must resolve inside this directory. Outside → `PermissionError` (→ HTTP 403 on REST). |
| `max_pdf_mb` | `200` | Reject PDFs larger than this before `read_bytes`. Validated `>= 1`. |

### `health`

| Field | Default | Notes |
|---|---|---|
| `http_timeout` | `10.0` | OCR / Ollama / pgvector health-check request timeout (seconds) |

### `logging`

| Field | Default | Notes |
|---|---|---|
| `level` | `INFO` | Python logging level (`DEBUG`/`INFO`/`WARNING`/`ERROR`) |

### `llm` (optional sidecar)

| Field | Default | Notes |
|---|---|---|
| `minimax_base_url` | `https://api.minimax.io` | MiniMax OpenAI-compatible endpoint |
| `minimax_model` | `MiniMax-Text-01` | Model name |
| `ollama_url` | `""` | Alternative LLM endpoint (e.g. local Qwen) |
| `ollama_model` | `qwen3.5:9b` | Local Ollama model |
| `timeout_s` | `60.0` | LLM HTTP timeout |
| `generate_max_tokens` | `500` | Default max_tokens for generation |
| `generate_temperature` | `0.0` | Default temperature |
| `contextual_retrieval` | `false` | **Irreversible once ingested** — see [`limitations.md`](limitations.md) |
| `multi_query` | `false` | Generate `multi_query_count - 1` paraphrases (not N) |
| `hyde` | `false` | HyDE: embed a hypothetical doc instead of the query |
| `multi_query_count` | `4` | Number of paraphrased queries to generate (plus original) |

**LLM provider precedence:** if `MINIMAX_API_KEY` is set, MiniMax is used
regardless of `llm.ollama_url`. If you want local Ollama for LLM features,
leave `MINIMAX_API_KEY` empty.

### `vlm` (optional vision-language model for chart/image description)

When enabled, extracted images from marker (charts, figures, diagrams) are
described by a VLM at ingest time. Descriptions are embedded and stored as
`chunk_type='image'` chunks, making charts searchable alongside text.

| Field | Default | Notes |
|---|---|---|
| `enabled` | `false` | Master switch. When false, images are captured but not described. |
| `provider` | `ollama` | Provider label (informational — all use OpenAI-compat protocol) |
| `base_url` | `""` | Required when enabled. e.g. `http://gpu:11434/v1` (Ollama), `https://api.minimax.io/v1` |
| `model` | `""` | Required when enabled. e.g. `gemma3:27b`, `qwen2.5-vl:7b`, `MiniMax-M3` |
| `api_key` | `""` | From `.env` (`LEAN_VLM_API_KEY`). Empty for local Ollama/vLLM. |
| `detail` | `default` | Image resolution tier: `low`, `default`, or `high`. Higher = better quality, more tokens. |
| `timeout_s` | `120.0` | VLM HTTP timeout (first model load can be slow) |
| `max_concurrency` | `4` | Reserved for future parallel image description |
| `max_tokens` | `1000` | Max response tokens per image description |
| `disable_thinking` | `true` | MiniMax-M3: skip `<think>` reasoning for faster structured JSON output. Set `false` for complex charts that benefit from reasoning. |

**Validation:** if `vlm.enabled=true`, then `vlm.base_url` and `vlm.model` must
be non-empty. `detail` must be one of `low`/`default`/`high`.

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
| `mcp_port`, `api_port` | `1 <= port <= 65535` |
| `chunk_hard_cap` | `> chunk_target_max` |
| `rrf_k` | `> 0` |
| `fetch_multiplier` | `>= 1` |
| `eval_k` | `> 0` |
| `search_max_k`, `search_max_query_len`, `search_fetch_k_cap`, `max_pdf_mb` | `>= 1` |
| `vlm.enabled` | If true, `vlm.base_url` and `vlm.model` must be non-empty |
| `vlm.detail` | One of `low`, `default`, `high` |
| `vlm.max_concurrency` | `>= 1` |
