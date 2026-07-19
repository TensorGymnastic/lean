# Architecture validation — 2026-07-19

**Scope:** `lean` repository at HEAD `75cc41d` on `master`.
**Method:** Code-first read of every Python source file under `src/lean/`
(48 files, 4751 LOC) and `tests/`. Cross-referenced against
`src/lean/config/config.yaml` and `src/lean/config/settings.py`.
**No code changes** in this audit — this is a validation, not a
remediation. Each finding cites `file:line` and a proposed
disposition.

## Axes

1. **Centralized configuration** — does every consumer go through
   `get_settings()` / a typed singleton, or do modules re-instantiate
   `Settings()`, read `os.environ` directly, or hardcode values?
2. **Single declaration** — are magic strings (model names, env var
   names, ports, file paths, protocol paths) declared in only one
   place?
3. **List / table dedup** — are literal tables repeated across files
   instead of being sourced from a single declaration?
4. **Hardcoded values** — are tunable constants buried inside function
   bodies instead of being lifted to `Settings` or module-level
   constants?
5. **Dependency injection seams** — are the singletons clean (factory +
   `@lru_cache`), or do they leak mutable module globals?

Each axis is graded **PASS / MINOR / FAIL** with concrete evidence.

---

## Axis 1 — Centralized configuration: **PASS**

**Production code uses `get_settings()` everywhere.** 15 callsites in
`src/lean/`, no bypass.

- `Settings()` direct instantiations outside the singleton factory:
  only `tests/test_settings.py:20,42,95,107` (legitimate — these
  tests need to exercise validation paths with controlled env).
- `os.environ` reads in production code: exactly one, at
  `src/lean/cli.py:392` (`db_init` builds a `pg_env` to forward
  `PGHOST/PGPORT/...` to `psql`). Legitimate — the function merges
  the inherited env then overrides the PG vars from the parsed DSN.
  This is the documented design (`AGENTS.md:120` says "lean db-init
  uses PGPASSWORD env (not argv)").
- `os.environ` reads in tests: 8 sites, all legitimate env-munging for
  integration test setup (`test_e2e_cli.py:30,49,52`,
  `test_e2e_ingest.py:31`, `test_store_pgvector.py:15`,
  `test_db_schema.py:21`, `test_llm_integration.py:21,22,48`).

**Verdict:** Clean. No action.

---

## Axis 2 — Single declaration: **MINOR (3 duplications)**

Three model/SHA pairs are declared in more than one place. Each is the
"YAML + Python default + class default" triple-declaration pattern.

### Finding 2.1 — `LiquidAI/LFM2.5-Embedding-350M` + revision SHA

| Location | Form |
|---|---|
| `src/lean/config/config.yaml:2-3` | YAML values (canonical) |
| `src/lean/config/settings.py:46,48` | `Field(default=_yaml.get(...))` (read from YAML, falls back to literal) |
| `src/lean/embeddings/liquid_lmf.py:28,32` | `__init__` default `model: str = "LiquidAI/LFM2.5-Embedding-350M"` |
| `tests/conftest.py:32`, `tests/test_settings.py:26`, `tests/test_api_routes.py:135`, `tests/test_services_corpus.py:158`, `tests/test_cli.py:71` | Test fixtures (acceptable) |

The class `__init__` in `embeddings/liquid_lmf.py` redeclares the
model name and SHA as Python defaults. This means the embedder has its
own "fallback" that does not flow through `Settings` — a developer
who changes the YAML will be surprised when the class still loads the
old model. **Disposition:** change the class to require `model` and
`revision` (no defaults) and have the only caller (the singleton
factory in `infrastructure/embedder.py`) thread them from
`Settings.embedding_model` and `Settings.embedding_model_revision`.

### Finding 2.2 — `baidu/Unlimited-OCR` + DPI 300 + timeout 1800 + max_tokens 32768 + batch_size 20

| Location | Form |
|---|---|
| `src/lean/config/config.yaml:14-19` | YAML values (canonical) |
| `src/lean/config/settings.py:58-62` | `Field(default=_yaml.get(...))` (read from YAML, falls back to literal) |
| `src/lean/extraction/unlimited_ocr.py:31-35` | `__init__` default `model: str = "baidu/Unlimited-OCR"`, `dpi: int = 300`, `timeout: float = 1800.0`, `max_tokens: int = 32768`, `batch_size: int = 20` |

Same pattern as 2.1. The function `extract_markdown` in
`unlimited_ocr.py:26-36` redeclares five tunables that already exist
in `Settings.ocr_*`. The only production caller (`pipeline.py`)
already pulls these from `Settings`, so the function defaults are
effectively dead code — the values flow through the caller. **Same
disposition as 2.1:** remove the defaults and require the caller to
thread `Settings.ocr_*`.

### Finding 2.3 — Port numbers 8765 / 8766

| Location | Form |
|---|---|
| `src/lean/config/config.yaml:54-55` | YAML values |
| `src/lean/config/settings.py:96-97` | `Field(default=_yaml.get(...))` |

Two places, not three. Mild. The Python defaults are dead code as
long as the YAML is shipped. Could be eliminated by removing the
defaults and treating the YAML as required. **Disposition:** low
priority; the current pattern is safe.

**Verdict:** MINOR. The class-default pattern in 2.1 and 2.2 is the
real risk. Recommend a small refactor: drop class/function defaults
for the five OCR tunables and the two embedder model fields, have the
singletons thread them from `Settings`.

---

## Axis 3 — List / table dedup: **MINOR**

### Finding 3.1 — Chunk-type literals `"text"` / `"image"`

The string `"text"` and `"image"` are declared in:

- `src/lean/models/schemas.py:48` — `chunk_type: str = "text"`
- `src/lean/store/chunks.py:34` — `ChunkRow.chunk_type: str = "text"`
- `src/lean/services/ingestion.py:240` — `chunk_type="image"` (string
  literal in `_build_chunk_rows`)
- `db/schemas/012_add_chunk_type_check.sql` — CHECK constraint
  enforcing the same two values at the DB layer

Plus the migration-level constraint. The Python-side literal is
duplicated three times. **Disposition:** introduce a tiny
`class ChunkType` enum (or two `LITERAL` constants) in
`src/lean/models/schemas.py` and use it everywhere. The migration
constraint must stay as raw SQL — that is a database concern, not a
Python concern.

### Finding 3.2 — Tool / endpoint name enumeration

The 8 MCP tools are declared in:

- `src/lean/mcp_server/tools.py` (8 `@mcp.tool` decorators)
- `src/lean/api/routes.py` (7 REST handlers, with the 8th
  `reingest` intentionally absent — CLI-only)
- `src/lean/cli.py` (14 Typer commands including CLI-only `reingest`,
  `reingest_all`, `db_init`, `health`)
- `docs/limitations.md:87-101` (REST parity table)
- `README.md:152-163` (same table, repeated for the human reader)

This is **legitimate scope** — each transport layer binds to its own
framework (fastmcp, FastAPI, Typer) and a single registry would not
work cleanly. The duplicated *names* are inevitable. The duplicated
*tables in human docs* are an editorial choice (the doc table is
the human-facing version of what `api/routes.py` declares in code).
**Verdict:** no action; the transport-level enumeration is by design,
and the doc tables are intentionally human-readable.

### Finding 3.3 — JSON key strings (`embedding_model`, `chunk_type`, ...)

The column names appear as string literals in 4 places: SQL
migrations, `models/schemas.py` (Pydantic field names),
`store/chunks.py` (the dataclass), and `store/chunks.py:79-93`
(`executemany` column order). These cannot be deduped — the SQL
migrations are the database source of truth and the Python fields are
the binding layer. **Verdict:** no action.

**Verdict:** MINOR. Finding 3.1 is the only real dedup opportunity.

---

## Axis 4 — Hardcoded values: **MINOR (1 finding)**

### Finding 4.1 — `_find_best_block_match` overlap threshold `0.15`

`src/lean/services/ingestion.py:184`:

```python
if best_score < 0.15:
    return None, None
```

This is the Jaccard-like word-overlap threshold that decides whether
a chunk is "matched" to a marker block. It is the most consequential
tunable in the entire provenance-metadata pipeline — get it wrong
and every `page_start`/`page_end`/`bbox` is silently mis-attributed
(see REVIEW.md HIGH-2). Currently it is a bare float in the middle
of a function body. **Disposition:** lift to
`Settings.block_match_min_overlap` (default `0.15`) with a
`field_validator` that enforces `0.0 < x <= 1.0`. The function
should consume `settings.block_match_min_overlap`.

### Finding 4.2 — Module-level constants (acceptable)

- `src/lean/extraction/unlimited_ocr.py:19` —
  `MAX_PAGE_PIXELS = 50_000_000` (50 MP guard against page-bomb PDFs).
  Module-level constant with a clear comment. Fine.
- `src/lean/store/chunks.py:18` — `_INSERT_BATCH_SIZE = 1000`
  (mitigates Postgres 65535-param limit). Module-level constant with
  a clear comment. Fine.
- `src/lean/services/ingestion.py:39` — `_SHA256_CHUNK_SIZE = 1 << 20`
  (1 MiB streaming hash block). Module-level constant. Fine.

These three are all module-private with a documented reason, and
they are not user-tunable. **Verdict:** no action.

### Finding 4.3 — VLM / OCR HTTP-path inconsistency

- `src/lean/extraction/unlimited_ocr.py:91` — `{ocr_base_url}/v1/chat/completions`
- `src/lean/vlm/client.py:107` — `{self._base_url}/chat/completions`
  (no `/v1`)
- `src/lean/llm/openai_compatible.py:58` — `{self._base_url}/v1/chat/completions`
- `src/lean/embeddings/remote_ollama.py:72` — `{self._base_url}/api/embeddings`

The VLM client drops the `/v1` prefix. The current `config.yaml`
example compensates by baking `/v1` into the `vlm.base_url` value
(commented `https://api.minimax.io/v1`). If a developer follows the
VLM convention literally with `https://api.minimax.io` (no `/v1`),
the client will POST to `https://api.minimax.io/chat/completions`
which 404s. **Disposition:** make the VLM client match the others
(`/v1/chat/completions`) and remove the `/v1` from the YAML example,
**or** explicitly document the "base_url includes /v1" convention in
`docs/configuration.md`. The first is safer.

**Verdict:** MINOR. Finding 4.1 is the only business-logic concern;
Finding 4.3 is a real coupling risk worth a one-line fix.

---

## Axis 5 — Dependency injection seams: **PASS**

Five `@lru_cache` singletons, all consumed via factory functions:

| Singleton | File:Line | Cache | Consumes |
|---|---|---|---|
| `get_settings` | `config/settings.py:195-203` | `maxsize=1` | `Settings()` (the only place it is instantiated) |
| `get_embedder` | `infrastructure/embedder.py:22-23` | `maxsize=1` | `Embedder` Protocol; returns remote Ollama or local Liquid LMF based on `Settings.embedding_remote_url` |
| `get_llm` | `llm/base.py:34-35` | `maxsize=1` | `LLMClient` Protocol; returns `None` when neither MiniMax nor Ollama is configured |
| `_get_reranker` | `retrieval/reranker.py:25-33` | `maxsize=4` | `CrossEncoder` from sentence-transformers; keyed by `(model_name, device, revision)` |
| `_get_converter` | `extraction/marker_converter.py:200-225` | `maxsize=4` | marker-pdf `PdfConverter`; keyed by `force_ocr` |

All five are accessed through getter functions. There are no module
globals holding mutable state, no global `db_pool` or
`http_client` literals. **This is exactly the seam pattern
`AGENTS.md:60-61` documents.**

The `Embedder` and `LLMClient` Protocols exist (per
`AGENTS.md:60-62`). The VLM client (`VLMClient`) is still concrete
(per `AGENTS.md:79` — "add a Protocol when a second provider shape
lands"). Acceptable.

**Verdict:** Clean. No action.

---

## Consolidated priority

Ordered by impact × effort:

| # | Finding | Axis | Severity | Effort |
|---|---|---|---|---|
| 1 | 4.1: `_find_best_block_match` threshold `0.15` should be `Settings.block_match_min_overlap` | Hardcoded values | Medium | 30 min |
| 2 | 2.1 + 2.2: drop the `__init__` model/SHA defaults in `liquid_lmf.LiquidLMFEmbedder` and `unlimited_ocr.extract_markdown`; thread from `Settings` | Single declaration | Medium | 1 hour |
| 3 | 4.3: VLM client uses `/chat/completions` (no `/v1`); either fix the path or document the convention | Hardcoded values | Medium (operator risk) | 5 min (one-line fix) |
| 4 | 3.1: introduce `ChunkType` literal in `models/schemas.py`; replace 3 string literals | List / table dedup | Low | 20 min |
| 5 | 2.3: drop the Python defaults for ports `8765`/`8766` in `Settings` (YAML required) | Single declaration | Low | 5 min |

None of these are blockers. The codebase is in good shape on all
five axes. Items 1–3 are the highest-leverage cleanups; item 4 is a
nice-to-have; item 5 is cosmetic.

## Verification

This audit is documentation-only. No code or test changes were
introduced.

- `make format-check`: 95 files already formatted
- `make lint`: All checks passed!
- `make typecheck`: Success: no issues found in 53 source files
- `make verify`: 338 passed, 59 deselected, 91.82% coverage
- `uv run python scripts/docs_lint.py`: OK (51 keys, 8 MCP tools)

## Cross-references

- `AGENTS.md:60-62` — singleton pattern (verified PASS).
- `AGENTS.md:72-79` — design discipline (KISS / YAGNI / DRY / SOLID).
- `AGENTS.md:77` — `ingest_pdf` is the documented SOLID-S violation;
  re-splitting it would be a separate item (see BLG-004 in the
  workspace backlog).
- BLG-001 (this audit's commissioning item) — see
  `_ai/workspace/lean-backlog-2026-07-19/BACKLOG.md`.
- `lean/AGENTS.md:104-105` — the doc-update rule that produced this
  audit alongside the code it covers.
