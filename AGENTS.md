# AGENTS.md — lean

## Purpose

Repository-local guidance for agents working in `lean`.

`lean` is a dockerized MCP server for corpus ingestion driven by YAML
domain manifests. The package is a single Python project, not a monorepo;
domains ship in-tree under `src/lean/domains/`. Built-in domains:

- `pdf_lss` — Lean Six Sigma PDF corpus (marker-pdf primary, OCR
  secondary, markitdown fallback, optional VLM image enrichment).
- `code` — Markdown / source-file corpus (direct file read).
- `web` — URL corpus (trafilatura + httpx).

Adding a fourth domain = drop a YAML in `configs/` + maybe 1-2 adapter
files + a `tools.py` module that registers MCP / REST / CLI commands.

## Agent skill library

The agent has access to four skill/tool systems. **Invoke skills
BEFORE writing code, scaffolding, or making commits** — they enforce
discipline that catches most costly mistakes.

### 1. Superpowers

The active meta-system that hooks every response. Calls itself
automatically when a task fits a skill; you can also invoke any skill
explicitly with the `skill` tool.

Core skills (installed at `~/.cache/opencode/packages/superpowers`):

- **brainstorming** — Socratic design refinement. Run before any
  creative work (new features, components, behavior changes). HARD
  GATE: no code until design is presented and approved. Specs land
  in `docs/superpowers/specs/YYYY-MM-DD-<slug>-design.md`.
- **writing-plans** — Turn an approved design into a task checklist
  for a junior engineer. Comes after `brainstorming`.
- **test-driven-development** — Red-green-refactor for any logic
  change. Write the failing test first, commit it, watch it fail,
  implement, watch it pass, refactor.
- **systematic-debugging** — Hypothesis-driven debugging loop for
  test failures and unexpected behavior. Reproduce → minimize →
  hypothesize → instrument → fix → regression-test.
- **code-review-and-quality** — Multi-axis review (correctness,
  readability, architecture, security, performance) before merge.
- **verification-before-completion** — Run the actual command,
  confirm the actual output, then claim completion. Never assert
  "passes" without the green test run.
- **requesting-code-review** / **receiving-code-review** — When to
  ask for review; how to respond to feedback without sycophancy.
- **dispatching-parallel-agents** — For 2+ independent tasks, fan
  out as parallel `task` calls instead of serializing.
- **incremental-implementation** — Deliver changes incrementally when
  the change touches more than one file.

### 2. Matt Pocock's skill library

User-invoked orchestration skills for engineering workflows. Install
via `npx skills@latest add mattpocock/skills`. The repo-local
configuration files written by `/setup-matt-pocock-skills` live in
`docs/agents/`.

| Skill | Use when |
|---|---|
| `/setup-matt-pocock-skills` | **Run once per repo** before any other Matt Pocock skill. Configures issue tracker, triage labels, domain doc layout. |
| `/ask-matt` | "Which skill fits my situation?" router. |
| `/grill-me`, `/grill-with-docs` | Relentless interview about a plan or design. `grill-with-docs` also writes ADR/controlled-vocabulary updates inline. |
| `/tdd` | TDD discipline with red-green-refactor emphasis. |
| `/to-spec` | Turn the current conversation into a spec. Synthesizes what's already been discussed (no interview). |
| `/to-tickets` | Break a spec/plan/conversation into tracer-bullet tickets with explicit blocking edges. |
| `/implement` | Build the work described by a spec or tickets, calling `/tdd` and `/code-review` at seams. |
| `/code-review` | Two-axis review (Standards + Spec), run as parallel sub-agents. |
| `/triage` | Move GitHub issues through needs-triage → needs-info → ready-for-agent → ready-for-human → wontfix. |
| `/improve-codebase-architecture` | Scan for deepening opportunities, present a visual HTML report, then grill through one. Run weekly. |
| `/prototype` | Throwaway prototype to answer a design question. |
| `/diagnosing-bugs` | Disciplined bug-fix loop. |
| `/research` | Investigate a question against primary sources, write cited Markdown. |
| `/domain-modeling` | Sharpen terminology, update `CONTEXT.md` and the glossary. |
| `/codebase-design` | Shared vocabulary for designing deep modules. |
| `/resolving-merge-conflicts` | Resolve by intent to each side's primary source; never `--abort`. |

Skills split on **who can invoke** them. *User-invoked* (above)
require you to type the slash command. *Model-invoked* skills
(brainstorming, TDD, debugging, etc.) can also be reached for
automatically when the task fits.

### 3. Oh-My-OpenCode (the `task` tool categories)

The plugin adds pre-tuned sub-agents you dispatch via `task` with a
`category`. Configure in `.opencode/oh-my-opencode.json` (project) or
`~/.config/opencode/oh-my-opencode.json` (user).

| Category | Default model | Use for |
|---|---|---|
| `quick` | `claude-haiku-4-5` | Trivial: typo fixes, one-line changes, single-file edits |
| `unspecified-low` | `claude-sonnet-4-6` | General tasks, low effort |
| `unspecified-high` | `claude-opus-4-6` | General tasks, high effort, complex reasoning |
| `visual-engineering` | `gemini-3-pro` | Frontend, UI/UX, design, animation |
| `ultrabrain` | `gpt-5.3-codex` (xhigh) | Deep logical reasoning, complex architecture |
| `deep` | `gpt-5.3-codex` (medium) | Autonomous problem-solving, thorough research |
| `artistry` | `gemini-3-pro` | Creative/unconventional approaches |
| `writing` | `kimi-for-coding/k2p5` | Documentation, prose, technical writing |

Specialist agents (dispatched explicitly by name):

- **explore** — fast codebase navigation (grep/find/ripgrep).
- **librarian** — reads external docs, fetches upstream sources.
- **multimodal-looker** — reads images / screenshots / diagrams.
- **oracle** — architecture and debugging consultation.
- **Prometheus (planner)** — produces implementation plans.
- **Atlas (plan executor)** — runs the plan step-by-step.

**Rule of thumb:** use `category="unspecified-high"` for anything
non-trivial and let the runtime pick the model. Use named agents
(`explore`, `librarian`) when you specifically need their role.

### 4. Workspace skills (`/home/sl/dev`)

Local skills in `.github/skills/` and `.opencode/skills/` cover
domain-specific workflows:

- `.opencode/skills/memory-management` — persist context across
  sessions via OpenMemory MCP.
- `.opencode/skills/git-commit-standards` — atomic-commit message
  format; consult before every commit.
- `.opencode/skills/github-issues-integration` — fetch / comment /
  update / attach to GitHub issues.
- `.opencode/skills/pure-cli-exploration` — `rg`, `fdfind`, `ast-grep`
  cheat sheet for codebase searches.
- `.opencode/skills/documentation` — when and where to write docs.
- `.github/skills/browser-navigation`, `navigation-crawl-maintenance` —
  UI automation + crawl replay.
- `.github/skills/repository-platform-assessment` — audit a repo's
  platform integration.

Plus the workspace-level docs at `/home/sl/dev/_ai/docs/` for
durable cross-project knowledge.

## OpenSpec workflow (spec-driven change management)

Spec-driven development for AI coding assistants. Install with
`npm install -g @fission-ai/openspec@latest`. The lean repo **does
not yet have an `openspec/` directory** — sibling repos
(`repos/quanti/web-document-ingestor/`, `repos/storefront/`) use it.

**Use for any change that:** touches public API or schema, is
expensive to reverse (months of work), would be second-guessed
without context, or has multiple viable approaches worth recording.
**Skip for:** typo fixes, single-line edits, config tweaks.

Default loop (run in the AI assistant chat):

```text
/opsx-explore <idea>          → optional: think it through first
/opsx-propose <change-name>   → AI drafts proposal, specs, design, tasks
/opsx-apply                   → AI builds it, checking off tasks
/opsx-archive                 → specs updated, change filed away
```

Folder layout:

```text
openspec/
├── specs/                  # source of truth (system behavior)
│   └── <domain>/spec.md    #   organized by domain
└── changes/                # proposed updates (one folder per change)
    └── <change-name>/
        ├── proposal.md     #   the "why" and "what"
        ├── design.md       #   the "how"
        ├── tasks.md        #   implementation checklist
        └── specs/          #   delta specs (ADDED / MODIFIED / REMOVED)
```

Commands are sourced from `.opencode/commands/opsx-*.md` (see
`repos/storefront/.opencode/commands/` for the canonical templates).
The archive step merges deltas into `specs/`.

For the lean repo, the natural domain breakdown is:

- `openspec/specs/extraction/` — pipeline + backend contracts
- `openspec/specs/transport/` — MCP / REST / CLI surface
- `openspec/specs/storage/` — pgvector + chunk schemas
- `openspec/specs/retrieval/` — search + rerank

## Recommended workflow for typical tasks

| Task type | Workflow |
|---|---|
| New feature or behavior change | `brainstorming` → `writing-plans` → `test-driven-development` → `code-review-and-quality` |
| Bug in existing behavior | `systematic-debugging` → fix → `test-driven-development` → `code-review-and-quality` |
| Refactor / cleanup | `improving-codebase-architecture` (or `code-simplification`) → `test-driven-development` → `code-review-and-quality` |
| External library / API research | `research` skill (writes cited Markdown to repo) |
| Multi-file change touching >3 modules | OpenSpec: `/opsx-explore` → `/opsx-propose` → `/opsx-apply` → `/opsx-archive` |
| Configuration / YAML / .env changes | Direct edit + `make verify`; use OpenSpec if the change introduces new settings keys |
| Architecture audit | `code-review-and-quality` with the full multi-axis template |
| Skill/workflow question | `/ask-matt` |
| Visual question (mockup, diagram) | Ask Gemini via `task(category="visual-engineering")` |

For multi-task problems, dispatch **parallel `task` calls** instead
of serializing (see `dispatching-parallel-agents`).

## Architecture (project facts)

**Layering (enforced):** transport (mcp / api / cli) → services → store → infrastructure.
No business logic in transports. Each store repo is CRUD-narrow.

- `src/lean/core/config/settings.py` — pydantic-settings: `.env` (secrets) + YAML overlay via `CoreSettings.from_yaml(path)`. Singleton via `get_settings()`.
- `src/lean/core/config/domain_config.py` — `DomainConfig` pydantic model loaded from a YAML manifest.
- `src/lean/core/transports/yaml_loader.py` — `build_from_yaml(path)` is the canonical wiring entry point.
- `src/lean/core/extraction/` — universal pipeline (`base.Pipeline` + `base.Extractor` Protocol): marker / OCR / markitdown / metadata / pipeline_helpers.
- `src/lean/core/vlm/` — `OpenAICompatibleVLM` (httpx, `/v1/chat/completions`); chart-extraction prompts in `vlm/prompts.py`.
- `src/lean/core/chunker/` — `markdown_ast.py` (mistune) + `recursive.py` (tiktoken).
- `src/lean/core/embeddings/` — `liquid_lmf.py` (local CPU) + `remote_ollama.py` (GPU); factory in `infrastructure/embedder.py`.
- `src/lean/core/store/` — `base`, `documents`, `chunks`, `search`, `analytics`.
- `src/lean/core/retrieval/` — cross-encoder reranker + postprocessors.
- `src/lean/core/llm/` — OpenAI-compatible client; optional sidecar for HyDE/multi-query/contextual retrieval.
- `src/lean/core/services/` — `ingestion.py` (extract→VLM→embed→store), `search.py` (hybrid→rerank), `corpus.py` (CRUD).
- `src/lean/core/transports/` — `mcp.py` (FastMCP), `api.py` (FastAPI), `cli.py` (Typer), `yaml_loader.py`, `health.py`, `builder.py`, `registration.py`.
- `src/lean/core/auth/` — bearer-token middleware.
- `src/lean/core/eval/` — retrieval evaluation harness.
- `src/lean/core/adapters.py` — shared `@mcp_tool` / `@rest_route` / `@cli_command` decorators.
- `src/lean/domains/` — built-in domain adapters (`pdf_lss/`, `code/`, `web/`).
- `src/lean/cli.py` — universal entry point: `lean --config <yaml> <command>`.
- `configs/` — three example YAML manifests.
- `db/schemas/` — SQL migrations.
- `scripts/` — `marker_server.py`, `build_eval_dataset.py`, `canonical-queries.json`.

### Universal CLI surface

Every domain exposes the 5 universal commands + domain-specific ones:

- `lean --config <yaml> db-init` — apply SQL migrations
- `lean --config <yaml> health` — check DB + embedder + optional VLM/LLM
- `lean --config <yaml> mcp-serve` — start MCP server
- `lean --config <yaml> api-serve` — start FastAPI REST API
- `lean --config <yaml> eval` — run retrieval evaluation

## Engineering rules

- Python 3.12+, uv-managed
- Strict mypy, ruff (line-length 100, includes `S` bandit rules)
- **TDD:** write test first → watch fail → implement → watch pass → refactor
- Coverage gate: `fail_under = 80` enforced in pyproject.toml
- `from __future__ import annotations` in all modules
- One responsibility per module, independently testable
- **File-size ceiling ~500 LOC, function-size ceiling ~50 LOC**
- Heavy operations run via `anyio.to_thread.run_sync` (event loop)
- Singletons: `@lru_cache` (`get_embedder`, `_get_reranker`); manual `_settings_cache` dict for `get_settings` / `get_llm`
- Validation at startup: API key min 16 chars, ports 1–65535, `chunk_hard_cap > chunk_target_max`
- Model revisions pinned to SHA hashes (`embedding_model_revision`, `rerank_model_revision`)
- Ingest paths confined to `settings.corpus_root` (default `data/`)
- Optional ML deps isolated: `--extra local-models` / `--extra marker` / `--extra web`

## Design discipline (KISS / YAGNI / DRY / SOLID)

- **KISS** — 30-line function > 300-line "flexible" one.
- **YAGNI** — every column / config field must have a reader within 1 commit.
- **DRY** — shared helpers in one place: `lean.core.adapters` for decorators, `lean.core.extraction` for pipeline primitives.
- **SOLID-S** — services own orchestration, stores own CRUD, transports own I/O.
- **SOLID-O** — extraction backends pluggable via `Extractor` Protocol; VLM provider-agnostic via `VLMClient` Protocol.
- **SOLID-D** — embedder + VLM use Protocol + factory.

## Single-declaration rule (settings)

- `CoreSettings` is the canonical source of truth for app config.
- Env vars take precedence over Python defaults.
- The YAML overlay flows through a single helper, `_apply_overlay_to_kwargs` (in `settings.py`), reused by `yaml_loader._apply_settings_overrides`.
- Marker/OCR tunables live in `CoreSettings` (`marker_remote_url`, `marker_force_ocr`, `ocr_model`, `ocr_dpi`, `ocr_timeout_s`, `ocr_max_tokens`, `ocr_batch_size`) and are projected onto each extractor adapter by `_merge_extractor_defaults`. Declare them once in `settings.marker.*` / `settings.ocr.*`.
- `vlm.enabled` lives **only** at the top level of the YAML (`vlm.enabled: true|false`). `settings.vlm.*` holds transport-level fields only (`base_url`, `model`, `api_key`, `detail`, `timeout_s`, `max_concurrency`, `max_tokens`, `disable_thinking`).

## Validation

- `make verify` — ruff + mypy + pytest (unit only) + coverage ≥ 80%
- `make verify-all` — all tests including integration/e2e
- Integration tests need `SUPABASE_DB_URL` env var
- E2E tests need full stack running (Supabase + GPU servers)
- CI: 3 jobs (verify matrix Python 3.12+3.13, security pip-audit+trivy+CodeQL, integration with Postgres)

## Documentation

- `README.md` — onboarding entry point
- `docs/configuration.md` — every `CoreSettings` field, defaults, validators
- `docs/operations.md` — Docker, healthcheck, post-ingest reindex
- `docs/architecture.md` — pipeline + directory layout
- `docs/evaluation.md` — `lean eval` methodology + caveats
- `docs/limitations.md` — known caveats + closed backlog items
- `docs/decisions/` — ADR scaffolding (write new ADRs here when needed)
- `/home/sl/dev/_ai/docs/` — workspace-level durable knowledge

**Rule:** when you add a field to `CoreSettings`, a CLI command, or an
MCP tool, update the corresponding doc page in the same commit.
Verify with `make verify`.

## Key decisions (load-bearing)

- **Extraction fallback chain:** marker-pdf → Unlimited-OCR → markitdown. Marker returns figures; OCR/markitdown do not.
- **Marker remote GPU:** pure-Python HTTP wrapper around `PdfConverter` (44× faster). Server vendored at `scripts/marker_server.py`. Configure via `settings.marker.remote_url`.
- **VLM default:** MiniMax M3 via API. `disable_thinking: true` for clean JSON. Local Ollama Qwen3.5 fallback supported. Send images to a third-party — disable for PII corpora.
- **Transport ports (M5):** `mcp_http_port` and `api_port` have **no Python defaults**. Source: `settings.transport` in YAML or `MCP_HTTP_PORT` / `API_PORT` env.
- **LLM key alias (C-3):** `Settings.llm_api_key` reads `MINIMAX_API_KEY` (the name `.env.example` uses). Without this alias, every LLM feature silently no-ops.
- **Chunk types:** `text` (default) and `image` (VLM-described). `CHUNK_TYPE_TEXT` / `CHUNK_TYPE_IMAGE` constants in `lean.core.models.schemas`; `VALID_CHUNK_TYPES` is derived.
- **Provenance metadata:** every chunk tracks `embedding_dim`; image chunks track `image_hash` and `provenance_model`.
- **LiquidAI/LFM2.5-Embedding-350M** (1024-dim), remote Ollama → local CPU fallback. Pinned to SHA.
- **Bearer token auth** on HTTP transport (`hmac.compare_digest`); stdio is local-trusted.
- **Multi-stage Docker build** with non-root user (uid=1000).

See `docs/limitations.md` for runtime caveats.

## Quick verification checklist

Before claiming any task complete:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/lean
uv run pytest -m 'not integration and not e2e and not slow' --cov=lean
```

If any check cannot run, state the gap explicitly in the report
(per `verification-before-completion` and the doc-update rule above).