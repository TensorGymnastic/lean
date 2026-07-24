# ADR-0007: Eliminate Pipeline Singleton in Service Layer

## Status

Proposed 2026-07-24

## Context

The `_pipeline_cache` / `set_pipeline()` / `get_pipeline()` pattern is a
module-level singleton in `lean.core.extraction.base`. The ingestion
service (`lean.core.services.ingestion`) reaches into this global at
three call sites:

1. `_run_extraction` — `get_pipeline().extract(pdf_path)`
2. `_persist_ingest` — `get_pipeline().hooks.heading_for_chunk`
3. `ingest_pdf` — `get_pipeline().hooks.describe_one`

This makes unit testing require `patch("lean.core.services.ingestion.get_pipeline")`
on every test that exercises the ingestion path. Seven tests currently
do this.

ADR-0006 introduced `DomainHooks` injection but left the pipeline itself
as a singleton.

## Decision

Pass `Pipeline` as an explicit keyword-only parameter to the service
functions that need it (`_run_extraction`, `_persist_ingest`,
`ingest_pdf`, `reingest`). The tool-layer functions (in domain
`tools.py`) call `get_pipeline()` once and pass the pipeline through —
they are the bridge between the framework and the service layer.

The `_pipeline_cache` singleton remains in `extraction.base` as the
process-level registration point (set once at startup by
`build_from_yaml`). It is NOT removed — it is scoped to its role:
"wiring the pipeline at startup." The service layer no longer touches
it directly.

## Consequences

- Service-layer functions accept `pipeline: Pipeline` as a keyword arg.
- Tests pass `pipeline=mock_pipeline` directly — no `patch()` needed.
- The yaml_loader re-export of `get_pipeline` is removed (dead indirection).
- `builder.py`'s redundant `set_pipeline` call is removed (yaml_loader already sets it).
- Tool functions remain thin: `return await _ingest_pdf(path, pipeline=get_pipeline())`.
