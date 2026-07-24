# ADR-0007: Pass Pipeline as Explicit Parameter to Service Functions

## Status

Accepted 2026-07-24

## Context

The `_pipeline_cache` / `set_pipeline()` / `get_pipeline()` pattern is a
module-level singleton in `lean.core.extraction.base`. The ingestion
service (`lean.core.services.ingestion`) reached into this global at
three call sites:

1. `_run_extraction` — `get_pipeline().extract(pdf_path)`
2. `_persist_ingest` — `get_pipeline().hooks.heading_for_chunk`
3. `ingest_pdf` — `get_pipeline().hooks.describe_one`

This made unit testing require `patch("lean.core.services.ingestion.get_pipeline")`
on every test that exercises the ingestion path. Seven tests previously
did this.

## Decision

Pass `Pipeline` as an explicit keyword-only parameter to the service
functions that need it (`_run_extraction`, `_persist_ingest`,
`ingest_pdf`, `reingest`). The tool-layer functions (in domain
`tools.py`) call `get_pipeline()` once and pass the pipeline through —
they are the bridge between the framework and the service layer.

The `_pipeline_cache` singleton remains in `extraction.base` as the
process-level registration point (set once at startup by
`build_from_yaml`). The service layer no longer touches it directly.

Additionally:
- The `get_pipeline()` re-export function in `yaml_loader.py` was removed
  (the module now imports directly from `extraction.base`).
- The redundant `set_pipeline()` call in `builder.py` was removed
  (`build_from_yaml` already sets it before constructing the builder).

## Consequences

- Service-layer functions accept `pipeline: Pipeline` as a keyword arg.
- Tests pass `pipeline=mock_pipeline` directly — no `patch()` needed.
- Tool functions remain thin: `return await _ingest_pdf(path, pipeline=get_pipeline())`.
- The singleton persists as the startup wiring mechanism — multiple
  domains cannot coexist in the same process. This is accepted as a
  trade-off of the current tool registration model.
