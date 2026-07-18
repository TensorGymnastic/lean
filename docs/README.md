# lean documentation

Reference docs for `lean` MCP server. The README is the onboarding entry point;
this directory is for **why** and **how it actually behaves** — context that
doesn't fit in a quick-start.

## Reference

- [`configuration.md`](configuration.md) — every `Settings` field, defaults,
  validators, and gotchas (the single source of truth for `config.yaml`)
- [`operations.md`](operations.md) — Docker ports, healthcheck semantics,
  post-ingest reindex, reingest dedup behavior, monitoring
- [`evaluation.md`](evaluation.md) — `lean eval` methodology, pipeline gap,
  non-determinism caveat
- [`architecture.md`](architecture.md) — retrieval pipeline, directory layout,
  transport tier pattern
- [`limitations.md`](limitations.md) — known caveats (page fields NULL, eval
  pseudo-queries, dedup orphans)
- [`ocr-server-deployment.md`](ocr-server-deployment.md) — GPU host setup for
  `baidu/Unlimited-OCR` + Ollama embedding model (already exists)

## Decisions

- [`decisions/`](decisions/) — Architecture Decision Records (ADRs) for
  load-bearing choices not obvious from the code

## Archive

- [`archive/`](archive/) — superseded planning artifacts (do not delete;
  retained for historical context)
