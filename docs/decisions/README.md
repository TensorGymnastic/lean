# Decisions

Architecture Decision Records (ADRs) for load-bearing choices that aren't
obvious from the code. This is the canonical format when you make a new
architectural decision — for *existing* behavior, document it in
[`limitations.md`](../limitations.md) or the relevant guide instead.

## Index

| ADR | Status | Decision |
|---|---|---|
| [0001](0001-extraction-fallback-chain.md) | Accepted | PDF extraction fallback chain (marker → Unlimited-OCR → markitdown) |
| [0002](0002-port-no-python-defaults.md) | Accepted | `mcp_http_port` / `api_port` have no Python defaults (env or YAML required) |
| [0003](0003-minimax-api-key-alias.md) | Accepted | `Settings.llm_api_key` reads `MINIMAX_API_KEY` via validation_alias |
| [0004](0004-vlm-single-enabled-flag.md) | Accepted | `vlm.enabled` lives only at top-level YAML — single source of truth |
| [0005](0005-dedup-by-sha256.md) | Accepted | Document dedup is by `source_sha256`, not by path |
| [0006](0006-composition-injection.md) | Accepted | Domain behavior threaded via `DomainHooks` dataclass, replacing module-global mutation |
| [0007](0007-eliminate-pipeline-singleton.md) | Accepted | Pipeline passed as explicit parameter to service functions |

Pending / not-yet-decided:

- LFM2.5-Embedding-350M + asymmetric prompts → still in `configuration.md`
  §embedding (also a candidate for ADR; promote when a second embedder
  arrives).
- Unlimited-OCR via transformers (NOT vLLM) → still in
  `ocr-server-deployment.md` (vendor-pinned contract; revisit when
  GPU memory pressure becomes user-visible).
- Rerank enabled in `lean-pdf-lss.yaml` config, disabled in `code` /
  `web` → stays in `configuration.md` (per-domain config, not a
  cross-cutting decision).
- `ivfflat` post-bulk-ingest `REINDEX` → stays in `operations.md`
  (operational runbook, not architectural).
- Contextual Retrieval irreversibility → stays in `limitations.md`
  (known-cost disclosure, not a decision to reconsider).
- Pseudo-eval methodology → stays in `evaluation.md` (the doc IS the
  decision; an ADR would duplicate).
- REST API as near-full mirror (7/8) → stays in `limitations.md`
  (audit disclosure, not a design choice worth reaffirming).

If a future decision is **expensive to reverse**, write it as an ADR
rather than burying it in a guide.

---

## When to write an ADR

Write an ADR when the choice:

- Affects a public API or schema
- Would be expensive to reverse (months of work, breaking change)
- Is the kind of thing future-you will second-guess without context
- Was explicitly debated (multiple viable options were considered)

**Don't** write an ADR for:

- A bug fix (use a commit message)
- A refactor with no behavior change (use a commit message)
- Documenting existing silent behavior (use [`limitations.md`](../limitations.md))
- Adding a field to config.yaml (use [`configuration.md`](../configuration.md))

When in doubt: write the ADR. The cost of writing one is 10 minutes;
the cost of *not* having one when someone tries to reverse the decision
is hours of re-investigation.

---

## Template

```markdown
# ADR-NNNN: <short imperative title>

## Status

Accepted | Superseded by ADR-XXXX | Deprecated

## Date

YYYY-MM-DD

## Context

What is the issue? What constraints apply? What forces are in tension?
(2-5 sentences. The "why are we even making this choice?")

## Decision

What did we choose? Be specific — name the technology, version, config
knob, or pattern. One paragraph.

## Alternatives considered

### <Alternative A>
- Pros: ...
- Cons: ...
- Rejected: <one-line summary>

### <Alternative B>
- Pros: ...
- Cons: ...
- Rejected: <one-line summary>

## Consequences

What becomes easier? What becomes harder? What new obligations do we
take on? (operational, performance, security, maintenance)

## References

- Code: <file:line>
- Config: <config key>
- Tests: <test file>
- Related: <link to other ADR or doc>
```

Filename: `NNNN-short-kebab-case-title.md` (zero-padded 4-digit number).
Once accepted, don't delete — supersede with a new ADR that references
the old one.

---

## Lifecycle

```
PROPOSED → ACCEPTED → (SUPERSEDED | DEPRECATED)
```

- **Don't delete old ADRs.** They capture historical context.
- When a decision changes, write a new ADR that references and supersedes
  the old one (e.g. "Supersedes ADR-0002").
