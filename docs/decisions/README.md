# Decisions

Architecture Decision Records (ADRs) for load-bearing choices that aren't
obvious from the code. This is the canonical format when you make a new
architectural decision — for *existing* behavior, document it in
[`limitations.md`](../limitations.md) or the relevant guide instead.

## Index

_No formal ADRs yet. The choices documented below currently live in their
relevant guide pages — use them as reference, not as authoritative decisions._

| Decision | Where it lives |
|---|---|
| LFM2.5-Embedding-350M + asymmetric prompts | [`configuration.md` §embedding](../configuration.md#embedding) |
| Unlimited-OCR via transformers (NOT vLLM) | [`ocr-server-deployment.md`](../ocr-server-deployment.md) |
| Rerank enabled in this repo's config | [`configuration.md` §retrieval](../configuration.md#retrieval) |
| `ivfflat` post-bulk-ingest `REINDEX` | [`operations.md` §post-ingest-reindex](../operations.md#post-ingest-reindex) |
| Contextual Retrieval irreversibility | [`limitations.md` §contextual-retrieval-is-irreversible](../limitations.md#contextual-retrieval-is-irreversible) |
| Pseudo-eval methodology | [`evaluation.md`](../evaluation.md) |
| REST API as partial mirror | [`limitations.md` §rest-api-is-a-partial-mirror](../limitations.md#rest-api-is-a-partial-mirror-not-full-parity) |
| Dedup by SHA-256 | [`operations.md` §reingest-semantics](../operations.md#reingest-semantics) + [`limitations.md` §dedup](../limitations.md#dedup-is-by-sha-256-not-by-path) |

If a future decision is **expensive to reverse**, write it as an ADR below
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
