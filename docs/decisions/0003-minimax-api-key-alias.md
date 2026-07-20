# ADR-0003: Settings.llm_api_key reads MINIMAX_API_KEY

## Status

Accepted

## Date

2026-07-20

## Context

`lean` supports an optional LLM sidecar for HyDE / multi-query /
Contextual Retrieval features. The sidecar reads its API key from
`Settings.llm_api_key`. The `.env.example` file at the repo root
advertised the env-var name `MINIMAX_API_KEY` (matching the cloud
provider), but `CoreSettings` declared the field without a
`validation_alias`:

```python
llm_api_key: str = Field(default="", description="LLM provider API key")
```

`pydantic-settings` reads env vars case-insensitively and matches them
against the field name. With no alias, `Settings.llm_api_key` looked
for `LLM_API_KEY`. A user who followed `.env.example` literally and set
`MINIMAX_API_KEY=sk-…` got `settings.llm_api_key == ""` at runtime.
`get_llm()` then returns `None` (the field was falsy), and every
LLM-side feature — `llm.contextual_retrieval`,
`llm.multi_query`, `llm.hyde` — silently no-ops.

This is the highest-cost class of bug: a feature declared as
"configured" by `Settings` (passed validation, didn't raise) but
disabled by a field name mismatch. Production users had no signal.

The audit's C-3 finding made the issue explicit; the audit's
verifiability principle ("every field must have a reader within 1
commit") is the rule it violates.

## Decision

`Settings.llm_api_key` gains `validation_alias="MINIMAX_API_KEY"`:

```python
llm_api_key: str = Field(
    default="",
    description="LLM provider API key (MiniMax by default)",
    validation_alias="MINIMAX_API_KEY",
)
```

The `.env.example` continues to declare `MINIMAX_API_KEY`. The
integration test (`tests/test_llm_integration.py`) reads
`MINIMAX_API_KEY` directly via `os.environ.get(...)` and now matches
the field name. The unit test `test_llm_api_key_alias_is_minimax` in
`tests/test_settings.py` locks the contract: setting `MINIMAX_API_KEY`
populates `settings.llm_api_key` and the env-varname-naming change is
observable.

`VLM_API_KEY` was already correctly aligned (the field name and alias
match case-insensitively); no change required there.

## Alternatives considered

### Rename `llm_api_key` to `minimax_api_key`

- Pros: field name matches env var.
- Cons: locks the codebase to the MiniMax provider. Future provider
  rotations require the same kind of rename. Plus: the field exists
  to gate "LLM provider configured", not "MiniMax configured".
- Rejected: provider-specific naming defeats generalization.

### Two fields: `minimax_api_key` + `openai_api_key` + `provider`

- Pros: supports multiple LLM providers cleanly.
- Cons: significant API expansion; not in scope of the audit fix.
- Rejected: defer until a real second-provider use case emerges.

## Consequences

- A user who follows `.env.example` now gets the documented behavior.
  LLM-side features activate when `MINIMAX_API_KEY` is set.
- A user who has been working around the bug by setting both
  `MINIMAX_API_KEY` and `LLM_API_KEY` keeps working — no breaking
  change.
- The pre-existing test that sets `s.llm_api_key = "minimax-key"`
  directly on a `MagicMock` (in `tests/test_factories.py`) keeps
  working — the change is at the env-binding layer, not the field
  semantics.

## References

- Code: `src/lean/core/config/settings.py:103-108`
- Env: `.env.example` (`MINIMAX_API_KEY=`)
- Tests: `tests/test_settings.py::test_llm_api_key_alias_is_minimax`,
  `tests/test_factories.py::test_get_llm_minimax`
- Audit finding: `_ai/workspace/lean-audit-2026-07-19/AUDIT.md` C-3
