# ADR-0004: VLM enrichment is opt-in via `vlm.enabled` (single source of truth)

## Status

Accepted

## Date

2026-07-20

## Context

The VLM (Vision-Language Model) enrichment pipeline ingests chart and
figure images at extract time, describing them in structured JSON
and embedding the description alongside the document text. Two
operational risks follow:

1. **PII / trade-secret leakage.** The VLM call ships chart images
   to a third-party endpoint (MiniMax M3 by default; Ollama on
   local GPU otherwise). For a corpus with restricted content, that
   is unacceptable.
2. **Cost.** VLM calls are billed per image. A user who forgot they
   had VLM enabled on a 500-page book with 30 figures pays for 15,000
   descriptions.

Originally, the codebase maintained `vlm.enabled` in *two* places:

- `CoreSettings.vlm_enabled` (Python default `False`)
- `DomainConfig.vlm.enabled` (YAML under top-level `vlm:` block)

`DomainConfig.vlm.enabled` gated the install of the VLM hook
(`yaml_loader._install_vlm_hooks`). `CoreSettings.vlm_enabled`
gated the startup warning ("VLM enabled — images will be sent to
…"). The two were independent.

A user who set `settings.vlm.enabled: true` but forgot the top-level
`vlm: enabled: true` got a startup warning that VLM would send
images, but no actual VLM hook installed — silent feature absence.
A user who set the top-level flag but not `settings.vlm.enabled` got
the reverse: hook active but no warning that images leave the host.

## Decision

`vlm.enabled` lives in **exactly one place**: the top-level `vlm:`
block of the domain YAML. `CoreSettings.vlm_enabled` is removed;
the startup warning reads directly from
`settings.domain_config["vlm"]["enabled"]`.

```yaml
# Correct: single declaration
domain:
  name: lean-pdf-lss
vlm:
  enabled: false         # this is the only flag
  base_url: "..."
```

```yaml
# Anti-pattern (removed): no `settings.vlm.enabled` anywhere
```

The two responsibilities — *"did the operator opt in?"* and *"what's
the endpoint?"* — are correctly answered by one block. The startup
warning (`if vlm_enabled: logger.warning("VLM enabled — images will
be sent to %s", base_url)`) and the hook install
(`if domain_config.vlm.enabled: install_vlm_hooks(...)`) both read
the same flag.

The pipeline enforces fail-fast: if `vlm.enabled: true` but
`vlm.base_url` or `vlm.model` is empty, `_validate_cross_field`
raises at construction time.

## Alternatives considered

### Two flags, "redundant for safety"

- Pros: defensive — both must be true.
- Cons: silent feature absence when one side is missed; user
  has to read two places to know whether VLM is active; the
  audit identified this as a real confusion.
- Rejected: matches the original bug.

### Auto-detect from `vlm.base_url`

- Pros: zero config.
- Cons: violates least surprise — a placeholder URL like
  `http://example.com/never-called/` would silently activate the
  feature.
- Rejected: opt-in matters more than convenience here.

### Make VLM always on

- Pros: simpler.
- Cons: rejects corpora with PII / trade-secret concerns, which the
  project explicitly serves (LSS Six Sigma consulting material).
- Rejected: not aligned with the project's user base.

## Consequences

- A `.yaml` declaring `vlm.enabled: true` and a non-empty
  `vlm.base_url` is the complete contract. No second flag to set.
- A `vlm.enabled: true` with empty `vlm.base_url` raises at startup
  with a clear error.
- The startup warning fires when (and only when) VLM will actually
  run — no false alarms, no silent missing features.
- The compliance caveat ("disable for corpora with PII / trade-secret
  concerns") reduces to one boolean in the operator's YAML.

## References

- Code: `src/lean/core/config/settings.py` (`vlm_enabled` field removed),
  `src/lean/core/config/domain_config.py:VLMConfig`,
  `src/lean/core/transports/yaml_loader.py:_install_vlm_hooks`
- Config: `configs/lean-pdf-lss.yaml:vlm:` (top-level only)
- Tests: `tests/test_settings.py::test_vlm_enabled_requires_base_url_and_model`,
  `tests/test_services_ingestion.py`
- Doc: `docs/configuration.md` § vlm
- Audit finding: `_ai/workspace/lean-audit-2026-07-19/AUDIT.md` R-2
