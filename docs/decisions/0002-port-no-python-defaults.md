# ADR-0002: Transport ports (mcp_http_port, api_port) have no Python defaults

## Status

Accepted. Supersedes the prior M5 closure claim in
`docs/limitations.md:245-249` (which was aspirational — the test
suite still asserted Python defaults; the actual closure landed in
commit `79ec908`).

## Date

2026-07-20

## Context

Before this decision, `CoreSettings.mcp_http_port` and
`CoreSettings.api_port` had Python defaults of `8765` and `8766`
respectively. Two problems followed:

1. A YAML declaring `settings.transport.{mcp,api}_port` and the
   corresponding Python defaults disagreed, the Python defaults won,
   and the operator's port assignment was silently ignored. A
   deployment that put two services on different ports and shipped a
   custom YAML would still bind to 8765/8766.
2. The "settings is the single source of truth" promise was broken:
   the application would start without either env or YAML supplying
   these fields, masking misconfiguration until runtime.

`CoreSettings` already enforced validation at startup (API key min
length, ports in 1..65535, `chunk_hard_cap > chunk_target_max`). The
ports were exceptions — they had defaults.

The audit's C-2 finding was that the prior fix declared the work done
without removing the defaults; tests still relied on them.

## Decision

`mcp_http_port` and `api_port` are now declared **without Python
defaults**. Validation rejects `CoreSettings()` construction if neither
`MCP_HTTP_PORT` / `API_PORT` env vars nor a YAML overlay supplied
them. The `tests/conftest.py` autouse fixture supplies `8765`/`8766`
for the unit suite so tests still run without env munging.

```python
mcp_http_port: int = Field(validation_alias="MCP_HTTP_PORT")
api_port: int = Field(validation_alias="API_PORT")
```

The `_apply_overlay_to_kwargs` helper treats empty-string YAML values
as "skip, do not clobber env" so an operator with
`settings.transport: { mcp_http_port: "", api_port: "" }` and
`MCP_HTTP_PORT=9000` env still binds to 9000.

## Alternatives considered

### Keep Python defaults, assert they equal the YAML

- Pros: less code churn; existing tests don't need the autouse fixture.
- Cons: the original bug class returns the moment a deployment
  disagrees with the Python default. Centralisation promise stays
  broken.
- Rejected: doesn't fix the audit finding.

### Read ports only from env (drop YAML support)

- Pros: env-only is operationally simpler.
- Cons: operators who'd rather not set env vars (local dev, CI test
  runs, single-binary deploys) lose the option of declaring ports in
  YAML. The current contract — env overrides YAML — is the right one
  to keep.
- Rejected: reduces flexibility without solving the original bug.

## Consequences

- Every `lean` invocation MUST come from either `--config <yaml>` or
  `LEAN_CONFIG` env or env vars carrying `MCP_HTTP_PORT` /
  `API_PORT`. CLI startup is loud and immediate.
- Unit tests that construct `CoreSettings` directly need the autouse
  fixture in `tests/conftest.py` (or `Settings(mcp_http_port=8765,
  api_port=8766)` at the call site). Refactor of test fixtures is
  mechanical but unavoidable.
- The YAML key `mcp_port` was renamed to `mcp_http_port` to match
  the field name; clients with the old key see the field fall through
  to env (and validation fails on missing env). One-time migration.

## References

- Code: `src/lean/core/config/settings.py:121-124`
- Tests: `tests/conftest.py` `_default_transport_ports`,
  `tests/test_settings.py::test_transport_ports_require_env`
- Config: `configs/lean-{pdf-lss,code,web}.yaml:transport`
- Supersedes: `docs/limitations.md:245-252` (M5 false-claim closure)
- Audit finding: `_ai/workspace/lean-audit-2026-07-19/AUDIT.md` C-2
