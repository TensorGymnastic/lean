# ADR-0006: Composition by injection (DomainHooks dataclass)

## Status

Accepted

## Date

2026-07-24

## Context

`_install_vlm_hooks` in `yaml_loader.py` wires domain behavior (VLM
image description, custom metadata extraction, chunk-type classification)
by mutating module globals:

```python
ing._domain_describe_one = _describe_one        # services.ingestion
md_mod._custom_extractor = extractor             # extraction.metadata
set_chunk_type_for(_heading_for_chunk)           # extraction.pipeline_helpers
```

This forces "one process = one domain" as an unstated invariant.
Running two domains in the same interpreter silently corrupts state —
the second domain's hooks overwrite the first's.

## Decision

Introduce a frozen, slotted `DomainHooks` dataclass carried by `Pipeline`:

```python
@dataclass(frozen=True, slots=True)
class DomainHooks:
    describe_one: Callable[..., Any] | None = None
    custom_extractor: Callable[[Path], Any] | None = None
    heading_for_chunk: Callable[[ChunkResult], str] | None = None
```

`Pipeline.__init__` accepts an optional `hooks: DomainHooks | None`
parameter (default `None` for backward compatibility). Services read
domain behavior from `pipeline.hooks` instead of module globals.

`_install_vlm_hooks` becomes `_build_domain_hooks`: a pure function
that returns a `DomainHooks` instance instead of mutating globals.

## Alternatives considered

### Option A: Constructor hooks dict
- Pass a `dict[str, Callable]` through the `Pipeline` constructor.
- Pros: no new type.
- Cons: weakly documented, typo-prone, missing-key behavior is easy to
  get wrong in a strict-typed codebase.
- Rejected: a typed dataclass is more idiomatic and safer.

## Consequences

- Two `Pipeline` instances can hold different hooks without cross-talk.
- The `set_pipeline()` singleton still enforces one active pipeline per
  process; concurrent multi-domain serving would require removing that
  singleton separately.
- Services gain an optional parameter (`pipeline.hooks.*`) but the
  public `ingest_pdf(path)` API is unchanged.
- Existing tests that monkeypatch `ingestion._domain_describe_one`
  must switch to constructing a `Pipeline` with `hooks=DomainHooks(...)`.

## References

- Code: `src/lean/core/extraction/base.py`, `src/lean/core/transports/yaml_loader.py`
- Related: Oracle consultation session 2026-07-24 (Option B recommended)
