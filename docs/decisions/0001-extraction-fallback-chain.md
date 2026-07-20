# ADR-0001: PDF extraction fallback chain

## Status

Accepted

## Date

2026-07-20

## Context

`lean` ingests user-supplied PDFs. The set of valid PDFs spans:

- clean digital text (table of contents, headings, flowing body) — trivial for marker-pdf
- scanned image-only PDFs — marker with `force_ocr: true` is the only local path
- mixed / malformed / encrypted PDFs — every backend refuses in its own way

The naive approach is to pick one extractor and document its failure
modes. The audit-mandated contract is "every backend fails visibly so the
ingestion pipeline can fall through". Three backends are wired in series,
ordered by quality and cost:

1. `datalab-to/marker` — surya OCR + texify. Highest quality, slowest on CPU.
2. `baidu/Unlimited-OCR` — remote transformers-based OCR. Quality close to
   marker for image-only PDFs; fast on GPU.
3. `Microsoft MarkItDown` — pure-Python PDF→markdown. Handles simple digital
   PDFs; no figure extraction; no OCR quality.

The extraction ordering is the contract. Changing it is expensive
because reviewers and downstream users reason about it.

## Decision

The pipeline instantiates extractors in YAML `extractors:` order
(`Pipeline._extractors`). Each `Extractor.is_configured()` is checked
first; configured extractors are tried in order until one returns an
`ExtractionResult` without raising `BackendUnavailable`. Marker returns
figures, OCR/markitdown do not. The orchestrator preserves the
`extraction_method` in `IngestResult` so callers can detect fall-throughs.

```python
# Universal ordering — see src/lean/core/extraction/base.py:Pipeline
for extractor in self._extractors:
    if not extractor.is_configured():
        continue
    try:
        return extractor.extract(pdf_path)
    except BackendUnavailable as e:
        logger.warning("extractor %s unavailable: %s", extractor, e)
        continue
```

## Alternatives considered

### One-extractor (marker-only)

- Pros: simpler pipeline; one backend to operate; one quality bar.
- Cons: scanned PDFs are un-ingestable; remote-Ollama-only deployments
  pay the 44× CPU penalty even when the GPU marker server is up.
- Rejected: a corpus with mixed-quality PDFs is the headline use case.

### Different ordering (markitdown first, marker fallback)

- Pros: markitdown is pure-Python — zero infrastructure cost for the
  common digital-text case.
- Cons: digital PDFs are already fine with marker; putting markitdown
  first means losing figures (which marker extracts) for the majority of
  clean PDFs, while saving nothing for scanned ones.
- Rejected: quality regression for the common case.

### OCR-first

- Pros: works on scanned PDFs in any environment.
- Cons: slower, lower-quality on clean digital PDFs (regression),
  introduces a third-party OCR call before the primary extractor.
- Rejected: same quality regression as above.

## Consequences

- Each backend must implement `is_configured()` + `extract()` from the
  Protocol. Adding a fourth backend is one entry in `extractors:` +
  one adapter file.
- `IngestResult.extraction_method` lets the calling code detect silent
  fall-throughs (marker offline → OCR successful but no figures).
- Operators running marker on GPU get a 44× speedup vs CPU with no code
  changes — only a YAML edit.

## References

- Code: `src/lean/core/extraction/base.py:Pipeline`
- Code: `src/lean/core/extraction/{marker,ocr,markitdown}.py`
- Config: `configs/lean-pdf-lss.yaml:51-58`
- Tests: `tests/test_extractor_settings_w7.py`, `tests/test_ocr_extractor.py`
- Related: `docs/limitations.md` (silent fall-through caveat)
