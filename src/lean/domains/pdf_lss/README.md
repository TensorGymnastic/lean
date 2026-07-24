# `pdf_lss` domain — Lean Six Sigma PDF corpus

The built-in PDF corpus domain. Uses marker-pdf (primary), OCR
(secondary), and markitdown (fallback) extractors directly from
`lean.core.extraction`, with optional VLM enrichment for chart/image
chunks.

## Files

| File | Purpose |
|---|---|
| `tools.py` | `@mcp_tool` / `@rest_route` / `@cli_command` declarations; imports universal tools from `lean.core.tools.universal` |
| `metadata.py` | Domain-specific metadata extraction (delegates to core, adds `publisher` heuristic) |
| `parser.py` | JSON + markdown-fence parser used by the chart extractor |
| `chart_extraction_prompt.txt` | Default prompt sent to the VLM for chart description |

## YAML contract

`configs/lean-pdf-lss.yaml` lists:

```yaml
domain:
  name: lean-pdf-lss
  version: 0.1.0

settings:
  ocr: { base_url: "" }
  embedding: { model: LiquidAI/LFM2.5-Embedding-350M, dim: 1024 }

extractors:
  - adapter: lean.core.extraction.MarkerExtractor
    config: {}
  - adapter: lean.core.extraction.UnlimitedOCRExtractor
    config: { hf_token: "" }
  - adapter: lean.core.extraction.MarkitdownExtractor
    config: {}

vlm:
  enabled: true
  prompt_file: ../src/lean/domains/pdf_lss/chart_extraction_prompt.txt
  parser: lean.domains.pdf_lss.parser.parse

tools:
  module: lean.domains.pdf_lss.tools
```

## Adding a 4th domain

To add another PDF-style domain:

1. Create `src/lean/domains/my_domain/` with `tools.py`, optionally `metadata.py`.
2. Drop a YAML manifest in `configs/lean-my-domain.yaml`.
3. Run `uv run lean --config configs/lean-my-domain.yaml --help` to verify the surface.
