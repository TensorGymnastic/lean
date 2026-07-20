# `pdf_lss` domain — Lean Six Sigma PDF corpus

The built-in PDF corpus domain. Wraps marker-pdf (primary), OCR
(secondary), and markitdown (fallback) extractors behind the universal
`Extractor` Protocol, with optional VLM enrichment for chart/image
chunks.

## Files

| File | Purpose |
|---|---|
| `adapters.py` | Thin wrappers around the three core extractors (`MarkerAdapter`, `UnlimitedOCRAdapter`, `MarkitdownAdapter`) |
| `tools.py` | `@mcp_tool` / `@rest_route` / `@cli_command` declarations; `register_mcp` / `register_api` / `register_cli` entrypoints |
| `metadata.py` | Domain-extended `PdfMetadata` (adds `publisher`, override of `_custom_extractor` wiring) |
| `parser.py` | JSON + markdown-fence parser used by the chart extractor |
| `chart_extraction_prompt.txt` | Default prompt sent to the VLM for chart description |

## YAML contract

`configs/lean-pdf-lss.yaml` lists:

```yaml
domain:
  name: lean-pdf-lss
  version: 0.1.0

settings:
  ocr: { base_url: "" }     # remote OCR server URL (empty disables OCR backend)
  embedding: { model: LiquidAI/LFM2.5-Embedding-350M, dim: 1024 }

extractors:
  - adapter: lean.domains.pdf_lss.adapters.MarkerAdapter
    config: { remote_url: "", force_ocr: false }
  - adapter: lean.domains.pdf_lss.adapters.UnlimitedOCRAdapter
    config: { base_url: "http://gpu:8001", model: "baidu/Unlimited-OCR" }
  - adapter: lean.domains.pdf_lss.adapters.MarkitdownAdapter
    config: {}

vlm:
  enabled: true
  prompt_inline: "..."     # or prompt_file: prompts.txt
  parser: lean.core.vlm.prompts.parse_description

tools:
  module: lean.domains.pdf_lss.tools
```

## Adding a 4th domain

To add another PDF-style domain, copy this layout:

1. Create `src/lean/domains/my_domain/` with `adapters.py`, `tools.py`, optionally `metadata.py`.
2. Drop a YAML manifest in `configs/lean-my-domain.yaml`.
3. Run `uv run lean --config configs/lean-my-domain.yaml --help` to verify the surface.
