# `code` domain — markdown / source-file corpus

The built-in code-domain corpus. Reads `.md`, `.markdown`, `.txt`,
`.rst`, and source files directly as a single-page extraction (no PDF
pipeline needed).

## Files

| File | Purpose |
|---|---|
| `adapters.py` | `MarkdownFileExtractor` — reads the file, wraps content as a single-page `ExtractionResult` |
| `metadata.py` | File-stat-based metadata extractor (`FileMetadata`: path, size, mtime, sha256) |
| `tools.py` | `@mcp_tool` / `@rest_route` / `@cli_command` declarations + `register_*` entrypoints |

## YAML contract

`configs/lean-code.yaml`:

```yaml
domain:
  name: lean-code
  version: 0.1.0

extractors:
  - adapter: lean.domains.code.adapters.MarkdownFileExtractor
    config: {}

tools:
  module: lean.domains.code.tools
```

## CLI surface

- `lean --config configs/lean-code.yaml ingest <file>` — ingest a single file
- `lean --config configs/lean-code.yaml ingest-directory --dir <dir> --pattern "*.md"` — bulk ingest

`corpus_stats`, `search`, `list-documents`, `delete`, `get-chunk`,
`get-markdown` follow the universal pattern from `pdf_lss`.
