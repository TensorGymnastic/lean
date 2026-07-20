# `web` domain — URL corpus

The built-in URL-scraping corpus. Fetches HTML pages via httpx, extracts
clean markdown with trafilatura, and ingests each page as a single
document.

## Files

| File | Purpose |
|---|---|
| `adapters.py` | `WebPageExtractor` — downloads the URL, runs trafilatura, returns markdown |
| `metadata.py` | URL metadata extractor (host, path, fetch time) |
| `tools.py` | `@mcp_tool` / `@rest_route` / `@cli_command` declarations |

## YAML contract

`configs/lean-web.yaml`:

```yaml
domain:
  name: lean-web
  version: 0.1.0

extractors:
  - adapter: lean.domains.web.adapters.WebPageExtractor
    config: { timeout_s: 30.0 }

tools:
  module: lean.domains.web.tools
```

## CLI surface

- `lean --config configs/lean-web.yaml ingest <url>` — fetch + ingest a single URL
- `lean --config configs/lean-web.yaml ingest-list <file>` — bulk ingest (one URL per line in `<file>`)

`corpus_stats`, `search`, `list-documents`, `delete`, `get-chunk`,
`get-markdown` follow the universal pattern.
