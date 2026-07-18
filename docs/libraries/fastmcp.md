# fastmcp

The MCP server framework lean uses (v3.4.4).

## What lean uses

8 tools, 4 resources, 3 prompts — registered via decorators on a
`FastMCP("lean")` instance in `src/lean/mcp_server/`. Two transports:
`stdio` (default, local-trusted) and `http` (bearer-authed).

## Registration

```python
from fastmcp import FastMCP

mcp = FastMCP("lean")

@mcp.tool
def search(query: str, k: int = 5) -> list[dict]:
    """Semantic + hybrid search of the corpus."""
    ...

@mcp.resource("lean://documents/{doc_id}/chunks")
def get_chunks(doc_id: str) -> str:
    """Return all chunks for a document as JSON."""
    ...

@mcp.prompt
def lean_qa(question: str) -> str:
    """Generate a Lean Six Sigma Q&A prompt."""
    ...
```

Tool/Resource/Prompt docstrings become the MCP description — write them
as user-facing API docs.

## Transports

| Transport | Use case | Auth |
|---|---|---|
| `stdio` | Local MCP clients (Claude Desktop, opencode) | None — local-trusted |
| `http` | Remote HTTP clients | Bearer token via `LEAN_MCP_API_KEY` |

`mcp.run(transport="http", host="0.0.0.0", port=8000)` for HTTP. For
production with auth and TLS, see gofastmcp.com/deployment/http.

## Gotchas

- **Async tools:** use `async def` — fastmcp runs them on its event loop.
  Wrap blocking work in `anyio.to_thread.run_sync` so the loop stays
  responsive (this is enforced by lean engineering rules).
- **Singletons:** `get_settings`, `get_embedder`, `get_llm`, `_get_reranker`
  use `@lru_cache`. Don't construct fresh instances per call — the
  embedder alone loads ~500 MB torch.
- **Tool return types:** return JSON-serializable types (str, dict,
  list, Pydantic models via `.model_dump(mode="json")`). fastmcp
  generates the JSON schema from the type hints — be precise.
- **Bearer auth on HTTP:** see [`fastapi.md`](fastapi.md) — lean's
  middleware pattern reuses the same `hmac.compare_digest` approach.

## Resources

- Docs: <https://gofastmcp.com>
- lean entrypoint: `src/lean/mcp_server/__main__.py`
- lean tool/resource/prompt files: `src/lean/mcp_server/{tools,resources,prompts}.py`
