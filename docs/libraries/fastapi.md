# FastAPI

Lean's REST transport — a near-full mirror of the MCP tools (7 of 8
endpoints; `reingest` is CLI-only — see
[`limitations.md`](../limitations.md#rest-api-now-mirrors-7-of-8-mcp-tools-near-full-parity)).

## What lean uses

- **`FastAPI(title=..., version=...)`** — single app instance in `src/lean/api/routes.py`
- **`HTTPBearer(auto_error=False)`** — extract Bearer token, no auto-401
- **`@app.exception_handler(ExceptionType)`** — map domain errors to HTTP codes
- **`anyio.to_thread.run_sync(callable)`** — wrap sync services so the
  event loop stays responsive (same pattern as MCP)

## Bearer auth pattern

```python
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import hmac

_security = HTTPBearer(auto_error=False)

async def _verify_token(
    creds: HTTPAuthorizationCredentials | None = Security(_security),
) -> None:
    settings = get_settings()
    if not creds or not hmac.compare_digest(creds.credentials, settings.lean_mcp_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")

@app.get("/search", dependencies=[Depends(_verify_token)])
async def search(
    query: str,
    k: int = 5,
    chunk_type: str | None = None,
) -> list[dict]:
    ...
```

Key points:

- **`auto_error=False`** so missing token returns `None`, not auto-401.
  Lean verifies and raises its own 401 with timing-safe comparison.
- **`hmac.compare_digest`** for constant-time comparison (prevents
  timing attacks).
- **`dependencies=[Depends(...)]`** — applies auth without polluting the
  function signature.

## Exception handlers (domain → HTTP)

```python
@app.exception_handler(ValueError)
async def _value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})

@app.exception_handler(PermissionError)
async def _permission_error_handler(_request: Request, exc: PermissionError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": str(exc)})

@app.exception_handler(FileNotFoundError)
async def _not_found_handler(_request: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})
```

Without these, domain errors return 500. Lean maps: `ValueError` → 400,
`PermissionError` → 403, `FileNotFoundError` → 404.

## Gotchas

- **Path vs query params:** lean uses `query` and `k` as query params
  on `GET /search`. For ingestion, `POST /ingest` uses `path` as a
  query param (not JSON body) — non-obvious. **Security note:** URLs
  are logged by reverse proxies / ALBs / browser history, so `path`
  may leak source PDF filenames (which can carry PII) into access logs.
  A breaking change to a JSON body model is on the roadmap. See
  [`limitations.md`](../limitations.md).
- **Sync services:** FastAPI runs sync route handlers in a thread pool,
  but lean wraps anyway with `anyio.to_thread.run_sync` for explicit
  control (also matches the MCP pattern).
- **OpenAPI auto-docs:** available at `/docs` (Swagger UI) and `/redoc`
  in dev — useful for testing the bearer auth flow.

## Resources

- Docs: <https://fastapi.tiangolo.com>
- lean app: `src/lean/api/routes.py`
- bearer middleware alternative: `src/lean/auth/bearer.py` (ASGI, not used in routes.py)
