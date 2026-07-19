"""Static bearer-token middleware for MCP HTTP transport.

Validates ``Authorization: Bearer <LEAN_MCP_API_KEY>`` against the env var.
stdio transport bypasses auth (process boundary is the trust boundary).
Returns 401 JSON on missing/mismatched token.
"""

from __future__ import annotations

import hmac
import json

from starlette.types import ASGIApp, Receive, Scope, Send


class BearerTokenMiddleware:
    def __init__(self, app: ASGIApp, *, token: str) -> None:
        self._app = app
        self._expected = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        auth_header = headers.get(b"authorization", b"").decode("latin-1")
        token = auth_header.removeprefix("Bearer ").strip()

        if not token or not hmac.compare_digest(token, self._expected):
            await self._send_unauthorized(send)
            return

        await self._app(scope, receive, send)

    async def _send_unauthorized(self, send: Send) -> None:
        body = json.dumps({"error": "unauthorized"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
