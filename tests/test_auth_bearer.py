"""Tests for the BearerTokenMiddleware."""

from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_bearer_middleware_accepts_valid_token() -> None:
    """Request with correct Bearer token passes through."""
    from lean.core.auth.bearer import BearerTokenMiddleware

    received: list[dict] = []

    async def mock_app(scope, receive, send):
        received.append(scope)

    middleware = BearerTokenMiddleware(mock_app, token="secret-key")

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [(b"authorization", b"Bearer secret-key")],
    }

    async def receive():
        return {"type": "http.request", "body": b""}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)

    assert len(received) == 1
    assert len(sent) == 0


@pytest.mark.asyncio
async def test_bearer_middleware_rejects_missing_token() -> None:
    """Request without Authorization header gets 401."""
    from lean.core.auth.bearer import BearerTokenMiddleware

    async def mock_app(scope, receive, send):
        pytest.fail("Should not reach app")

    middleware = BearerTokenMiddleware(mock_app, token="secret-key")

    scope = {"type": "http", "method": "GET", "path": "/test", "headers": []}

    async def receive():
        return {"type": "http.request", "body": b""}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)

    assert sent[0]["status"] == 401


@pytest.mark.asyncio
async def test_bearer_middleware_rejects_wrong_token() -> None:
    """Request with wrong token gets 401."""
    from lean.core.auth.bearer import BearerTokenMiddleware

    async def mock_app(scope, receive, send):
        pytest.fail("Should not reach app")

    middleware = BearerTokenMiddleware(mock_app, token="secret-key")

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [(b"authorization", b"Bearer wrong-key")],
    }

    async def receive():
        return {"type": "http.request", "body": b""}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)

    assert sent[0]["status"] == 401
    body = json.loads(sent[1]["body"])
    assert body["error"] == "unauthorized"


@pytest.mark.asyncio
async def test_bearer_middleware_passes_non_http() -> None:
    """Non-HTTP requests (e.g. lifespan) pass through without auth check."""
    from lean.core.auth.bearer import BearerTokenMiddleware

    received: list[dict] = []

    async def mock_app(scope, receive, send):
        received.append(scope)

    middleware = BearerTokenMiddleware(mock_app, token="secret-key")

    scope = {"type": "lifespan"}

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(message):
        pass

    await middleware(scope, receive, send)

    assert len(received) == 1
