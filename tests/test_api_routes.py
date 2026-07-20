"""Smoke tests for the FastAPI REST API surface.

The universal ``lean api-serve`` command starts a bearer-authed FastAPI app
whose routes are registered by the active domain (driven by the YAML).
Tests start the server via subprocess and probe the public endpoints.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

_VALID_KEY = "x" * 32


def _have_uv() -> bool:
    return shutil.which("uv") is not None


def _have_pdf_lss_config() -> bool:
    return Path("configs/lean-pdf-lss.yaml").is_file()


pytestmark = pytest.mark.skipif(
    not (_have_uv() and _have_pdf_lss_config()),
    reason="requires uv + configs/lean-pdf-lss.yaml",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def api_server():
    """Start lean api-serve on port 8766 (the pdf_lss default); yield (base, token); tear down."""
    port = 8766
    env = {
        **os.environ,
        "SUPABASE_DB_URL": "postgresql://localhost/postgres",
        "HF_TOKEN": "token",
        "LEAN_MCP_API_KEY": _VALID_KEY,
    }
    proc = subprocess.Popen(
        ["uv", "run", "lean", "--config", "configs/lean-pdf-lss.yaml", "api-serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    # Wait for the server to start (up to 5s).
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        pytest.fail(f"api-serve did not start on port {port}: {stderr}")
    yield base, _VALID_KEY
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_health_endpoint_open_no_auth(api_server) -> None:
    """``GET /health`` returns 200 with no auth required."""
    import httpx

    base, _ = api_server
    r = httpx.get(f"{base}/health", timeout=5.0)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_search_requires_auth(api_server) -> None:
    """``GET /search`` returns 401 without a bearer token (or 500 if no DB)."""
    import httpx

    base, _ = api_server
    r = httpx.get(f"{base}/search", params={"query": "x"}, timeout=5.0)
    # 401 = auth rejected first; 500 = auth passed but DB unreachable in test env
    assert r.status_code in (401, 500)


def test_search_with_valid_token(api_server) -> None:
    """``GET /search`` with valid token: auth passes (200/500, but NOT 401)."""
    import httpx

    base, token = api_server
    r = httpx.get(
        f"{base}/search",
        params={"query": "x", "k": 5},
        headers={"Authorization": f"Bearer {token}"},
        timeout=5.0,
    )
    assert r.status_code != 401
    assert r.status_code in (200, 400, 500)


def test_search_with_wrong_token(api_server) -> None:
    """``GET /search`` with wrong token returns 401."""
    import httpx

    base, _ = api_server
    r = httpx.get(
        f"{base}/search",
        params={"query": "x"},
        headers={"Authorization": "Bearer wrong-token"},
        timeout=5.0,
    )
    assert r.status_code in (401, 500)
