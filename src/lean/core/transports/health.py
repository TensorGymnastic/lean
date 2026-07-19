"""Universal health-check probe.

Returns a dict of ``{component: {status: ..., ...}}`` for the operator's
``lean health`` command. Domains can extend by adding their own checks.
"""

from __future__ import annotations

import logging

import httpx

from lean.core.config.settings import CoreSettings

logger = logging.getLogger(__name__)


def _probe(name: str, url: str, path: str, timeout: float) -> dict[str, object]:
    if not url:
        return {"status": "not_configured"}
    try:
        resp = httpx.get(f"{url.rstrip('/')}{path}", timeout=timeout)
        return {"status": "ok" if resp.status_code == 200 else "error", "url": url}
    except Exception as exc:
        return {"status": "error", "url": url, "error": str(exc)}


def _check_database(settings: CoreSettings) -> dict[str, object]:
    from lean.core.store.base import StoreConnection

    try:
        conn = StoreConnection.from_env()
        try:
            with conn.conn.cursor() as cur:
                cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                has_pgvector = cur.fetchone() is not None
        finally:
            conn.close()
        return {"status": "ok", "pgvector": has_pgvector}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def check_health(settings: CoreSettings) -> dict[str, dict[str, object]]:
    """Run all built-in health checks and return a dict."""
    return {
        "ocr": _probe(
            "ocr",
            settings.domain_config.get("ocr", {}).get("base_url", "")
            if settings.domain_config
            else "",
            "/health",
            settings.health_http_timeout,
        ),
        "database": _check_database(settings),
        "ollama": _probe(
            "ollama", settings.embedding_remote_url, "/api/tags", settings.health_http_timeout
        ),
    }


__all__ = ["check_health"]
