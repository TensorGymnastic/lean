"""Universal health-check probe.

Returns a dict of ``{component: {status: ..., ...}}`` for the operator's
``lean health`` command. Domains can extend by adding their own checks.
"""

from __future__ import annotations

import logging

import httpx

from lean.core.config.settings import CoreSettings

logger = logging.getLogger(__name__)


def _probe(url: str, path: str, timeout: float) -> dict[str, object]:
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


def _check_ocr(settings: CoreSettings) -> dict[str, object]:
    """Probe the configured OCR backend (returns ``not_configured`` if no URL set)."""
    return _probe(settings.ocr_base_url, "/health", settings.health_http_timeout)


def _check_ollama(settings: CoreSettings) -> dict[str, object]:
    """Probe the configured Ollama endpoint (returns ``not_configured`` if no URL set)."""
    return _probe(settings.embedding_remote_url, "/api/tags", settings.health_http_timeout)


def check_health(settings: CoreSettings) -> dict[str, dict[str, object]]:
    """Run all built-in health checks and return a dict."""
    return {
        "ocr": _check_ocr(settings),
        "database": _check_database(settings),
        "ollama": _check_ollama(settings),
    }


__all__ = ["check_health", "_check_database", "_check_ocr", "_check_ollama"]
