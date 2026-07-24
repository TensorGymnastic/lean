"""Coverage tests for the universal transport factories (FastMCP, FastAPI, health).

Subprocess smoke tests in ``test_api_routes.py`` exercise the runtime
path but don't count toward coverage (they run in a separate process).
These tests directly invoke the factory functions for unit coverage.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

# ---------- build_api ----------


def test_build_api_returns_fastapi_app() -> None:
    """``build_api`` returns a FastAPI instance."""
    from lean.core.transports.api import build_api

    s = MagicMock()
    s.api_key = "x" * 32
    app = build_api(name="t", version="0.0.0", settings=s)
    assert isinstance(app, FastAPI)


def test_build_api_exposes_health_endpoint() -> None:
    """``build_api`` always registers ``GET /health``."""
    from lean.core.transports.api import build_api

    s = MagicMock()
    s.api_key = "x" * 32
    app = build_api(name="t", version="0.0.0", settings=s)
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/health" in paths


def test_build_api_calls_register_fn() -> None:
    """``build_api`` invokes ``register_fn(app, services, settings)`` when provided."""
    from lean.core.transports.api import build_api

    called = []

    def reg(app, services, settings):
        called.append(settings.api_key)

    s = MagicMock()
    s.api_key = "y" * 32
    build_api(name="t", version="0.0.0", settings=s, register_fn=reg)
    assert called == ["y" * 32]


def test_build_api_maps_value_error_to_400() -> None:
    """``ValueError`` raised inside a route is mapped to HTTP 400 by the exception handler."""
    from fastapi.testclient import TestClient

    from lean.core.transports.api import build_api

    s = MagicMock()
    s.api_key = "x" * 32
    app = build_api(name="t", version="0.0.0", settings=s)

    @app.get("/boom")
    def boom() -> None:
        raise ValueError("kaboom")

    client = TestClient(app)
    r = client.get("/boom")
    assert r.status_code == 400
    assert r.json() == {"detail": "kaboom"}


def test_build_api_maps_permission_error_to_403() -> None:
    """``PermissionError`` raised inside a route is mapped to HTTP 403."""
    from fastapi.testclient import TestClient

    from lean.core.transports.api import build_api

    s = MagicMock()
    s.api_key = "x" * 32
    app = build_api(name="t", version="0.0.0", settings=s)

    @app.get("/forbidden")
    def forbidden() -> None:
        raise PermissionError("nope")

    client = TestClient(app)
    r = client.get("/forbidden")
    assert r.status_code == 403
    assert r.json() == {"detail": "nope"}


def test_build_api_maps_not_found_to_404() -> None:
    """``FileNotFoundError`` raised inside a route is mapped to HTTP 404."""
    from fastapi.testclient import TestClient

    from lean.core.transports.api import build_api

    s = MagicMock()
    s.api_key = "x" * 32
    app = build_api(name="t", version="0.0.0", settings=s)

    @app.get("/missing")
    def missing() -> None:
        raise FileNotFoundError("nope.pdf")

    client = TestClient(app)
    r = client.get("/missing")
    assert r.status_code == 404
    assert r.json() == {"detail": "nope.pdf"}


# ---------- Health ----------


def test_check_health_returns_dict() -> None:
    """``check_health`` returns a dict with the three component probes."""
    from lean.core.transports.health import check_health

    s = MagicMock()
    s.health_http_timeout = 5.0
    s.ocr_base_url = ""
    s.embedding_remote_url = ""
    s.db_url = "postgresql://invalid:invalid@127.0.0.1:1/postgres"

    result = check_health(s)
    assert "ocr" in result
    assert "database" in result
    assert "ollama" in result


def test_check_ocr_with_empty_url_returns_not_configured() -> None:
    """``_check_ocr`` returns ``not_configured`` when ``ocr_base_url`` is empty."""
    from lean.core.transports.health import _check_ocr

    s = MagicMock()
    s.ocr_base_url = ""
    result = _check_ocr(s)
    assert result["status"] == "not_configured"


def test_check_ollama_with_empty_url_returns_not_configured() -> None:
    """``_check_ollama`` returns ``not_configured`` when ``embedding_remote_url`` is empty."""
    from lean.core.transports.health import _check_ollama

    s = MagicMock()
    s.embedding_remote_url = ""
    result = _check_ollama(s)
    assert result["status"] == "not_configured"


def test_check_ocr_reports_error_on_connection_failure() -> None:
    """``_check_ocr`` reports ``status=error`` when httpx.get raises."""
    from lean.core.transports.health import _check_ocr

    s = MagicMock()
    s.ocr_base_url = "http://gpu:8000"
    s.health_http_timeout = 5.0
    with patch("lean.core.transports.health.httpx.get", side_effect=Exception("boom")):
        result = _check_ocr(s)
    assert result["status"] == "error"
    assert "boom" in result["error"]


def test_check_ocr_reports_ok_on_200() -> None:
    """``_check_ocr`` reports ``status=ok`` when the OCR backend returns HTTP 200."""
    from lean.core.transports.health import _check_ocr

    s = MagicMock()
    s.ocr_base_url = "http://gpu:8000"
    s.health_http_timeout = 5.0
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("lean.core.transports.health.httpx.get", return_value=mock_resp):
        result = _check_ocr(s)
    assert result["status"] == "ok"
    assert result["url"] == "http://gpu:8000"


def test_check_ocr_reports_error_on_non_200() -> None:
    """``_check_ocr`` reports ``status=error`` when the OCR backend returns non-200."""
    from lean.core.transports.health import _check_ocr

    s = MagicMock()
    s.ocr_base_url = "http://gpu:8000"
    s.health_http_timeout = 5.0
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    with patch("lean.core.transports.health.httpx.get", return_value=mock_resp):
        result = _check_ocr(s)
    assert result["status"] == "error"


def test_check_database_reports_error_on_connection_failure() -> None:
    """``_check_database`` returns ``status=error`` when ``StoreConnection.from_env`` raises."""
    from lean.core.transports.health import _check_database

    s = MagicMock()
    s.db_url = "postgresql://localhost/postgres"
    with patch(
        "lean.core.store.base.StoreConnection.from_env",
        side_effect=Exception("connection refused"),
    ):
        result = _check_database(s)
    assert result["status"] == "error"
    assert "connection refused" in result["error"]


# ---------- YAML loader integration ----------


def test_build_from_yaml_happy_path(tmp_path: Path) -> None:
    """``build_from_yaml`` loads a minimal manifest and returns a TransportBuilder."""
    from lean.core.transports.builder import TransportBuilder
    from lean.core.transports.yaml_loader import build_from_yaml

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "domain: {name: smoke, version: 0.1.0}\n"
        "extractors:\n"
        "  - adapter: lean.domains.pdf_lss.adapters.MarkitdownAdapter\n"
        "    config: {}\n"
        "tools: {module: lean.domains.pdf_lss.tools}\n"
    )
    with (
        patch("lean.core.transports.yaml_loader.set_pipeline"),
        patch("lean.core.transports.yaml_loader.get_settings"),
    ):
        builder = build_from_yaml(cfg)
    assert isinstance(builder, TransportBuilder)


def test_yaml_loader_imports_no_such_module_raises_value_error(tmp_path: Path) -> None:
    """A YAML that references a nonexistent extractor module raises ValueError."""
    from lean.core.transports.yaml_loader import build_from_yaml

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "domain: {name: x, version: 0.1.0}\n"
        "extractors:\n"
        "  - adapter: this.module.does.not.exist.MyClass\n"
        "    config: {}\n"
        "tools: {module: lean.domains.pdf_lss.tools}\n"
    )
    with pytest.raises(ValueError):
        build_from_yaml(cfg)


def test_yaml_loader_top_level_unknown_keys_become_settings(tmp_path: Path) -> None:
    """Unknown top-level keys in the YAML land under ``settings`` (for forwarding)."""
    from lean.core.transports.yaml_loader import build_from_yaml

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "domain: {name: x, version: 0.1.0}\n"
        "extractors:\n"
        "  - adapter: lean.domains.pdf_lss.adapters.MarkitdownAdapter\n"
        "    config: {}\n"
        "tools: {module: lean.domains.pdf_lss.tools}\n"
        "foo:\n"
        "  bar: baz\n"
    )
    with (
        patch("lean.core.transports.yaml_loader.set_pipeline"),
        patch("lean.core.transports.yaml_loader.get_settings"),
    ):
        # Should not raise — unknown key is silently absorbed
        build_from_yaml(cfg)


# ---------- WP-5: register_* without module-level shims ----------


def test_yaml_domain_registers_mcp_without_module_shim() -> None:
    """_YamlDomain.register_mcp wires tools directly, no register_mcp shim needed."""
    import sys
    import types

    from lean.core.adapters import mcp_tool
    from lean.core.transports.yaml_loader import _YamlDomain

    fake_mod = types.ModuleType("fake_tools")
    sys.modules["fake_tools"] = fake_mod

    @mcp_tool
    def my_tool() -> int:
        return 42

    fake_mod.my_tool = my_tool

    cfg = MagicMock()
    cfg.domain.name = "x"
    cfg.domain.version = "0.1.0"
    cfg.domain.description = ""
    domain = _YamlDomain(cfg, fake_mod)

    mcp = MagicMock()
    domain.register_mcp(mcp, {}, MagicMock())
    mcp.tool.assert_called_once()

    del sys.modules["fake_tools"]


def test_build_from_yaml_uses_init_not_new(tmp_path: Path) -> None:
    """build_from_yaml must use TransportBuilder(__init__), not __new__ bypass."""
    from lean.core.transports import builder as builder_mod
    from lean.core.transports.yaml_loader import _YamlDomain, build_from_yaml

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "domain: {name: x, version: 0.1.0}\n"
        "extractors:\n"
        "  - adapter: lean.domains.pdf_lss.adapters.MarkitdownAdapter\n"
        "    config: {}\n"
        "tools: {module: lean.domains.pdf_lss.tools}\n"
    )

    init_calls: list[object] = []

    def recording_init(self: object, domain: object) -> None:
        init_calls.append(domain)

    with (
        patch.object(builder_mod.TransportBuilder, "__init__", recording_init),
        patch("lean.core.transports.yaml_loader.set_pipeline"),
        patch("lean.core.transports.yaml_loader.get_settings"),
    ):
        build_from_yaml(cfg)

    assert len(init_calls) == 1
    assert isinstance(init_calls[0], _YamlDomain)
