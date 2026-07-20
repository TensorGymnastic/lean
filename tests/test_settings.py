"""Tests for lean.core.config.settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

_VALID_KEY = "x" * 32


@pytest.fixture(autouse=True)
def _default_transport_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    """M5: ports have no Python default — every test must supply them via env."""
    monkeypatch.setenv("MCP_HTTP_PORT", "8765")
    monkeypatch.setenv("API_PORT", "8766")


def test_settings_from_env(monkeypatch) -> None:
    """Settings load correctly from environment variables."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    monkeypatch.setenv("OCR_BASE_URL", "http://3080ti:8000")
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-from-minimax-alias")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)

    from lean.core.config.settings import Settings

    settings = Settings()

    assert settings.db_url == "postgresql://postgres:postgres@localhost:54322/postgres"
    assert settings.hf_token == "test-hf-token"
    assert settings.ocr_base_url == "http://3080ti:8000"
    assert settings.llm_api_key == "sk-test-from-minimax-alias"
    assert settings.api_key == _VALID_KEY
    assert settings.embedding_model == "LiquidAI/LFM2.5-Embedding-350M"
    assert settings.embedding_dim == 1024
    assert settings.chunk_target_max == 450
    assert settings.chunk_hard_cap == 500
    assert settings.mcp_http_port == 8765
    assert settings.api_port == 8766


def test_llm_api_key_alias_is_minimax(monkeypatch) -> None:
    """Regression: ``llm_api_key`` must read ``MINIMAX_API_KEY`` from the env.

    Earlier the field had no alias and pydantic-settings read ``LLM_API_KEY``,
    so users following ``.env.example`` got ``llm_api_key == ""`` and
    ``get_llm()`` silently returned ``None`` (every LLM feature no-op'd).
    """
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-from-env-example")

    from lean.core.config.settings import Settings

    settings = Settings()
    assert settings.llm_api_key == "sk-from-env-example"


def test_settings_defaults_for_optional_fields(monkeypatch) -> None:
    """Required fields come from env; optional fields use defaults."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)

    from lean.core.config.settings import Settings

    settings = Settings()
    assert settings.ocr_base_url is not None


def test_transport_ports_require_env_or_yaml(monkeypatch) -> None:
    """M5: ``mcp_http_port`` and ``api_port`` have no Python default.

    The fixture provides them via env. Without env (or YAML overlay) the
    fields are missing and validation fails — that's the design intent.
    """
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)

    from lean.core.config.settings import Settings

    settings = Settings()
    assert settings.mcp_http_port == 8765
    assert settings.api_port == 8766

    monkeypatch.setenv("MCP_HTTP_PORT", "9999")
    monkeypatch.setenv("API_PORT", "9998")
    settings = Settings()
    assert settings.mcp_http_port == 9999
    assert settings.api_port == 9998


def test_transport_ports_missing_without_env() -> None:
    """When neither env nor YAML supplies the port, instantiation fails fast."""
    import os

    os.environ.pop("MCP_HTTP_PORT", None)
    os.environ.pop("API_PORT", None)
    os.environ.update(
        {
            "SUPABASE_DB_URL": "postgresql://localhost/postgres",
            "HF_TOKEN": "token",
            "LEAN_MCP_API_KEY": _VALID_KEY,
        }
    )

    from lean.core.config.settings import Settings, clear_settings_cache

    clear_settings_cache()
    with pytest.raises(ValidationError, match="MCP_HTTP_PORT"):
        Settings()


def test_get_settings_factory_caches_singleton(monkeypatch) -> None:
    """get_settings() returns the SAME cached instance each call."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)

    from lean.core.config.settings import Settings, clear_settings_cache, get_settings

    clear_settings_cache()

    s1 = get_settings()
    s2 = get_settings()
    assert isinstance(s1, Settings)
    assert isinstance(s2, Settings)
    assert s1 is s2

    clear_settings_cache()


def test_get_settings_picks_up_env_changes_after_cache_clear(monkeypatch) -> None:
    """After cache_clear(), get_settings() reflects new env values."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)

    from lean.core.config.settings import clear_settings_cache, get_settings

    clear_settings_cache()
    s1 = get_settings()
    assert s1.hf_token == "token"

    monkeypatch.setenv("HF_TOKEN", "changed-token")
    assert get_settings().hf_token == "token"

    clear_settings_cache()
    assert get_settings().hf_token == "changed-token"

    clear_settings_cache()


def test_rejects_short_api_key(monkeypatch) -> None:
    """API key shorter than 16 characters is rejected."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "short")

    from lean.core.config.settings import Settings

    with pytest.raises(ValidationError, match="at least 16"):
        Settings()


def test_rejects_default_api_key(monkeypatch) -> None:
    """Default 'change-me' API key is rejected."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "change-me")

    from lean.core.config.settings import Settings

    with pytest.raises(ValidationError, match="change-me"):
        Settings()


@pytest.mark.parametrize(
    "field,value",
    [
        ("search_max_k", 0),
        ("search_max_query_len", 0),
        ("search_fetch_k_cap", 0),
        ("max_pdf_mb", 0),
    ],
)
def test_rejects_non_positive_resource_limits(monkeypatch, field, value) -> None:
    """Resource-limit fields must be >= 1 (DoS prevention)."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "x" * 32)

    from lean.core.config.settings import Settings

    with pytest.raises(ValidationError, match=field):
        Settings(**{field: value})


def test_rejects_hard_cap_le_target_max(monkeypatch) -> None:
    """chunk_hard_cap must be strictly greater than chunk_target_max."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "x" * 32)

    from lean.core.config.settings import Settings

    with pytest.raises(ValidationError, match="chunk_hard_cap"):
        Settings(chunk_target_max=500, chunk_hard_cap=500)


def test_rejects_non_positive_rrf_k(monkeypatch) -> None:
    """rrf_k must be > 0."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "x" * 32)

    from lean.core.config.settings import Settings

    with pytest.raises(ValidationError, match="rrf_k"):
        Settings(rrf_k=0)


def test_rejects_fetch_multiplier_below_one(monkeypatch) -> None:
    """fetch_multiplier must be >= 1."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "x" * 32)

    from lean.core.config.settings import Settings

    with pytest.raises(ValidationError, match="fetch_multiplier"):
        Settings(fetch_multiplier=0)


def test_rejects_non_positive_eval_k(monkeypatch) -> None:
    """eval_k must be > 0."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "x" * 32)

    from lean.core.config.settings import Settings

    with pytest.raises(ValidationError, match="eval_k"):
        Settings(eval_k=0)


def test_vlm_enabled_requires_base_url_and_model(monkeypatch) -> None:
    """Enabling VLM without base_url or model is rejected at startup.

    The single source of truth is now ``domain_config.vlm.enabled`` (set
    via YAML's top-level ``vlm:`` block), not a top-level ``vlm_enabled``
    field on Settings.
    """
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "x" * 32)

    from lean.core.config.settings import Settings

    with pytest.raises(ValidationError, match="vlm.enabled"):
        Settings(domain_config={"vlm": {"enabled": True}})


def test_rejects_invalid_vlm_detail(monkeypatch) -> None:
    """vlm_detail must be 'low', 'default', or 'high'."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "x" * 32)

    from lean.core.config.settings import Settings

    with pytest.raises(ValidationError, match="vlm_detail"):
        Settings(vlm_detail="medium")


def test_rejects_vlm_max_concurrency_below_one(monkeypatch) -> None:
    """vlm_max_concurrency must be >= 1."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "x" * 32)

    from lean.core.config.settings import Settings

    with pytest.raises(ValidationError, match="vlm_max_concurrency"):
        Settings(vlm_max_concurrency=0)


@pytest.mark.parametrize("port_field", ["mcp_http_port", "api_port"])
@pytest.mark.parametrize(
    "bad_port",
    [0, 70000],
)
def test_rejects_port_out_of_range(monkeypatch, port_field, bad_port) -> None:
    """Ports must be in 1..65535."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "x" * 32)

    from lean.core.config.settings import Settings, clear_settings_cache

    env_var_name = port_field.upper()
    if env_var_name not in ("MCP_HTTP_PORT", "API_PORT"):
        env_var_name = {"mcp_http_port": "MCP_HTTP_PORT", "api_port": "API_PORT"}.get(
            port_field, env_var_name
        )
    monkeypatch.setenv(env_var_name, str(bad_port))
    clear_settings_cache()
    with pytest.raises(ValidationError, match="port"):
        Settings()
