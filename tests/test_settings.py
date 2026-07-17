"""Tests for lean.config.settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

_VALID_KEY = "x" * 32


def test_settings_from_env(monkeypatch) -> None:
    """Settings load correctly from environment variables."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    monkeypatch.setenv("OCR_BASE_URL", "http://3080ti:8000")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)

    from lean.config.settings import Settings

    settings = Settings()

    assert settings.supabase_db_url == "postgresql://postgres:postgres@localhost:54322/postgres"
    assert settings.hf_token == "test-hf-token"
    assert settings.ocr_base_url == "http://3080ti:8000"
    assert settings.lean_mcp_api_key == _VALID_KEY
    assert settings.embedding_model == "LiquidAI/LFM2.5-Embedding-350M"
    assert settings.embedding_dim == 1024
    assert settings.chunk_target_max == 450
    assert settings.chunk_hard_cap == 500
    assert settings.mcp_http_port == 8765
    assert settings.api_port == 8766


def test_settings_defaults_for_optional_fields(monkeypatch) -> None:
    """Required fields come from env; optional fields use defaults."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)

    from lean.config.settings import Settings

    settings = Settings()
    assert settings.ocr_base_url is not None


def test_get_settings_factory_caches_singleton(monkeypatch) -> None:
    """get_settings() returns the SAME cached instance each call."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)

    from lean.config.settings import Settings, get_settings

    get_settings.cache_clear()

    s1 = get_settings()
    s2 = get_settings()
    assert isinstance(s1, Settings)
    assert isinstance(s2, Settings)
    assert s1 is s2

    get_settings.cache_clear()


def test_get_settings_picks_up_env_changes_after_cache_clear(monkeypatch) -> None:
    """After cache_clear(), get_settings() reflects new env values."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)

    from lean.config.settings import get_settings

    get_settings.cache_clear()
    s1 = get_settings()
    assert s1.hf_token == "token"

    monkeypatch.setenv("HF_TOKEN", "changed-token")
    assert get_settings().hf_token == "token"

    get_settings.cache_clear()
    assert get_settings().hf_token == "changed-token"

    get_settings.cache_clear()


def test_rejects_short_api_key(monkeypatch) -> None:
    """API key shorter than 16 characters is rejected."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "short")

    from lean.config.settings import Settings

    with pytest.raises(ValidationError, match="at least 16"):
        Settings()


def test_rejects_default_api_key(monkeypatch) -> None:
    """Default 'change-me' API key is rejected."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "change-me")

    from lean.config.settings import Settings

    with pytest.raises(ValidationError, match="change-me"):
        Settings()
