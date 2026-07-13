"""Tests for lean.config.settings."""

from __future__ import annotations


def test_settings_from_env(monkeypatch) -> None:
    """Settings load correctly from environment variables."""
    monkeypatch.setenv("SUPABASE_URL", "http://localhost:54321")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-service-key")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    monkeypatch.setenv("VLLM_BASE_URL", "http://3080ti:8000")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "test-api-key")

    from lean.config.settings import Settings

    settings = Settings()

    assert settings.supabase_url == "http://localhost:54321"
    assert settings.supabase_service_key == "test-service-key"
    assert settings.supabase_db_url == "postgresql://postgres:postgres@localhost:54322/postgres"
    assert settings.hf_token == "test-hf-token"
    assert settings.vllm_base_url == "http://3080ti:8000"
    assert settings.lean_mcp_api_key == "test-api-key"
    assert settings.embedding_model == "LiquidAI/LFM2.5-Embedding-350M"
    assert settings.embedding_dim == 1024
    assert settings.chunk_target_min == 350
    assert settings.chunk_target_max == 450
    assert settings.chunk_hard_cap == 500
    assert settings.sources_bucket == "sources"
    assert settings.markdown_bucket == "markdown"
    assert settings.mcp_http_port == 8765
    assert settings.api_port == 8766


def test_settings_defaults_for_optional_fields(monkeypatch) -> None:
    """Required fields come from env; optional fields use defaults."""
    monkeypatch.setenv("SUPABASE_URL", "http://localhost:54321")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "key")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "apikey")
    # VLLM_BASE_URL not set — should use default

    from lean.config.settings import Settings

    settings = Settings()
    assert settings.vllm_base_url == "http://localhost:8000"


def test_get_settings_factory_returns_fresh_instance(monkeypatch) -> None:
    """get_settings() returns a new Settings each call."""
    monkeypatch.setenv("SUPABASE_URL", "http://localhost:54321")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "key")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", "apikey")

    from lean.config.settings import Settings, get_settings

    s1 = get_settings()
    s2 = get_settings()
    assert isinstance(s1, Settings)
    assert isinstance(s2, Settings)
    assert s1 is not s2
