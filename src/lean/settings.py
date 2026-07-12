"""Application settings loaded from environment."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration is env-driven via pydantic-settings.

    Copy ``.env.example`` to ``.env`` and fill in the values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Supabase ---
    supabase_url: str = Field(
        description="Supabase project URL (or local CLI URL http://localhost:54321)"
    )
    supabase_service_key: str = Field(description="Supabase service_role key")
    supabase_db_url: str = Field(
        description="Direct Postgres DSN for pgvector ops (port 54322 for local CLI)"
    )

    # --- HuggingFace ---
    hf_token: str = Field(description="HF token for gated models and trust_remote_code")

    # --- vLLM (remote 3080 Ti) ---
    vllm_base_url: str = Field(
        default="http://localhost:8000",
        description="vLLM OpenAI-compatible base URL (e.g. http://3080ti-host:8000)",
    )

    # --- MCP auth ---
    lean_mcp_api_key: str = Field(description="Bearer token for MCP HTTP transport")

    # --- Embeddings ---
    embedding_model: str = "LiquidAI/LFM2.5-Embedding-350M"
    embedding_dim: int = 1024

    # --- Chunking (tokens) ---
    chunk_target_min: int = 350
    chunk_target_max: int = 450
    chunk_hard_cap: int = 500

    # --- Storage buckets ---
    sources_bucket: str = "sources"
    markdown_bucket: str = "markdown"

    # --- Transport ---
    mcp_http_host: str = "127.0.0.1"
    mcp_http_port: int = 8765
    api_port: int = 8766


def get_settings() -> Settings:
    """Factory used as a FastAPI/dependency-injection helper.

    Each call re-reads the environment — do not cache at module level
    so tests can monkeypatch env vars between calls.
    """
    return Settings()
