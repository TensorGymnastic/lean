"""Application settings: .env for secrets, config.yaml for app config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml() -> dict[str, Any]:
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.is_file():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml()


class Settings(BaseSettings):
    """Unified settings: env vars (secrets) override YAML defaults (app config)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets (from .env only) ---
    supabase_url: str = Field(description="Supabase project URL")
    supabase_service_key: str = Field(description="Supabase service_role key")
    supabase_db_url: str = Field(description="Direct Postgres DSN for pgvector")
    hf_token: str = Field(default="", description="HF token for gated models")
    lean_mcp_api_key: str = Field(description="Bearer token for MCP HTTP transport")

    # --- App config (from config.yaml, overridable by env) ---
    vllm_base_url: str = _yaml.get("vllm", {}).get("base_url", "http://localhost:8000")

    embedding_model: str = _yaml.get("embedding", {}).get("model", "LiquidAI/LFM2.5-Embedding-350M")
    embedding_dim: int = _yaml.get("embedding", {}).get("dim", 1024)
    embedding_device: str = _yaml.get("embedding", {}).get("device", "cpu")
    embedding_remote_url: str = _yaml.get("embedding", {}).get("remote_url", "")
    embedding_remote_model: str = _yaml.get("embedding", {}).get("remote_model", "")

    ocr_model: str = _yaml.get("ocr", {}).get("model", "baidu/Unlimited-OCR")
    ocr_dpi: int = _yaml.get("ocr", {}).get("dpi", 300)
    ocr_timeout_s: float = _yaml.get("ocr", {}).get("timeout_s", 600.0)
    ocr_max_tokens: int = _yaml.get("ocr", {}).get("max_tokens", 32768)

    chunk_target_min: int = _yaml.get("chunking", {}).get("target_min", 350)
    chunk_target_max: int = _yaml.get("chunking", {}).get("target_max", 450)
    chunk_hard_cap: int = _yaml.get("chunking", {}).get("hard_cap", 500)
    max_section_heading_level: int = _yaml.get("chunking", {}).get("max_heading_level", 4)
    token_counter_encoding: str = _yaml.get("chunking", {}).get("token_encoding", "cl100k_base")

    search_top_k: int = _yaml.get("retrieval", {}).get("top_k", 5)
    min_similarity: float = _yaml.get("retrieval", {}).get("min_similarity", 0.0)
    hybrid_search_enabled: bool = _yaml.get("retrieval", {}).get("hybrid_search", True)
    rerank_enabled: bool = _yaml.get("retrieval", {}).get("rerank", {}).get("enabled", False)
    rerank_model: str = (
        _yaml.get("retrieval", {})
        .get("rerank", {})
        .get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    rerank_top_n: int = _yaml.get("retrieval", {}).get("rerank", {}).get("top_n", 5)

    sources_bucket: str = _yaml.get("storage", {}).get("sources_bucket", "sources")
    markdown_bucket: str = _yaml.get("storage", {}).get("markdown_bucket", "markdown")

    mcp_http_host: str = _yaml.get("transport", {}).get("mcp_host", "127.0.0.1")
    mcp_http_port: int = _yaml.get("transport", {}).get("mcp_port", 8765)
    api_port: int = _yaml.get("transport", {}).get("api_port", 8766)


def get_settings() -> Settings:
    """Factory that re-reads env on each call (test-friendly)."""
    return Settings()
