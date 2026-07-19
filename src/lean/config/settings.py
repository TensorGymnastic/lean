"""Application settings: .env for secrets, config.yaml for app config."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


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
    supabase_db_url: str = Field(description="Direct Postgres DSN for pgvector")
    hf_token: str = Field(default="", description="HF token for gated models")
    lean_mcp_api_key: str = Field(description="Bearer token for MCP HTTP transport")

    # --- App config (from config.yaml, overridable by env) ---
    ocr_base_url: str = _yaml.get("ocr", {}).get("base_url", "")

    embedding_model: str = _yaml.get("embedding", {}).get("model", "LiquidAI/LFM2.5-Embedding-350M")
    embedding_model_revision: str = _yaml.get("embedding", {}).get(
        "model_revision", "f35ae2c91d687658dbf1f2b449382f0b019b9808"
    )
    embedding_dim: int = _yaml.get("embedding", {}).get("dim", 1024)
    embedding_device: str = _yaml.get("embedding", {}).get("device", "cpu")
    embedding_remote_url: str = _yaml.get("embedding", {}).get("remote_url", "")
    embedding_remote_model: str = _yaml.get("embedding", {}).get("remote_model", "")
    embedding_num_ctx: int = _yaml.get("embedding", {}).get("num_ctx", 32768)
    embedding_http_timeout: float = _yaml.get("embedding", {}).get("http_timeout", 30.0)
    embedding_max_retries: int = _yaml.get("embedding", {}).get("max_retries", 3)

    ocr_model: str = _yaml.get("ocr", {}).get("model", "baidu/Unlimited-OCR")
    ocr_dpi: int = _yaml.get("ocr", {}).get("dpi", 300)
    ocr_timeout_s: float = _yaml.get("ocr", {}).get("timeout_s", 1800.0)
    ocr_max_tokens: int = _yaml.get("ocr", {}).get("max_tokens", 32768)
    ocr_batch_size: int = _yaml.get("ocr", {}).get("batch_size", 20)

    marker_force_ocr: bool = _yaml.get("marker", {}).get("force_ocr", False)
    marker_remote_url: str = _yaml.get("marker", {}).get("remote_url", "")

    chunk_target_max: int = _yaml.get("chunking", {}).get("target_max", 450)
    chunk_hard_cap: int = _yaml.get("chunking", {}).get("hard_cap", 500)
    chunk_overlap: int = _yaml.get("chunking", {}).get("overlap", 50)
    max_section_heading_level: int = _yaml.get("chunking", {}).get("max_heading_level", 4)
    token_counter_encoding: str = _yaml.get("chunking", {}).get("token_encoding", "cl100k_base")

    search_top_k: int = _yaml.get("retrieval", {}).get("top_k", 5)
    search_max_k: int = _yaml.get("retrieval", {}).get("max_k", 100)
    search_max_query_len: int = _yaml.get("retrieval", {}).get("max_query_len", 2000)
    search_fetch_k_cap: int = _yaml.get("retrieval", {}).get("fetch_k_cap", 500)
    min_similarity: float = _yaml.get("retrieval", {}).get("min_similarity", 0.0)
    hybrid_search_enabled: bool = _yaml.get("retrieval", {}).get("hybrid_search", True)
    fetch_multiplier: int = _yaml.get("retrieval", {}).get("fetch_multiplier", 8)
    fetch_k_floor: int = _yaml.get("retrieval", {}).get("fetch_k_floor", 40)
    rrf_k: int = _yaml.get("retrieval", {}).get("rrf_k", 60)
    rerank_enabled: bool = _yaml.get("retrieval", {}).get("rerank", {}).get("enabled", False)
    rerank_model: str = (
        _yaml.get("retrieval", {})
        .get("rerank", {})
        .get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    rerank_model_revision: str = (
        _yaml.get("retrieval", {})
        .get("rerank", {})
        .get("model_revision", "c5ee24cb16019beea0893ab7796b1df96625c6b8")
    )
    rerank_top_n: int = _yaml.get("retrieval", {}).get("rerank", {}).get("top_n", 5)

    mcp_http_host: str = _yaml.get("transport", {}).get("mcp_host", "127.0.0.1")
    mcp_http_port: int = _yaml.get("transport", {}).get("mcp_port")
    api_port: int = _yaml.get("transport", {}).get("api_port")

    # --- Storage paths ---
    corpus_root: str = _yaml.get("storage", {}).get("corpus_root", "data")
    max_pdf_mb: int = _yaml.get("storage", {}).get("max_pdf_mb", 200)

    # --- Health checks ---
    health_http_timeout: float = _yaml.get("health", {}).get("http_timeout", 10.0)

    # --- Logging ---
    log_level: str = _yaml.get("logging", {}).get("level", "INFO")

    # --- Ingestion tuning ---
    block_match_min_overlap: float = _yaml.get("ingestion", {}).get("block_match_min_overlap", 0.15)

    # --- Eval ---
    eval_sample_size: int = _yaml.get("eval", {}).get("sample_size", 50)
    eval_k: int = _yaml.get("eval", {}).get("k", 5)
    eval_seed: int = _yaml.get("eval", {}).get("seed", 42)

    # --- LLM (optional sidecar for Contextual Retrieval + query transforms) ---
    minimax_api_key: str = Field(default="", description="MiniMax API key")
    llm_minimax_base_url: str = _yaml.get("llm", {}).get(
        "minimax_base_url", "https://api.minimax.io"
    )
    llm_minimax_model: str = _yaml.get("llm", {}).get("minimax_model", "MiniMax-Text-01")
    llm_ollama_url: str = _yaml.get("llm", {}).get("ollama_url", "")
    llm_ollama_model: str = _yaml.get("llm", {}).get("ollama_model", "qwen3.5:9b")
    llm_timeout_s: float = _yaml.get("llm", {}).get("timeout_s", 60.0)
    llm_generate_max_tokens: int = _yaml.get("llm", {}).get("generate_max_tokens", 500)
    llm_generate_temperature: float = _yaml.get("llm", {}).get("generate_temperature", 0.0)
    llm_contextual_retrieval: bool = _yaml.get("llm", {}).get("contextual_retrieval", False)
    llm_multi_query: bool = _yaml.get("llm", {}).get("multi_query", False)
    llm_hyde: bool = _yaml.get("llm", {}).get("hyde", False)
    llm_multi_query_count: int = _yaml.get("llm", {}).get("multi_query_count", 4)

    # --- VLM (optional vision-language model for chart/image description) ---
    vlm_enabled: bool = (_yaml.get("vlm") or {}).get("enabled", False)
    vlm_base_url: str = (_yaml.get("vlm") or {}).get("base_url", "")
    vlm_model: str = (_yaml.get("vlm") or {}).get("model", "")
    vlm_api_key: str = Field(default="", description="VLM API key")
    vlm_detail: str = (_yaml.get("vlm") or {}).get("detail", "default")
    vlm_timeout_s: float = (_yaml.get("vlm") or {}).get("timeout_s", 120.0)
    vlm_max_concurrency: int = (_yaml.get("vlm") or {}).get("max_concurrency", 4)
    vlm_max_tokens: int = (_yaml.get("vlm") or {}).get("max_tokens", 1000)
    vlm_disable_thinking: bool = (_yaml.get("vlm") or {}).get("disable_thinking", True)

    @field_validator("lean_mcp_api_key")
    @classmethod
    def _validate_api_key(cls, v: str) -> str:
        if len(v) < 16:
            raise ValueError("LEAN_MCP_API_KEY must be at least 16 characters")
        if v == "change-me":
            raise ValueError("LEAN_MCP_API_KEY must not be the default 'change-me'")
        return v

    @field_validator("mcp_http_port", "api_port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return v

    @model_validator(mode="after")
    def _validate_cross_field(self) -> Settings:
        if self.chunk_hard_cap <= self.chunk_target_max:
            raise ValueError(
                f"chunk_hard_cap ({self.chunk_hard_cap}) must be > "
                f"chunk_target_max ({self.chunk_target_max})"
            )
        if self.rrf_k <= 0:
            raise ValueError(f"rrf_k must be > 0, got {self.rrf_k}")
        if self.fetch_multiplier < 1:
            raise ValueError(f"fetch_multiplier must be >= 1, got {self.fetch_multiplier}")
        if self.eval_k <= 0:
            raise ValueError(f"eval_k must be > 0, got {self.eval_k}")
        if self.search_max_k < 1:
            raise ValueError(f"search_max_k must be >= 1, got {self.search_max_k}")
        if self.search_max_query_len < 1:
            raise ValueError(f"search_max_query_len must be >= 1, got {self.search_max_query_len}")
        if self.search_fetch_k_cap < 1:
            raise ValueError(f"search_fetch_k_cap must be >= 1, got {self.search_fetch_k_cap}")
        if self.max_pdf_mb < 1:
            raise ValueError(f"max_pdf_mb must be >= 1, got {self.max_pdf_mb}")
        if self.vlm_enabled and (not self.vlm_base_url or not self.vlm_model):
            raise ValueError("vlm_enabled=true requires vlm_base_url and vlm_model to be set")
        if self.vlm_enabled:
            logger.warning(
                "VLM enabled — chart images will be sent to %s. "
                "Disable vlm.enabled for corpora with PII/trade-secret concerns.",
                self.vlm_base_url,
            )
        if self.vlm_detail not in ("low", "default", "high"):
            raise ValueError(
                f"vlm_detail must be 'low', 'default', or 'high', got '{self.vlm_detail}'"
            )
        if self.vlm_max_concurrency < 1:
            raise ValueError(f"vlm_max_concurrency must be >= 1, got {self.vlm_max_concurrency}")
        if not 0.0 < self.block_match_min_overlap <= 1.0:
            raise ValueError(
                f"block_match_min_overlap must be in (0, 1], got {self.block_match_min_overlap}"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — same Settings instance on every call.

    Use this everywhere instead of ``Settings()``. Critical for hot paths
    (api/routes.py:_verify_token runs per request, services/search.py per
    query). Call ``get_settings.cache_clear()`` to force re-read of env.
    """
    return Settings()
