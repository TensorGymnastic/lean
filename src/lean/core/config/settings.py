"""Universal settings: .env for secrets, config.yaml for app config.

``CoreSettings`` is the universal base. Domains extend it with their
own fields and pass their own ``config.yaml`` path to
``MySettings.from_yaml(domain_yaml_path)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def _load_yaml(config_path: Path | None) -> dict[str, Any]:
    """Load YAML from a domain-supplied path. Returns ``{}`` if missing."""
    if config_path is None or not config_path.is_file():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _apply_yaml_overlay(instance: CoreSettings, yaml_data: dict[str, Any]) -> None:
    """Walk YAML and set known fields on the instance.

    Flat YAML keys map directly to Settings field names. Nested keys
    (``embedding.model``) are tried as ``f"{section}_{key}"`` first,
    then fall back to storing under ``domain_config[section][key]``
    so the domain can still read them.

    Examples (with CoreSettings):
        ``embedding: {model: foo}``  →  ``instance.embedding_model = "foo"``
        ``foo: {bar: 1}``            →  ``instance.domain_config["foo"]["bar"] = 1``
    """
    fields = instance.__class__.model_fields
    for section, sub in yaml_data.items():
        if not isinstance(sub, dict):
            if section in fields:
                setattr(instance, section, sub)
            else:
                instance.domain_config[section] = sub
            continue
        for key, value in sub.items():
            if key in fields:
                setattr(instance, key, value)
                continue
            composite = f"{section}_{key}"
            if composite in fields:
                setattr(instance, composite, value)
                continue
            bucket = instance.domain_config.setdefault(section, {})
            if isinstance(bucket, dict):
                bucket[key] = value


class CoreSettings(BaseSettings):
    """Universal settings: env vars (secrets) override YAML defaults (app config).

    Subclass to add domain-specific fields. Domains construct via
    ``MySettings.from_yaml(my_yaml_path)``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets (from .env only) ---
    db_url: str = Field(
        description="Direct Postgres DSN for pgvector",
        validation_alias="SUPABASE_DB_URL",
    )
    hf_token: str = Field(default="", description="HF token for gated models")
    api_key: str = Field(
        description="Bearer token for HTTP transports",
        validation_alias="LEAN_MCP_API_KEY",
    )

    # --- Embedder (universal) ---
    embedding_model: str = "LiquidAI/LFM2.5-Embedding-350M"
    embedding_model_revision: str = "f35ae2c91d687658dbf1f2b449382f0b019b9808"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"
    embedding_remote_url: str = ""
    embedding_remote_model: str = ""
    embedding_num_ctx: int = 32768
    embedding_http_timeout: float = 30.0
    embedding_max_retries: int = 3

    # --- Chunker (universal) ---
    chunk_target_max: int = 450
    chunk_hard_cap: int = 500
    chunk_overlap: int = 50
    max_section_heading_level: int = 4
    token_counter_encoding: str = "cl100k_base"

    # --- Retrieval (universal) ---
    search_top_k: int = 5
    search_max_k: int = 100
    search_max_query_len: int = 2000
    search_fetch_k_cap: int = 500
    min_similarity: float = 0.0
    hybrid_search_enabled: bool = True
    fetch_multiplier: int = 8
    fetch_k_floor: int = 40
    rrf_k: int = 60
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_model_revision: str = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    rerank_top_n: int = 5

    # --- Transports (universal) ---
    mcp_http_host: str = "127.0.0.1"
    mcp_http_port: int = 8765
    api_port: int = 8766

    # --- Storage (universal) ---
    corpus_root: str = "data"
    max_pdf_mb: int = 200

    # --- Health checks ---
    health_http_timeout: float = 10.0

    # --- Logging ---
    log_level: str = "INFO"

    # --- Eval ---
    eval_sample_size: int = 50
    eval_k: int = 5
    eval_seed: int = 42

    # --- LLM sidecar (universal; optional) ---
    llm_api_key: str = Field(default="", description="LLM provider API key")
    llm_base_url: str = "https://api.minimax.io"
    llm_model: str = "MiniMax-Text-01"
    llm_ollama_url: str = ""
    llm_ollama_model: str = "qwen3.5:9b"
    llm_timeout_s: float = 60.0
    llm_generate_max_tokens: int = 500
    llm_generate_temperature: float = 0.0
    llm_contextual_retrieval: bool = False
    llm_multi_query: bool = False
    llm_hyde: bool = False
    llm_multi_query_count: int = 4

    # --- VLM sidecar (universal; optional) ---
    # No domain prompt here — lean-core ships the transport; domains register prompts.
    vlm_enabled: bool = False
    vlm_base_url: str = ""
    vlm_model: str = ""
    vlm_api_key: str = Field(default="", description="VLM API key")
    vlm_detail: str = "default"
    vlm_timeout_s: float = 120.0
    vlm_max_concurrency: int = 4
    vlm_max_tokens: int = 1000
    vlm_disable_thinking: bool = True

    # --- Ingestion tuning (universal; domains may override defaults) ---
    block_match_min_overlap: float = 0.15

    # --- Bucket for YAML keys that don't map to a Settings field ---
    domain_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, v: str) -> str:
        if len(v) < 16:
            raise ValueError("API_KEY must be at least 16 characters")
        if v == "change-me":
            raise ValueError("API_KEY must not be the default 'change-me'")
        return v

    @field_validator("mcp_http_port", "api_port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return v

    @model_validator(mode="after")
    def _validate_cross_field(self) -> CoreSettings:
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
                "VLM enabled — images will be sent to %s. "
                "Disable for corpora with PII/trade-secret concerns.",
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

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> CoreSettings:
        """Build a Settings instance, overlaying YAML values onto defaults.

        Lean-core itself ships no ``config.yaml`` — domains pass their
        own. Calling ``CoreSettings.from_yaml(None)`` uses pure defaults.
        """
        yaml_data = _load_yaml(config_path)
        instance = cls()
        _apply_yaml_overlay(instance, yaml_data)
        return instance


# Backward-compat alias — code paths that imported ``Settings`` before
# the lean-core split. Domains should subclass ``CoreSettings``.
Settings = CoreSettings


# Singleton — per-subclass so subclasses don't share a cached instance.
_settings_cache: dict[type, Any] = {}


def get_settings(cls: type[CoreSettings] = CoreSettings) -> CoreSettings:
    """Cached singleton for the given Settings subclass.

    Usage:
        get_settings()                       # → CoreSettings
        get_settings(LeanLssSettings)        # → LeanLssSettings

    Tests should call ``clear_settings_cache()`` after mutating env.
    """
    if cls not in _settings_cache:
        _settings_cache[cls] = cls.from_yaml()
    return _settings_cache[cls]  # type: ignore[no-any-return]


def clear_settings_cache() -> None:
    """Clear all cached settings — call in tests after env mutation."""
    _settings_cache.clear()


__all__ = [
    "CoreSettings",
    "Settings",
    "get_settings",
    "clear_settings_cache",
]
