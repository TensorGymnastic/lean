"""Universal settings: .env for secrets, YAML overlay for app config.

``CoreSettings`` is the universal base. ``CoreSettings.from_yaml(path)``
loads env vars (from ``.env``) and then overlays YAML values on top, so
the singleton path (``get_settings``) sees the same end-state as
``lean.core.transports.yaml_loader.build_from_yaml``.

The single overlay rule lives in ``_apply_settings_overrides`` here so
there is exactly one canonical implementation — ``yaml_loader`` and
``CoreSettings.from_yaml`` both call it.
"""

from __future__ import annotations

import logging
import os
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


def _apply_overlay_to_kwargs(
    kwargs: dict[str, Any], field_names: set[str], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Overlay ``overrides`` onto a kwargs dict for ``cls(**kwargs)``.

    Same rules as ``_apply_settings_overrides``: flat keys, composite
    ``f"{section}_{key}"``, and unknown keys stashed under
    ``domain_config``. Returns a new dict — does not mutate ``kwargs``.

    Empty-string values in the overlay are skipped (not used to clobber
    a non-empty value already supplied by env vars): an operator who
    leaves ``settings.marker.remote_url: ""`` in the YAML but sets
    ``MARKER_REMOTE_URL=http://gpu:8000`` in the environment gets the
    GPU server. Use a literal placeholder string if you really want
    to clobber.
    """
    out = dict(kwargs)
    for section, sub in overrides.items():
        if not isinstance(sub, dict):
            value = sub
            if value in ("", None):
                continue
            if section in field_names:
                out[section] = value
                continue
            bucket = out.setdefault("domain_config", {})
            bucket[section] = value
            continue
        for key, value in sub.items():
            if value in ("", None):
                continue
            if key in field_names:
                out[key] = value
                continue
            composite = f"{section}_{key}"
            if composite in field_names:
                out[composite] = value
                continue
            bucket = out.setdefault("domain_config", {})
            bucket_section = bucket.setdefault(section, {})
            bucket_section[key] = value
    return out


def _read_env_kwargs(cls: type[CoreSettings]) -> dict[str, Any]:
    """Read every field's env var and return a kwargs dict for ``cls(**kwargs)``.

    Reads ``os.environ`` for each field's ``validation_alias`` (uppercased)
    or field name. Skips fields whose env var isn't set. Used by
    ``from_yaml`` so the YAML overlay is applied via kwargs (no need
    to construct, mutate, and re-validate).
    """
    out: dict[str, Any] = {}
    for name, field in cls.model_fields.items():
        alias = field.validation_alias or name
        env_val = os.environ.get(str(alias).upper())
        if env_val is not None:
            out[name] = env_val
    return out


def _apply_settings_overrides(instance: CoreSettings, overrides: dict[str, Any]) -> None:
    """Overlay a settings dict onto a CoreSettings instance.

    Used by ``lean.core.transports.yaml_loader`` callers that already
    have an instance (e.g. legacy tests). The YAML-aware constructor
    path uses ``_apply_overlay_to_kwargs`` + ``cls(**kwargs)`` instead.
    """
    fields = type(instance).model_fields
    for section, sub in overrides.items():
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
    """Universal settings: env vars (secrets) + pydantic defaults + YAML overlay.

    ``from_yaml(path)`` reads env first, then overlays the YAML so a
    missing Python default for ``mcp_http_port`` / ``api_port`` can be
    supplied by either ``settings.transport.mcp_port`` in YAML or
    ``MCP_HTTP_PORT`` env var.
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

    marker_remote_url: str = Field(
        default="",
        description="Remote marker server URL (empty = local CPU)",
        validation_alias="MARKER_REMOTE_URL",
    )
    marker_force_ocr: bool = False

    ocr_model: str = "baidu/Unlimited-OCR"
    ocr_dpi: int = 300
    ocr_timeout_s: float = 1800.0
    ocr_max_tokens: int = 32768
    ocr_batch_size: int = 20

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
    mcp_http_host: str = Field(default="127.0.0.1", validation_alias="MCP_HTTP_HOST")
    mcp_http_port: int = Field(validation_alias="MCP_HTTP_PORT")
    api_port: int = Field(validation_alias="API_PORT")

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
    llm_api_key: str = Field(
        default="",
        description="LLM provider API key (MiniMax by default)",
        validation_alias="MINIMAX_API_KEY",
    )
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

    # OCR server URL — universal env-overridable knob (used by the pdf_lss domain).
    ocr_base_url: str = Field(
        default="", description="OCR server URL", validation_alias="OCR_BASE_URL"
    )
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
        vlm_enabled = bool(self.domain_config.get("vlm", {}).get("enabled", False))
        if vlm_enabled and (not self.vlm_base_url or not self.vlm_model):
            raise ValueError(
                "vlm.enabled=true requires settings.vlm.base_url and settings.vlm.model to be set"
            )
        if vlm_enabled:
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
        if self.ocr_dpi < 50 or self.ocr_dpi > 1200:
            raise ValueError(f"ocr_dpi must be in [50, 1200], got {self.ocr_dpi}")
        if self.ocr_timeout_s <= 0:
            raise ValueError(f"ocr_timeout_s must be > 0, got {self.ocr_timeout_s}")
        if self.ocr_max_tokens < 256:
            raise ValueError(f"ocr_max_tokens must be >= 256, got {self.ocr_max_tokens}")
        if self.ocr_batch_size < 1:
            raise ValueError(f"ocr_batch_size must be >= 1, got {self.ocr_batch_size}")
        return self

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> CoreSettings:
        """Build a Settings instance: env vars first, then YAML overlay, then validate.

        Reads every field's env var into a kwargs dict, applies the YAML
        ``settings:`` block (only) to the dict, then constructs via
        ``cls(**kwargs)``. Pydantic-settings requires kwargs to be
        keyed by ``validation_alias`` (not field name) when an alias is
        set, so the overlay results are re-keyed accordingly.

        Top-level YAML keys (``domain``, ``extractors``, ``vlm``, etc.)
        belong to ``DomainConfig`` and are not merged into settings here.

        Pass ``None`` to skip the YAML step (uses pure env + defaults).
        """
        yaml_data = _load_yaml(config_path)
        env_kwargs = _read_env_kwargs(cls)
        field_names = set(cls.model_fields.keys())
        settings_block = yaml_data.get("settings", {})
        merged = _apply_overlay_to_kwargs(env_kwargs, field_names, settings_block)
        aliased: dict[str, Any] = {}
        for name, value in merged.items():
            field = cls.model_fields[name]
            key = str(field.validation_alias) if field.validation_alias else name
            aliased[key] = value
        return cls(**aliased)


# Backward-compat alias — code paths that imported ``Settings`` before
# the lean split. Domains should subclass ``CoreSettings``.
Settings = CoreSettings


# Singleton — per-subclass so subclasses don't share a cached instance.
_settings_cache: dict[type, Any] = {}
_active_yaml_path: Path | None = None


def set_active_yaml_path(path: Path | None) -> None:
    """Pin the YAML path used by the next ``get_settings()`` call.

    Called by ``lean.core.transports.yaml_loader.build_from_yaml`` so
    the singleton reads the manifest even though validation runs
    during ``cls()``. Cleared by ``clear_settings_cache()``.
    """
    global _active_yaml_path
    _active_yaml_path = path


def get_settings(cls: type[CoreSettings] = CoreSettings) -> CoreSettings:
    """Cached singleton for the given Settings subclass.

    When ``set_active_yaml_path`` has been called (i.e. the process is
    being driven by ``build_from_yaml``), the YAML overlay is applied
    before the instance is cached. Tests should call
    ``clear_settings_cache()`` after mutating env.
    """
    if cls not in _settings_cache:
        _settings_cache[cls] = cls.from_yaml(_active_yaml_path)
    return _settings_cache[cls]  # type: ignore[no-any-return]


def clear_settings_cache() -> None:
    """Clear all cached settings — call in tests after env mutation."""
    global _active_yaml_path
    _active_yaml_path = None
    _settings_cache.clear()


__all__ = [
    "CoreSettings",
    "Settings",
    "get_settings",
    "clear_settings_cache",
]
