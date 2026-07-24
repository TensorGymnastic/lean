"""YAML → Domain runtime.

Reads a ``DomainConfig``, instantiates the components it references,
and builds the three transports (MCP / REST / CLI).
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import anyio

from lean.core.config.domain_config import DomainConfig, ExtractorRef, VLMConfig
from lean.core.config.settings import (
    CoreSettings,
    get_settings,
    set_active_yaml_path,
)
from lean.core.extraction.base import DomainHooks, Extractor, Pipeline, set_pipeline
from lean.core.transports.builder import TransportBuilder
from lean.core.transports.registration import DomainRegistration

logger = logging.getLogger(__name__)


# The module path on any dotted-path field (extractors, vlm.parser,
# metadata.extractor, tools.module) must start with one of these prefixes.
# Anything else is rejected: user-controlled YAML must not be able to
# import arbitrary Python modules (``subprocess``, ``os``, ``pickle``, …)
# and instantiate them with attacker-supplied kwargs.
ALLOWED_ADAPTER_PREFIXES: tuple[str, ...] = (
    "lean.core.",
    "lean.domains.",
)


def _is_allowed(module_path: str) -> bool:
    return any(module_path.startswith(prefix) for prefix in ALLOWED_ADAPTER_PREFIXES)


def _import_object(dotted: str) -> Any:
    """Import a Python object by dotted path.

    Example: ``lean.core.extraction.MarkerExtractor``.

    The module path must start with one of ``ALLOWED_ADAPTER_PREFIXES``;
    otherwise a ``ValueError`` is raised. Import errors are also wrapped
    in ``ValueError`` so callers see an actionable message instead of a
    Python traceback.
    """
    if ":" in dotted:
        module_path, attr = dotted.split(":", 1)
    else:
        module_path, _, attr = dotted.rpartition(".")
    if not module_path or not attr:
        raise ValueError(f"invalid dotted path: {dotted!r}")
    if not _is_allowed(module_path):
        raise ValueError(f"adapter {dotted!r} not in allowed prefixes {ALLOWED_ADAPTER_PREFIXES}")
    try:
        module = importlib.import_module(module_path)
        obj = getattr(module, attr)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f"could not import adapter {dotted!r}: {e}") from e
    return obj


def _merge_extractor_defaults(
    adapter: str, config: dict[str, Any], settings: CoreSettings
) -> dict[str, Any]:
    """Overlay ``settings`` defaults onto ``config`` for known backends.

    Settings is the single source of truth for marker/ocr tunables. If
    the extractor config block has an empty string (or the field is
    missing), the value from ``settings`` wins — so a YAML author can
    declare ``settings.marker.remote_url`` once and have it apply to
    every marker extractor in the chain.

    Returns a new dict — does not mutate ``config``.
    """
    out = dict(config)
    name = adapter.rsplit(".", 1)[-1]
    if name == "MarkerExtractor":
        if not out.get("remote_url"):
            out["remote_url"] = settings.marker_remote_url
        if "force_ocr" not in out:
            out["force_ocr"] = settings.marker_force_ocr
    if name == "UnlimitedOCRExtractor":
        if not out.get("base_url"):
            out["base_url"] = settings.ocr_base_url
        if not out.get("model"):
            out["model"] = settings.ocr_model
        if not out.get("dpi"):
            out["dpi"] = settings.ocr_dpi
        if not out.get("timeout"):
            out["timeout"] = settings.ocr_timeout_s
        if not out.get("max_tokens"):
            out["max_tokens"] = settings.ocr_max_tokens
        if not out.get("batch_size"):
            out["batch_size"] = settings.ocr_batch_size
    return out


def _load_extractor(ref: ExtractorRef, settings: CoreSettings) -> Extractor:
    """Load an extractor adapter and instantiate it with its config."""
    obj_or_cls = _import_object(ref.adapter)
    config = _merge_extractor_defaults(ref.adapter, ref.config, settings)
    if isinstance(obj_or_cls, type):
        return obj_or_cls(**config)  # type: ignore[no-any-return]
    return obj_or_cls(config)  # type: ignore[no-any-return]


def _build_pipeline(domain_config: DomainConfig, settings: CoreSettings) -> Pipeline:
    extractors = [_load_extractor(ref, settings) for ref in domain_config.extractors]
    hooks = _build_vlm_hooks(domain_config, settings)
    return Pipeline(extractors, hooks=hooks)


def _build_vlm_hooks(domain_config: DomainConfig, settings: CoreSettings) -> DomainHooks:
    """Build DomainHooks for VLM enrichment (empty hooks if VLM disabled)."""
    if not domain_config.vlm.enabled:
        return DomainHooks()

    vlm = domain_config.vlm
    prompt = _resolve_prompt(vlm, domain_config)

    from lean.core.extraction.pipeline_helpers import hash_image
    from lean.core.vlm import OpenAICompatibleVLM

    # Create the VLM client once at hook-build time — not per-image.
    client = OpenAICompatibleVLM(
        base_url=settings.vlm_base_url,
        model=settings.vlm_model,
        api_key=settings.vlm_api_key,
        timeout=settings.vlm_timeout_s,
        detail=settings.vlm_detail,
        disable_thinking=settings.vlm_disable_thinking,
    )
    max_tokens = settings.vlm_max_tokens

    async def _describe_one(
        name: str, image: object, settings: CoreSettings, _: str
    ) -> tuple[str, dict[str, object], str]:
        raw = await anyio.to_thread.run_sync(
            lambda: client.describe_image(image, prompt=prompt, max_tokens=max_tokens)
        )

        parsed: dict[str, object]
        if vlm.parser:
            parser = _import_object(vlm.parser)
            parsed = dict(parser(raw))
        else:
            parsed = {"chart_type": "unknown", "description": raw.strip()}

        embed_text = str(parsed.get("description", raw.strip()))
        if parsed.get("title"):
            embed_text = f"{parsed['title']}\n\n{embed_text}"
        if parsed.get("key_data_points"):
            points = parsed["key_data_points"]
            if isinstance(points, list):
                embed_text += "\n\nKey data: " + "; ".join(str(p) for p in points)
        return embed_text, parsed, hash_image(image)  # type: ignore[arg-type]

    fmt = vlm.image_heading_format

    def _heading_for_chunk(chunk: object) -> str:
        meta = getattr(chunk, "image_meta", None) or {}
        title = meta.get("title") if isinstance(meta, dict) else None
        return str(title) if title else fmt.format(n=0)

    return DomainHooks(describe_one=_describe_one, heading_for_chunk=_heading_for_chunk)


def _resolve_prompt(vlm: VLMConfig, domain_config: DomainConfig) -> str:
    """Resolve the VLM prompt from inline text or a file path."""
    if vlm.prompt_inline:
        return vlm.prompt_inline
    if vlm.prompt_file:
        path = Path(vlm.prompt_file)
        if not path.is_absolute():
            yaml_path = getattr(domain_config, "_yaml_path", None)
            if yaml_path is not None:
                path = Path(yaml_path).parent / path
        return path.read_text(encoding="utf-8")
    return "Describe this image in detail."


def _load_tools_module(domain_config: DomainConfig) -> Any | None:
    """Load the domain's tools module if specified.

    Returns the imported module object (callers extract ``register_mcp``,
    ``register_api``, ``register_cli``, ``discover``, and any individual
    tool callables from it).
    """
    if not domain_config.tools.module:
        return None
    return importlib.import_module(domain_config.tools.module)


def build_from_yaml(yaml_path: Path) -> TransportBuilder:
    """Read a domain YAML and return a fully-wired TransportBuilder."""
    domain_config = DomainConfig.from_yaml(yaml_path)
    object.__setattr__(domain_config, "_yaml_path", yaml_path)

    set_active_yaml_path(yaml_path)
    settings = get_settings()

    pipeline = _build_pipeline(domain_config, settings)
    set_pipeline(pipeline)

    tools_module = _load_tools_module(domain_config)

    return _DomainBuilder(
        config=domain_config,
        settings=settings,
        pipeline=pipeline,
        tools_module=tools_module,
    ).build()


class _DomainBuilder:
    """Wires the YAML config + tools module into a TransportBuilder."""

    def __init__(
        self,
        *,
        config: DomainConfig,
        settings: CoreSettings,
        pipeline: Pipeline,
        tools_module: Any | None,
    ) -> None:
        self._config = config
        self._settings = settings
        self._pipeline = pipeline
        self._tools_module = tools_module

    def build(self) -> TransportBuilder:
        return TransportBuilder(_YamlDomain(self._config, self._tools_module))


class _YamlDomain(DomainRegistration[CoreSettings]):
    """DomainRegistration-shaped adapter that delegates to the tools module."""

    def __init__(self, config: DomainConfig, tools_module: Any | None) -> None:
        self._config = config
        self._tools_module = tools_module
        self.name = config.domain.name
        self.version = config.domain.version
        self.description = config.domain.description

    def build_settings(self) -> CoreSettings:
        return get_settings()

    def build_pipeline(self, settings: CoreSettings) -> Pipeline:
        return get_pipeline()

    def register_mcp(self, mcp: Any, services: dict[str, object], settings: CoreSettings) -> None:
        if self._tools_module is None:
            return
        from lean.core.adapters import register_mcp_tools

        register_mcp_tools(mcp, self._tools_module)

    def register_api(self, app: Any, services: dict[str, object], settings: CoreSettings) -> None:
        if self._tools_module is None:
            return
        from lean.core.adapters import register_api_tools

        register_api_tools(app, self._tools_module)

    def register_cli(self, app: Any, services: dict[str, object], settings: CoreSettings) -> None:
        if self._tools_module is None:
            return
        from lean.core.adapters import register_cli_tools

        register_cli_tools(app, self._tools_module)


def get_pipeline() -> Pipeline:
    """Return the registered pipeline (re-export for clarity)."""
    from lean.core.extraction.base import get_pipeline as _gp

    return _gp()


__all__ = ["build_from_yaml"]
