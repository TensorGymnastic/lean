"""YAML → Domain runtime.

Reads a ``DomainConfig``, instantiates the components it references,
and builds the three transports (MCP / REST / CLI).
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from lean.core.config.domain_config import DomainConfig, ExtractorRef, VLMConfig
from lean.core.config.settings import CoreSettings, get_settings
from lean.core.extraction.base import Extractor, Pipeline, set_pipeline
from lean.core.transports.builder import TransportBuilder

logger = logging.getLogger(__name__)


def _import_object(dotted: str) -> Any:
    """Import a Python object by dotted path.

    Example: ``lean.domains.pdf_lss.adapters.MarkerAdapter``.
    """
    if ":" in dotted:
        module_path, attr = dotted.split(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    module_path, _, attr = dotted.rpartition(".")
    if not module_path:
        raise ValueError(f"invalid dotted path: {dotted!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _load_extractor(ref: ExtractorRef) -> Extractor:
    """Load an extractor adapter and instantiate it with its config."""
    obj_or_cls = _import_object(ref.adapter)
    if isinstance(obj_or_cls, type):
        return obj_or_cls(**ref.config)  # type: ignore[no-any-return]
    return obj_or_cls(ref.config)  # type: ignore[no-any-return]


def _build_pipeline(domain_config: DomainConfig) -> Pipeline:
    extractors = [_load_extractor(ref) for ref in domain_config.extractors]
    return Pipeline(extractors)


def _apply_settings_overrides(settings: CoreSettings, overrides: dict[str, Any]) -> CoreSettings:
    """Apply domain YAML settings to a CoreSettings instance.

    Keys that match a Settings field are set directly. Nested dicts
    are tried as composite ``section_key`` (e.g. ``embedding.model``).
    Unknown keys go to ``settings.domain_config`` for the domain.
    """
    fields = settings.model_fields
    for section, sub in overrides.items():
        if not isinstance(sub, dict):
            if section in fields:
                setattr(settings, section, sub)
            else:
                settings.domain_config[section] = sub
            continue
        for key, value in sub.items():
            if key in fields:
                setattr(settings, key, value)
                continue
            composite = f"{section}_{key}"
            if composite in fields:
                setattr(settings, composite, value)
                continue
            bucket = settings.domain_config.setdefault(section, {})
            if isinstance(bucket, dict):
                bucket[key] = value
    return settings


def _install_vlm_hooks(domain_config: DomainConfig) -> None:
    """Wire the VLM prompt + parser + image-heading format into the framework."""
    if not domain_config.vlm.enabled:
        return

    from lean.core.extraction.pipeline_helpers import set_chunk_type_for

    vlm = domain_config.vlm
    prompt = _resolve_prompt(vlm, domain_config)

    from lean.core.extraction.pipeline_helpers import hash_image
    from lean.core.vlm import OpenAICompatibleVLM, VLMError

    def _describe_one(
        name: str, image: object, settings: CoreSettings, _: str
    ) -> tuple[str, dict[str, object], str]:
        client = OpenAICompatibleVLM(
            base_url=settings.vlm_base_url,
            model=settings.vlm_model,
            api_key=settings.vlm_api_key,
            timeout=settings.vlm_timeout_s,
            detail=settings.vlm_detail,
            disable_thinking=settings.vlm_disable_thinking,
        )
        try:
            raw = client.describe_image(image, prompt=prompt, max_tokens=settings.vlm_max_tokens)
        except VLMError:
            raise
        finally:
            client.close()

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

    import lean.core.services.ingestion as ing

    ing._domain_describe_one = _describe_one  # type: ignore[attr-defined]

    fmt = vlm.image_heading_format

    def _heading_for_chunk(chunk: object) -> str:
        meta = getattr(chunk, "image_meta", None) or {}
        title = meta.get("title") if isinstance(meta, dict) else None
        return str(title) if title else fmt.format(n=0)

    set_chunk_type_for(_heading_for_chunk)


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


def _install_metadata_extractor(domain_config: DomainConfig) -> None:
    """Wire the domain's PDF metadata extractor if configured."""
    if not domain_config.metadata.extractor:
        return
    extractor = _import_object(domain_config.metadata.extractor)
    import lean.core.extraction.metadata as md_mod

    md_mod._custom_extractor = extractor  # type: ignore[attr-defined]


def _load_tools_module(domain_config: DomainConfig) -> Any | None:
    """Load the domain's tools module if specified.

    Returns the imported module object (callers extract ``register_mcp``,
    ``register_api``, ``register_cli``, ``discover``, and any individual
    tool callables from it).
    """
    if not domain_config.tools.module:
        return None
    return importlib.import_module(domain_config.tools.module)


def _apply_tools_filter(module: Any, domain_config: DomainConfig) -> dict[str, Any]:
    """Return a dict of filtered tool callables from the module."""
    if module is None:
        return {}
    tools: dict[str, Any] = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        if domain_config.tools.enabled and name not in domain_config.tools.enabled:
            continue
        if name in domain_config.tools.disabled:
            continue
        attr = getattr(module, name, None)
        if attr is None:
            continue
        if callable(attr) and hasattr(attr, "__lean_tool_kind__"):
            tools[name] = attr
    return tools


def build_from_yaml(yaml_path: Path) -> TransportBuilder:
    """Read a domain YAML and return a fully-wired TransportBuilder."""
    domain_config = DomainConfig.from_yaml(yaml_path)
    object.__setattr__(domain_config, "_yaml_path", yaml_path)

    settings = get_settings()
    _apply_settings_overrides(settings, domain_config.settings)

    pipeline = _build_pipeline(domain_config)
    set_pipeline(pipeline)

    _install_vlm_hooks(domain_config)
    _install_metadata_extractor(domain_config)
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
        from lean.core.transports.builder import TransportBuilder as _TB

        builder = _TB.__new__(_TB)
        builder._domain = _YamlDomain(self._config, self._tools_module)  # type: ignore[assignment]
        builder._settings = self._settings
        builder._pipeline = self._pipeline
        set_pipeline(self._pipeline)
        builder._services = builder._build_services()
        builder._mcp = builder._build_mcp()
        builder._api = builder._build_api()
        builder._cli = builder._build_cli()
        return builder


class _YamlDomain:
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
        register_fn = getattr(self._tools_module, "register_mcp", None)
        if register_fn is not None:
            register_fn(mcp)

    def register_api(self, app: Any, services: dict[str, object], settings: CoreSettings) -> None:
        if self._tools_module is None:
            return
        register_fn = getattr(self._tools_module, "register_api", None)
        if register_fn is not None:
            register_fn(app)

    def register_cli(self, app: Any, services: dict[str, object], settings: CoreSettings) -> None:
        if self._tools_module is None:
            return
        register_fn = getattr(self._tools_module, "register_cli", None)
        if register_fn is not None:
            register_fn(app)


def get_pipeline() -> Pipeline:
    """Return the registered pipeline (re-export for clarity)."""
    from lean.core.extraction.base import get_pipeline as _gp

    return _gp()


__all__ = ["build_from_yaml"]
