"""YAML-driven domain configuration schema.

A domain YAML declares:
  - metadata:    name, version, description
  - settings:    overrides / extensions to CoreSettings (yaml fields)
  - extractors:  ordered list of extractor adapters (dotted-path + config)
  - vlm:         optional VLM prompt file + parser dotted-path
  - metadata:    optional PDF metadata extractor dotted-path
  - tools:       optional Python module to auto-discover MCP/API/CLI tools from
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExtractorRef(BaseModel):
    """A single extractor in the chain — references a Python class by dotted path."""

    adapter: str = Field(
        description=(
            "Dotted path to an Extractor class (e.g. lean.core.extraction.MarkerExtractor)"
        )
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="kwargs passed to the adapter constructor"
    )


class VLMConfig(BaseModel):
    """VLM enrichment — if enabled, image-bearing chunks get descriptions."""

    enabled: bool = False
    prompt_file: str | None = Field(
        default=None, description="Path to a .txt or .md file with the VLM prompt"
    )
    prompt_inline: str | None = Field(
        default=None, description="Inline prompt (alternative to prompt_file)"
    )
    parser: str | None = Field(
        default=None, description="Dotted path to a function (raw_text -> dict)"
    )
    image_heading_format: str = Field(
        default="Image {n}", description="Format string for image chunk headings"
    )


class MetadataConfig(BaseModel):
    """PDF metadata extraction — only used by PDF-based domains."""

    extractor: str | None = Field(
        default=None, description="Dotted path to a function (Path -> PdfMetadata)"
    )


class ToolsConfig(BaseModel):
    """Where to find domain-specific tools (MCP tools, REST routes, CLI commands)."""

    model_config = ConfigDict(extra="forbid")

    module: str | None = Field(default=None, description="Dotted path to a Python module to scan")


class DomainMetadata(BaseModel):
    """Identity block at the top of every domain YAML."""

    name: str
    version: str = "0.1.0"
    description: str = ""


class DomainConfig(BaseModel):
    """The full domain manifest — loaded from a YAML file.

    A YAML may contain settings overrides under any top-level key.
    Keys that don't match DomainConfig fields are forwarded to the
    CoreSettings overlay (so domains can add their own knobs).
    """

    domain: DomainMetadata
    settings: dict[str, Any] = Field(default_factory=dict)
    extractors: list[ExtractorRef] = Field(default_factory=list)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @field_validator("extractors")
    @classmethod
    def _at_least_one_extractor(cls, v: list[ExtractorRef]) -> list[ExtractorRef]:
        if not v:
            raise ValueError("at least one extractor is required")
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> DomainConfig:
        """Load a domain manifest from a YAML file.

        Top-level YAML keys that aren't part of DomainConfig are stored
        under ``settings`` so the domain can still consume them via
        ``CoreSettings.domain_config``.

        Raises ``ValueError`` for malformed YAML or non-mapping top-level
        values (instead of leaking ``yaml.YAMLError`` / a raw ``{}``).
        """
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"invalid YAML in {path}: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(
                f"domain YAML must be a mapping, got {type(data).__name__} (in {path})"
            )
        known = {f.alias or name for name, f in cls.model_fields.items()}
        extras = {k: v for k, v in data.items() if k not in known}
        if extras:
            existing = data.get("settings", {})
            merged = {**extras, **(existing or {})}
            data["settings"] = merged
        return cls(**data)


__all__ = [
    "DomainConfig",
    "DomainMetadata",
    "ExtractorRef",
    "VLMConfig",
    "MetadataConfig",
    "ToolsConfig",
]
