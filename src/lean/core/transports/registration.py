"""Domain registration protocol — the seam between lean and any domain.

A domain (e.g. lean-lss) subclasses ``DomainRegistration`` and
implements the hook methods to inject its extraction pipeline, VLM
prompt, schema extensions, and MCP/REST/CLI tool definitions.

The ``TransportBuilder`` calls these hooks exactly once at startup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from lean.core.config.settings import CoreSettings
from lean.core.extraction.base import Pipeline

if TYPE_CHECKING:
    from fastmcp import FastMCP


SettingsT = TypeVar("SettingsT", bound=CoreSettings)


class DomainRegistration[SettingsT: CoreSettings](ABC):
    """A domain plugin: provides settings + extraction + tool definitions.

    Subclass and implement the abstract methods to register a domain.
    The base class implements a sane default for each hook that returns
    "no additional behavior", so a minimal subclass only needs to
    override the methods it cares about.
    """

    name: str = "lean"
    version: str = "0.0.0"
    description: str = ""

    @abstractmethod
    def build_settings(self) -> SettingsT:
        """Return a (possibly subclassed) ``CoreSettings`` instance.

        Domain subclasses of ``CoreSettings`` add domain-specific fields
        (e.g. lean-lss adds `marker_remote_url`, `vlm_prompt_template`).
        """
        ...

    @abstractmethod
    def build_pipeline(self, settings: SettingsT) -> Pipeline:
        """Build the extraction pipeline (one or more ``Extractor`` instances).

        The returned ``Pipeline`` is what ``ingest_pdf`` invokes.
        """
        ...

    def register_mcp(self, mcp: FastMCP, services: dict[str, object], settings: SettingsT) -> None:
        """Register MCP tools/resources/prompts on the supplied ``FastMCP`` instance.

        Default: no-op. Domains decorate ``mcp`` with their tools.
        """
        return None

    def register_api(self, app: Any, services: dict[str, object], settings: SettingsT) -> None:
        """Register FastAPI routes on the supplied ``app``. Default: no-op."""
        return None

    def register_cli(self, app: Any, services: dict[str, object], settings: SettingsT) -> None:
        """Register Typer commands on the supplied ``app``. Default: no-op."""
        return None


__all__ = ["DomainRegistration"]
