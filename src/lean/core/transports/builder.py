"""TransportBuilder — the universal entry point for a domain deployment.

Usage from a domain's CLI/API/MCP entrypoint:

    from lean.core.transports import TransportBuilder
    from lean.core.transports.yaml_loader import build_from_yaml

    if __name__ == "__main__":
        builder = build_from_yaml(Path("configs/lean-pdf-lss.yaml"))
        builder.cli().run()
"""

from __future__ import annotations

from typing import Any

from lean.core.config.settings import CoreSettings
from lean.core.extraction.base import set_pipeline
from lean.core.transports.registration import DomainRegistration


class TransportBuilder:
    """Builds the three transports (MCP / REST / CLI) from a domain registration.

    On construction:
    - Builds the domain's ``Settings`` (subclassed from ``CoreSettings``).
    - Builds the domain's ``Pipeline`` and installs it as the singleton.
    - Builds the MCP app, the FastAPI app, and the Typer CLI app,
      with the domain's tools/routes/commands registered.

    The returned objects are independent — domains can serve one,
    multiple, or all three.
    """

    def __init__(self, domain: DomainRegistration[Any]) -> None:
        self._domain = domain
        self._settings: CoreSettings = domain.build_settings()
        self._pipeline = domain.build_pipeline(self._settings)
        set_pipeline(self._pipeline)

        self._services = self._build_services()

        self._mcp = self._build_mcp()
        self._api = self._build_api()
        self._cli = self._build_cli()

    def _build_services(self) -> dict[str, Any]:
        from lean.core.services.corpus import (
            corpus_stats,
            delete_document,
            get_chunk,
            get_document_markdown,
            list_chunks_by_document,
            list_documents,
        )
        from lean.core.services.ingestion import ingest_pdf, reingest
        from lean.core.services.search import search

        return {
            "search": search,
            "ingest_pdf": ingest_pdf,
            "reingest": reingest,
            "list_documents": list_documents,
            "list_chunks_by_document": list_chunks_by_document,
            "get_chunk": get_chunk,
            "get_document_markdown": get_document_markdown,
            "delete_document": delete_document,
            "corpus_stats": corpus_stats,
        }

    def _build_mcp(self) -> Any:
        from lean.core.transports.mcp import build_mcp

        return build_mcp(
            name=self._domain.name,
            version=self._domain.version,
            description=self._domain.description,
            services=self._services,
            settings=self._settings,
            register_fn=self._domain.register_mcp,
        )

    def _build_api(self) -> Any:
        from lean.core.transports.api import build_api

        return build_api(
            name=self._domain.name,
            version=self._domain.version,
            description=self._domain.description,
            register_fn=self._domain.register_api,
            settings=self._settings,
            services=self._services,
        )

    def _build_cli(self) -> Any:
        from lean.core.transports.cli import build_cli

        return build_cli(
            name=self._domain.name,
            help_text=self._domain.description or f"{self._domain.name} CLI",
            settings_factory=lambda: self._settings,
            build_mcp_fn=lambda: self._mcp,
            build_api_fn=lambda: self._api,
            serve_mcp_fn=lambda mcp, **kw: self._serve_mcp(mcp, **kw),
            serve_api_fn=lambda app, **kw: self._serve_api(app, **kw),
            register_fn=self._domain.register_cli,
        )

    def _serve_mcp(self, mcp: Any, *, transport: str, host: str, port: int) -> None:
        from lean.core.transports.mcp import serve_mcp

        serve_mcp(mcp, transport=transport, host=host, port=port, settings=self._settings)

    def _serve_api(self, app: Any, *, host: str, port: int, reload: bool = False) -> None:
        from lean.core.transports.api import serve_api

        serve_api(app, host=host, port=port, reload=reload)

    @property
    def settings(self) -> CoreSettings:
        return self._settings

    @property
    def services(self) -> dict[str, object]:
        return self._services

    @property
    def mcp(self) -> Any:
        return self._mcp

    @property
    def api(self) -> Any:
        return self._api

    @property
    def cli(self) -> Any:
        return self._cli


__all__ = ["TransportBuilder"]
