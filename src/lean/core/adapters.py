"""Shared decorator helpers used by every domain's tools.py module.

A domain's tools module declares each tool with the right decorator;
the YAML loader calls ``discover()`` to find them and ``register_mcp /
register_api / register_cli`` to wire them into the transports.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypeVar, overload

import typer

F = TypeVar("F", bound=Callable[..., Any])


@overload
def mcp_tool[F: Callable[..., Any]](fn: F, *, name: str | None = ...) -> F: ...
@overload
def mcp_tool(fn: None = ..., *, name: str | None = ...) -> Callable[[F], F]: ...
def mcp_tool[F: Callable[..., Any]](
    fn: F | None = None,
    *,
    name: str | None = None,
) -> Callable[[F], F] | F:
    """Marker decorator — auto-registered as an MCP tool.

    Use as ``@mcp_tool`` (bare) or ``@mcp_tool(name="x")``.
    """
    if fn is not None and callable(fn):
        fn.__lean_tool_kind__ = "mcp"  # type: ignore[attr-defined]
        fn.__lean_tool_name__ = name or fn.__name__  # type: ignore[attr-defined]
        return fn

    def deco(f: F) -> F:
        f.__lean_tool_kind__ = "mcp"  # type: ignore[attr-defined]
        f.__lean_tool_name__ = name or f.__name__  # type: ignore[attr-defined]
        return f

    return deco


def rest_route(method: str, path: str) -> Callable[[F], F]:
    """Marker decorator — auto-registered as a REST route."""

    def deco(fn: F) -> F:
        fn.__lean_tool_kind__ = "rest"  # type: ignore[attr-defined]
        fn.__lean_route__ = (method.upper(), path)  # type: ignore[attr-defined]
        return fn

    return deco


@overload
def cli_command[F: Callable[..., Any]](fn: F, *, name: str | None = ...) -> F: ...
@overload
def cli_command(fn: None = ..., *, name: str | None = ...) -> Callable[[F], F]: ...
def cli_command[F: Callable[..., Any]](
    fn: F | None = None,
    *,
    name: str | None = None,
) -> Callable[[F], F] | F:
    """Marker decorator — auto-registered as a Typer subcommand.

    Use as ``@cli_command`` (bare) or ``@cli_command(name="x")``.
    """
    if fn is not None and callable(fn):
        fn.__lean_tool_kind__ = "cli"  # type: ignore[attr-defined]
        fn.__lean_command_name__ = name or fn.__name__  # type: ignore[attr-defined]
        return fn

    def deco(f: F) -> F:
        f.__lean_tool_kind__ = "cli"  # type: ignore[attr-defined]
        f.__lean_command_name__ = name or f.__name__  # type: ignore[attr-defined]
        return f

    return deco


def discover_tools(module: object) -> dict[str, Callable[..., Any]]:
    """Return all decorated tool functions on ``module``, keyed by name."""
    import sys

    if hasattr(module, "__dict__"):
        mod = module
    else:
        mod = sys.modules[getattr(module, "__name__", module.__class__.__name__)]
    skip: set[str] = set()
    skip.update(
        {
            "asyncio",
            "fnmatch",
            "json",
            "Path",
            "Any",
            "typer",
            "discover_tools",
            "register_mcp",
            "register_api",
            "register_cli",
            "tools",
            "sys",
            "mod",
        }
    )
    skip.update(
        {
            "Chunk",
            "CorpusStats",
            "DocumentSummary",
            "IngestResult",
            "_ingest_pdf",
            "_search",
            "_get_chunk",
            "_list_documents",
            "_corpus_stats",
            "_delete_document",
            "_get_markdown",
            "_output",
        }
    )
    tools: dict[str, Callable[..., Any]] = {}
    for name in dir(mod):
        if name.startswith("_") or name in skip:
            continue
        attr = getattr(mod, name, None)
        if attr is None:
            continue
        if callable(attr) and hasattr(attr, "__lean_tool_kind__"):
            tools[name] = attr
    return tools


def register_mcp_tools(mcp: Any, module: object) -> None:
    """Register all MCP-decorated functions on the FastMCP app."""
    for fn in discover_tools(module).values():
        if getattr(fn, "__lean_tool_kind__", None) == "mcp":
            mcp.tool(fn)


def register_api_tools(app: Any, module: object) -> None:
    """Register all REST-decorated functions on the FastAPI app."""
    for fn in discover_tools(module).values():
        if getattr(fn, "__lean_tool_kind__", None) == "rest":
            route = getattr(fn, "__lean_route__", None)
            if route is None:
                continue
            method, path = route
            app.add_api_route(path, fn, methods=[method])


def register_cli_tools(app: typer.Typer, module: object) -> None:
    """Register all CLI-decorated functions on the Typer app."""
    for fn in discover_tools(module).values():
        if getattr(fn, "__lean_tool_kind__", None) == "cli":
            name = getattr(fn, "__lean_command_name__", fn.__name__)
            app.command(name=name)(fn)


def output(data: object, json_mode: bool) -> None:
    """Render data for the CLI: Pydantic model, list of models, or anything else."""
    if hasattr(data, "model_dump_json"):
        typer.echo(data.model_dump_json(indent=2))
        return
    if isinstance(data, list):
        if json_mode:
            typer.echo(
                json.dumps(
                    [d.model_dump(mode="json") if hasattr(d, "model_dump") else d for d in data],
                    indent=2,
                    default=str,
                )
            )
            return
        for item in data:
            if hasattr(item, "model_dump_json"):
                typer.echo(item.model_dump_json(indent=2))
            else:
                typer.echo(item)
        return
    typer.echo(json.dumps(data, indent=2, default=str))


__all__ = [
    "mcp_tool",
    "rest_route",
    "cli_command",
    "discover_tools",
    "register_mcp_tools",
    "register_api_tools",
    "register_cli_tools",
    "output",
]
