"""Coverage tests for ``lean.core.adapters`` — the shared decorators.

The decorator/registration contract is the public surface. We exercise
each decorator + the three register_* helpers using inline functions.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import typer

from lean.core.adapters import (
    cli_command,
    mcp_tool,
    output,
    register_api_tools,
    register_cli_tools,
    register_mcp_tools,
    rest_route,
)


def test_mcp_tool_bare_sets_kind_and_name() -> None:
    """``@mcp_tool`` (bare) sets ``__lean_tool_kind__ = "mcp"`` and uses the function name."""

    @mcp_tool
    def fn(x: int) -> int:
        return x

    assert fn.__lean_tool_kind__ == "mcp"
    assert fn.__lean_tool_name__ == "fn"


def test_mcp_tool_with_explicit_name() -> None:
    """``@mcp_tool(name="x")`` overrides the function name."""

    @mcp_tool(name="custom")
    def fn() -> None:
        pass

    assert fn.__lean_tool_name__ == "custom"


def test_mcp_tool_callable_form() -> None:
    """``@mcp_tool()`` (with parens) also works."""

    @mcp_tool()
    def fn() -> None:
        pass

    assert fn.__lean_tool_kind__ == "mcp"


@mcp_tool
def mcp_register_target() -> int:
    return 42


@cli_command  # this is CLI, not MCP
def cli_register_target() -> int:
    return 99


def test_register_mcp_tools_calls_mcp_tool_decorator() -> None:
    """``register_mcp_tools`` invokes ``mcp.tool(fn)`` for every MCP-marked function."""
    mcp = MagicMock()
    register_mcp_tools(mcp, sys.modules[__name__])
    mcp.tool.assert_called_once()
    called_fn = mcp.tool.call_args[0][0]
    assert called_fn is mcp_register_target


def test_rest_route_sets_kind_route_tuple() -> None:
    """``@rest_route("GET", "/foo")`` sets ``__lean_tool_kind__ = "rest"`` + ``__lean_route__``."""

    @rest_route("GET", "/foo")
    def handler() -> dict[str, Any]:
        return {}

    assert handler.__lean_tool_kind__ == "rest"
    assert handler.__lean_route__ == ("GET", "/foo")


@rest_route("POST", "/things_for_test")
def rest_make_target() -> dict[str, Any]:
    return {}


@rest_route("DELETE", "/things_for_test/{id}")
def rest_delete_target(id: int) -> dict[str, Any]:
    return {}


def test_register_api_tools_calls_add_api_route() -> None:
    """``register_api_tools`` registers each REST function with the correct method + path."""
    app = MagicMock()
    register_api_tools(app, sys.modules[__name__])
    assert app.add_api_route.call_count == 2


def test_cli_command_sets_kind_cli() -> None:
    """``@cli_command`` sets ``__lean_tool_kind__ = "cli"``."""

    @cli_command
    def cmd() -> None:
        pass

    assert cmd.__lean_tool_kind__ == "cli"
    assert cmd.__lean_command_name__ == "cmd"


def test_cli_command_with_name() -> None:
    """``@cli_command(name="x")`` sets ``__lean_command_name__``."""

    @cli_command(name="custom")
    def cmd() -> None:
        pass

    assert cmd.__lean_command_name__ == "custom"


@cli_command
def mycmd12345_for_test() -> None:
    pass


def test_register_cli_tools_registers_typer_command() -> None:
    """``register_cli_tools`` registers each CLI-marked function on the Typer app."""
    app = typer.Typer()
    register_cli_tools(app, sys.modules[__name__])
    cmd_names = {ci.name or ci.callback.__name__ for ci in app.registered_commands}
    assert "mycmd12345_for_test" in cmd_names


def not_decorated() -> int:
    return 2


def test_discover_tools_returns_only_marked_callables() -> None:
    """``discover_tools`` returns only callables with ``__lean_tool_kind__`` set."""
    from lean.core.adapters import discover_tools

    mod = sys.modules[__name__]
    tools = discover_tools(mod)
    assert "mcp_register_target" in tools
    assert "not_decorated" not in tools


def test_discover_tools_filters_unmarked_callables_without_skip_set() -> None:
    """Characterization test: __lean_tool_kind__ alone filters non-tool names."""
    from lean.core.adapters import discover_tools

    class _FakeModule:
        Chunk = type("Chunk", (), {})
        CorpusStats = type("CorpusStats", (), {})
        asyncio = type("asyncio", (), {})
        json = type("json", (), {})

        def _private_helper(self) -> None:
            pass

        def not_a_tool(self) -> int:
            return 1

        @staticmethod
        @mcp_tool
        def real_tool() -> int:
            return 42

    tools = discover_tools(_FakeModule)
    assert "real_tool" in tools
    assert "Chunk" not in tools
    assert "CorpusStats" not in tools
    assert "asyncio" not in tools
    assert "json" not in tools
    assert "_private_helper" not in tools
    assert "not_a_tool" not in tools


def test_discover_tools_on_real_domain_does_not_raise() -> None:
    """discover_tools on the real pdf_lss tools module must not raise."""
    import importlib

    from lean.core.adapters import discover_tools

    mod = importlib.import_module("lean.domains.pdf_lss.tools")
    tools = discover_tools(mod)
    assert len(tools) > 0


def test_output_pydantic_model_human_mode() -> None:
    """``output(model, json_mode=False)`` calls ``model.model_dump_json``."""
    from pydantic import BaseModel

    class M(BaseModel):
        x: int = 1

    with patch("lean.core.adapters.typer.echo") as echo:
        output(M(), json_mode=False)
    assert echo.called


def test_output_list_of_pydantic_json_mode() -> None:
    """``output([m1, m2], json_mode=True)`` emits a JSON array."""
    from pydantic import BaseModel

    class M(BaseModel):
        x: int = 1

    with patch("lean.core.adapters.typer.echo") as echo:
        output([M(x=1), M(x=2)], json_mode=True)
    parsed = json.loads(echo.call_args[0][0])
    assert parsed == [{"x": 1}, {"x": 2}]


def test_output_dict_passthrough() -> None:
    """``output({"foo": "bar"}, json_mode=True)`` emits a JSON object."""
    with patch("lean.core.adapters.typer.echo") as echo:
        output({"foo": "bar"}, json_mode=True)
    parsed = json.loads(echo.call_args[0][0])
    assert parsed == {"foo": "bar"}


def test_output_list_human_mode() -> None:
    """``output([m1, m2], json_mode=False)`` calls ``model_dump_json`` on each item."""
    from pydantic import BaseModel

    class M(BaseModel):
        x: int = 1

    with patch("lean.core.adapters.typer.echo") as echo:
        output([M(x=1), M(x=2)], json_mode=False)
    assert echo.call_count == 2
