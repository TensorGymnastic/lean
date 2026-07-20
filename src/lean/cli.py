"""Lean CLI — universal entry point.

Usage:
    lean --config <yaml> ingest <path>
    lean --config <yaml> search "..."
    lean --config <yaml> mcp-serve
    lean --config <yaml> api-serve

The YAML manifest declares which domain to run (PDF + VLM, code, web,
or a custom adapter chain). The same CLI surface works across all
domains because the framework ships the universal commands and the
domain supplies the rest.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer

from lean.core.config.settings import clear_settings_cache
from lean.core.transports.yaml_loader import build_from_yaml

app = typer.Typer(
    name="lean",
    help="Universal MCP+pgvector corpus server. Domain selected via --config <yaml>.",
    no_args_is_help=True,
    add_completion=False,
)


def _resolve_config_path(value: str | None) -> Path:
    if value:
        p = Path(value)
        if not p.is_file():
            raise typer.BadParameter(f"config file not found: {value}")
        return p
    env = os.environ.get("LEAN_CONFIG")
    if env:
        p = Path(env)
        if not p.is_file():
            raise typer.BadParameter(f"config file not found (from LEAN_CONFIG): {env}")
        return p
    for candidate in (Path("lean.yaml"), Path("lean.yml")):
        if candidate.is_file():
            return candidate
    raise typer.BadParameter(
        "no --config given, no LEAN_CONFIG env var, and no lean.yaml in "
        "current directory. Pass --config <path-to-yaml> or set LEAN_CONFIG."
    )


@app.callback()
def _main(
    ctx: typer.Context,
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        envvar="LEAN_CONFIG",
        help="Path to the domain YAML manifest (or set LEAN_CONFIG).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Resolve the YAML config and inject the configured CLI into the context."""
    config_path = _resolve_config_path(config)

    if verbose:
        level = logging.DEBUG
    else:
        try:
            level = getattr(logging, _peek_log_level(config_path), logging.INFO)
        except (AttributeError, ValueError):
            level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    clear_settings_cache()
    builder = build_from_yaml(config_path)
    domain_name = getattr(builder.settings, "domain_name", None) or config_path.stem

    if verbose:
        typer.echo(f"lean: loaded domain '{domain_name}' from {config_path}", err=True)

    ctx.obj = {"config_path": config_path, "builder": builder}


def _peek_log_level(config_path: Path) -> str:
    """Cheap parse of just the log_level from YAML before settings load.

    Malformed YAML is logged at DEBUG and treated as INFO — the real
    settings load happens later and will produce its own actionable error.
    """
    import logging

    import yaml

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        logging.getLogger(__name__).debug(
            "could not peek log level from %s", config_path, exc_info=True
        )
        return "INFO"
    return str(data.get("settings", {}).get("log_level", "INFO")).upper()


def _extract_config_arg(argv: list[str]) -> str | None:
    """Pull the value of ``--config <path>`` or ``--config=<path>`` from argv."""
    for i, arg in enumerate(argv[1:], 1):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
        if arg == "-c" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _merge_cli_commands(target: typer.Typer, source: typer.Typer) -> None:
    """Copy all registered commands from source into target."""
    for cmd_info in source.registered_commands:
        callback = cmd_info.callback
        if callback is None:
            continue
        name = cmd_info.name or callback.__name__
        target.command(name=name)(callback)


def main() -> None:
    """Entry point: load domain YAML, then invoke the configured CLI."""
    config_arg = _extract_config_arg(sys.argv)
    config_path = _resolve_config_path(config_arg or os.environ.get("LEAN_CONFIG"))

    if config_arg is None:
        sys.argv = [sys.argv[0], "--config", str(config_path), *sys.argv[1:]]

    clear_settings_cache()
    try:
        builder = build_from_yaml(config_path)
    except ValueError as e:
        typer.echo(f"lean: {e}", err=True)
        sys.exit(2)
    _merge_cli_commands(app, builder.cli)
    app()


if __name__ == "__main__":
    main()
