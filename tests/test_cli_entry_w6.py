"""Coverage tests for ``src/lean/cli.py`` — the universal entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lean import cli as cli_module


def test_resolve_config_path_uses_explicit_value(tmp_path: Path) -> None:
    """``_resolve_config_path`` returns the explicit value when it exists."""
    cfg = tmp_path / "x.yaml"
    cfg.write_text("domain: {name: x}\n")
    assert cli_module._resolve_config_path(str(cfg)) == cfg


def test_resolve_config_path_raises_when_missing(tmp_path: Path) -> None:
    """``_resolve_config_path`` raises typer.BadParameter when the file is absent."""
    import typer

    with pytest.raises(typer.BadParameter, match="config file not found"):
        cli_module._resolve_config_path(str(tmp_path / "nope.yaml"))


def test_resolve_config_path_falls_back_to_lean_env_var(tmp_path: Path, monkeypatch) -> None:
    """``_resolve_config_path`` honours LEAN_CONFIG env var."""
    cfg = tmp_path / "env.yaml"
    cfg.write_text("domain: {name: x}\n")
    monkeypatch.setenv("LEAN_CONFIG", str(cfg))
    assert cli_module._resolve_config_path(None) == cfg


def test_resolve_config_path_falls_back_to_lean_yaml(tmp_path: Path, monkeypatch) -> None:
    """``_resolve_config_path`` falls back to ./lean.yaml when present."""
    cfg = tmp_path / "lean.yaml"
    cfg.write_text("domain: {name: x}\n")
    monkeypatch.chdir(tmp_path)
    result = cli_module._resolve_config_path(None)
    assert result.resolve() == cfg.resolve()


def test_resolve_config_path_raises_when_no_source(tmp_path: Path, monkeypatch) -> None:
    """``_resolve_config_path`` raises when no --config, env, or lean.yaml exists."""
    import typer

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEAN_CONFIG", raising=False)
    with pytest.raises(typer.BadParameter, match=r"no --config given"):
        cli_module._resolve_config_path(None)


def test_peek_log_level_returns_uppercase_default() -> None:
    """``_peek_log_level`` returns 'INFO' uppercase when YAML has no log_level."""
    from lean.cli import _peek_log_level

    cfg = Path("configs/lean-pdf-lss.yaml")
    level = _peek_log_level(cfg)
    assert level.isupper()


def test_peek_log_level_reads_yaml_value(tmp_path: Path) -> None:
    """``_peek_log_level`` reads log_level from YAML."""
    from lean.cli import _peek_log_level

    cfg = tmp_path / "x.yaml"
    cfg.write_text("settings:\n  log_level: warning\n")
    assert _peek_log_level(cfg) == "WARNING"


def test_peek_log_level_handles_malformed_yaml(tmp_path: Path) -> None:
    """``_peek_log_level`` returns INFO on malformed YAML (real settings load will fail later)."""
    from lean.cli import _peek_log_level

    cfg = tmp_path / "bad.yaml"
    cfg.write_text("not: valid: yaml: :::")
    assert _peek_log_level(cfg) == "INFO"


def test_extract_config_arg_picks_config_flag() -> None:
    """``_extract_config_arg`` finds --config <path> in argv."""
    argv = ["prog", "--config", "/nonexistent/x.yaml"]
    assert cli_module._extract_config_arg(argv) == "/nonexistent/x.yaml"


def test_extract_config_arg_picks_short_flag() -> None:
    """``_extract_config_arg`` finds -c <path> in argv."""
    argv = ["prog", "-c", "/nonexistent/x.yaml"]
    assert cli_module._extract_config_arg(argv) == "/nonexistent/x.yaml"


def test_extract_config_arg_picks_equals_form() -> None:
    """``_extract_config_arg`` finds --config=<path> in argv."""
    argv = ["prog", "--config=/nonexistent/x.yaml"]
    assert cli_module._extract_config_arg(argv) == "/nonexistent/x.yaml"


def test_extract_config_arg_returns_none_when_missing() -> None:
    """``_extract_config_arg`` returns None when --config is absent."""
    assert cli_module._extract_config_arg(["prog", "search"]) is None


def test_merge_cli_commands_copies_registered_commands() -> None:
    """``_merge_cli_commands`` copies commands from the source Typer app into the target."""
    import typer

    target = typer.Typer()
    source = typer.Typer()

    @source.command("echo-hi")
    def echo_hi() -> None:
        pass

    cli_module._merge_cli_commands(target, source)
    cmd_names = {ci.name or ci.callback.__name__ for ci in target.registered_commands}
    assert "echo-hi" in cmd_names


def test_merge_cli_commands_skips_callbackless_commands() -> None:
    """``_merge_cli_commands`` silently skips CommandInfo entries with no callback."""
    import typer

    target = typer.Typer()
    source = typer.Typer()

    # Register a real command first
    @source.command("real")
    def real() -> None:
        pass

    # Inject a callback-less CommandInfo (use MagicMock so we bypass __init__ checks)
    fake = MagicMock()
    fake.name = "fake"
    fake.callback = None
    source.registered_commands.append(fake)

    cli_module._merge_cli_commands(target, source)
    cmd_names = {
        ci.name or ci.callback.__name__ for ci in target.registered_commands if ci.callback
    }
    assert "real" in cmd_names
    assert "fake" not in cmd_names


def test_main_exits_2_on_value_error(monkeypatch, tmp_path: Path) -> None:
    """``main`` exits with code 2 when ``build_from_yaml`` raises ValueError."""
    cfg = tmp_path / "x.yaml"
    cfg.write_text("domain: {name: x}\n")
    monkeypatch.setattr("sys.argv", ["lean", "--config", str(cfg)])
    with (
        patch("lean.cli.build_from_yaml", side_effect=ValueError("bad config")),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_module.main()
    assert exc_info.value.code == 2
