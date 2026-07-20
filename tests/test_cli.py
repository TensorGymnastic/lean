"""Smoke tests for the universal CLI entry point.

These tests verify CLI behavior end-to-end via subprocess (uv run lean ...)
so they exercise the full ``main()`` path including command merging.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_VALID_KEY = "x" * 32


def _have_uv() -> bool:
    return shutil.which("uv") is not None


def _have_pdf_lss_config() -> bool:
    return Path("configs/lean-pdf-lss.yaml").is_file()


pytestmark = pytest.mark.skipif(
    not (_have_uv() and _have_pdf_lss_config()),
    reason="requires uv + configs/lean-pdf-lss.yaml",
)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "lean", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_no_args_shows_help() -> None:
    """Running ``lean`` with no args and no LEAN_CONFIG exits with a clear error."""
    import os

    env = {k: v for k, v in os.environ.items() if k != "LEAN_CONFIG"}
    r = subprocess.run(
        ["uv", "run", "lean"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert r.returncode != 0
    assert "--config" in r.stderr or "LEAN_CONFIG" in r.stderr


def test_help_lists_universal_commands() -> None:
    """``--help`` lists all 5 universal commands (db_init, health, mcp-serve, api-serve, eval)."""
    r = _run(["--config", "configs/lean-pdf-lss.yaml", "--help"])
    assert "db_init" in r.stdout
    assert "health" in r.stdout
    assert "mcp-serve" in r.stdout
    assert "api-serve" in r.stdout
    assert "eval" in r.stdout


def test_help_lists_pdf_lss_specific_commands() -> None:
    """``--help`` for pdf_lss lists its 8 domain-specific commands."""
    r = _run(["--config", "configs/lean-pdf-lss.yaml", "--help"])
    assert "ingest" in r.stdout
    assert "search" in r.stdout
    assert "list-documents" in r.stdout
    assert "delete" in r.stdout
    assert "get-chunk" in r.stdout
    assert "get-markdown" in r.stdout
    assert "corpus-stats" in r.stdout


def test_help_lists_code_specific_commands() -> None:
    """``--help`` for code lists ingest-directory + ingest."""
    r = _run(["--config", "configs/lean-code.yaml", "--help"])
    assert "ingest" in r.stdout
    assert "ingest-directory" in r.stdout
    assert "search" in r.stdout


def test_help_lists_web_specific_commands() -> None:
    """``--help`` for web lists ingest-list + ingest."""
    r = _run(["--config", "configs/lean-web.yaml", "--help"])
    assert "ingest" in r.stdout
    assert "ingest-list" in r.stdout
    assert "search" in r.stdout


def test_eval_help_describes_metrics() -> None:
    """``lean eval --help`` describes the eval metrics (regression: command was missing)."""
    r = _run(["--config", "configs/lean-pdf-lss.yaml", "eval", "--help"])
    assert "hit_rate" in r.stdout
    assert "--k" in r.stdout
    assert "--dataset" in r.stdout


def test_unknown_command_fails() -> None:
    """Unknown commands exit non-zero with an error."""
    r = _run(["--config", "configs/lean-pdf-lss.yaml", "no-such-command"])
    assert r.returncode != 0
    assert "no-such-command" in r.stderr or "No such command" in r.stderr


def test_missing_config_exits_with_value_error() -> None:
    """A nonexistent config path exits non-zero with a clear error (no stack trace)."""
    r = _run(["--config", "/nonexistent/lean-does-not-exist.yaml", "eval", "--help"])
    assert r.returncode != 0
    assert "config file not found" in r.stderr or "No such" in r.stderr


def test_verbose_flag_accepted() -> None:
    """``--verbose`` is recognized at top level."""
    r = _run(["--config", "configs/lean-pdf-lss.yaml", "--verbose", "no-such"])
    assert "no-such" in r.stderr or r.returncode != 0
