"""E2E CLI smoke tests against the installed ``lean`` binary.

Marked @pytest.mark.e2e — requires:
- Local Supabase running (for any DB-touching command)
- The ``lean`` console-script available in PATH or the project venv

These tests run the actual CLI binary (not the Python module directly) so
they catch packaging issues, entry-point regressions, and CLI-only bugs
that the in-process ``CliRunner`` cannot.

Run with:
    SUPABASE_DB_URL=postgresql://postgres:postgres@localhost:54322/postgres \\
        RUN_E2E=1 uv run pytest tests/test_e2e_cli.py -v -m e2e
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("SUPABASE_DB_URL"),
        reason="SUPABASE_DB_URL not set — start Supabase first (make db-init)",
    ),
]


def _lean_binary() -> str:
    """Return path to the lean console-script binary (project venv)."""
    venv_bin = Path(__file__).parent.parent / ".venv" / "bin" / "lean"
    if venv_bin.is_file():
        return str(venv_bin)
    system_lean = shutil.which("lean")
    if system_lean:
        return system_lean
    pytest.skip("lean binary not found (install with: uv sync)")


def _lean_env() -> dict[str, str]:
    """Build env for lean subprocess invocations."""
    env = os.environ.copy()
    env.setdefault("HF_TOKEN", "test-token")
    env.setdefault("LEAN_MCP_API_KEY", "x" * 32)
    env.setdefault("LEAN_CONFIG", "configs/lean-pdf-lss.yaml")
    env.setdefault("MCP_HTTP_PORT", "8767")
    env.setdefault("API_PORT", "8766")
    env["SUPABASE_DB_URL"] = os.environ["SUPABASE_DB_URL"]
    env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")
    return env


def test_cli_help_exits_zero() -> None:
    """`lean --help` exits 0 and shows Usage + Commands."""
    result = subprocess.run(
        [_lean_binary(), "--help"],
        env=_lean_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "Usage" in combined
    assert "Commands" in combined


def test_cli_command_help_for_search() -> None:
    """`lean search --help` shows --json, --k, --chunk-type options."""
    result = subprocess.run(
        [_lean_binary(), "search", "--help"],
        env=_lean_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "--json" in combined
    assert "--k" in combined
    assert "--chunk-type" in combined


def test_cli_command_help_for_ingest() -> None:
    """`lean ingest --help` shows path argument and --json flag."""
    result = subprocess.run(
        [_lean_binary(), "ingest", "--help"],
        env=_lean_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "PATH" in combined or "path" in combined
    assert "--json" in combined


def test_cli_command_help_for_reingest_all() -> None:
    """`lean reingest-all --help` shows --force flag."""
    result = subprocess.run(
        [_lean_binary(), "reingest-all", "--help"],
        env=_lean_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "--force" in combined


def test_cli_command_help_for_eval() -> None:
    """`lean eval --help` shows --sample-size, --k, --dataset flags."""
    result = subprocess.run(
        [_lean_binary(), "eval", "--help"],
        env=_lean_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "--sample-size" in combined
    assert "--k" in combined
    assert "--dataset" in combined


def test_cli_list_documents_json_outputs_valid_json() -> None:
    """`lean list-documents --json` returns parseable JSON array."""
    result = subprocess.run(
        [_lean_binary(), "list-documents", "--json"],
        env=_lean_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list)
    if parsed:
        first = parsed[0]
        assert "id" in first
        assert "chunk_count" in first
        assert "extraction_method" in first


def test_cli_corpus_stats_json_outputs_valid_json() -> None:
    """`lean corpus-stats --json` returns a JSON object with expected keys."""
    result = subprocess.run(
        [_lean_binary(), "corpus-stats", "--json"],
        env=_lean_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    parsed = json.loads(result.stdout)
    assert "document_count" in parsed
    assert "chunk_count" in parsed
    assert "total_tokens" in parsed
    assert "embedding_dim" in parsed


def test_cli_search_no_results() -> None:
    """A nonsense query returns exit 0 with empty results."""
    result = subprocess.run(
        [_lean_binary(), "search", "xqzklmnopqrstuvwxyz-no-match-anywhere-12345", "--k", "3"],
        env=_lean_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0


def test_cli_invalid_chunk_type_fails() -> None:
    """`lean search --chunk-type=foo` returns a ValueError-mapped failure."""
    result = subprocess.run(
        [_lean_binary(), "search", "DMAIC", "--chunk-type", "bogus"],
        env=_lean_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "chunk_type" in combined


def test_cli_version_imports() -> None:
    """`python -c "import lean.cli"` succeeds — module is importable."""
    result = subprocess.run(
        [sys.executable, "-c", "import lean.cli; print('ok')"],
        env=_lean_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "ok" in result.stdout


def test_cli_search_via_json_option() -> None:
    """`lean search ... --json` returns valid JSON array."""
    result = subprocess.run(
        [_lean_binary(), "search", "DMAIC", "--k", "3", "--json"],
        env=_lean_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list)
