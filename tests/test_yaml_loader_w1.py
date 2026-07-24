"""RED tests for S4 (eval CLI/MCP/REST command) and S8/S9 (YAML loader errors + whitelist).

These tests must FAIL on current code, then pass after the fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lean.core.config.domain_config import DomainConfig
from lean.core.transports.yaml_loader import (
    ALLOWED_ADAPTER_PREFIXES,
    _import_object,
    build_from_yaml,
)

# ---------- S4: eval command exists ----------


def test_eval_command_registered_for_pdf_lss() -> None:
    """The pdf_lss config must expose an `eval` CLI command (regression: pre-unification had it)."""
    import subprocess

    result = subprocess.run(
        ["uv", "run", "lean", "--config", "configs/lean-pdf-lss.yaml", "eval", "--help"],
        capture_output=True,
        text=True,
    )
    assert "hit_rate" in result.stdout, (
        f"pdf_lss eval --help should mention hit_rate. "
        f"stdout={result.stdout[:500]} stderr={result.stderr[:500]}"
    )


# ---------- S8: YAML loader error handling ----------


def test_malformed_yaml_raises_value_error(tmp_path: Path) -> None:
    """Malformed YAML must produce a ValueError, not a raw yaml.ScannerError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: valid: yaml: :::")
    with pytest.raises(ValueError, match=r"invalid YAML"):
        DomainConfig.from_yaml(bad)


def test_non_mapping_yaml_raises_value_error(tmp_path: Path) -> None:
    """A YAML scalar at the top level must produce a ValueError, not silently be {}."""
    bad = tmp_path / "scalar.yaml"
    bad.write_text("just a string\n")
    with pytest.raises(ValueError, match=r"domain YAML must be a mapping"):
        DomainConfig.from_yaml(bad)


def test_missing_extractor_module_raises_value_error(tmp_path: Path) -> None:
    """A nonexistent adapter module must produce a ValueError, not ModuleNotFoundError."""
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "domain: {name: x, version: 0.1.0}\n"
        "settings:\n"
        "  transport: {mcp_http_port: 8765, api_port: 8766}\n"
        "extractors:\n"
        "  - adapter: does.not.exist.MyExtractor\n"
        "    config: {}\n"
        "tools: {module: x}\n"
    )
    with pytest.raises(ValueError, match=r"(not in allowed prefixes|could not import adapter)"):
        build_from_yaml(cfg)


# ---------- S9: adapter path validation ----------


def test_import_object_rejects_subprocess() -> None:
    """An attacker-controlled YAML must not be able to import subprocess."""
    with pytest.raises(ValueError, match=r"not in allowed prefixes"):
        _import_object("subprocess.check_output")


def test_import_object_rejects_os() -> None:
    with pytest.raises(ValueError, match=r"not in allowed prefixes"):
        _import_object("os.system")


def test_import_object_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match=r"not in allowed prefixes"):
        _import_object("../../etc/passwd")


def test_import_object_allows_lean_core() -> None:
    """A legitimate lean.core path must still resolve."""
    obj = _import_object("lean.core.adapters.mcp_tool")
    assert obj is not None


def test_import_object_allows_lean_domains() -> None:
    """A legitimate lean.domains path must still resolve."""
    obj = _import_object("lean.domains.pdf_lss.adapters.MarkitdownAdapter")
    assert obj is not None


def test_allowed_adapter_prefixes_is_a_tuple() -> None:
    """Sanity: the whitelist must be a tuple of strings and include both prefix families."""
    assert isinstance(ALLOWED_ADAPTER_PREFIXES, tuple)
    assert all(isinstance(p, str) for p in ALLOWED_ADAPTER_PREFIXES)
    assert any(p.startswith("lean.core.") for p in ALLOWED_ADAPTER_PREFIXES)
    assert any(p.startswith("lean.domains.") for p in ALLOWED_ADAPTER_PREFIXES)
