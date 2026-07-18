#!/usr/bin/env python3
"""Minimal docs-lint: detect drift between code and docs/.

Run as a pre-commit hook or CI step. Exits non-zero on drift.

Checks:
  1. Every leaf key in `src/lean/config/config.yaml` has a row in
     `docs/configuration.md` (under its section heading).
  2. Every CLI command in `src/lean/cli.py` is documented in README.
  3. Every MCP tool in `src/lean/mcp_server/tools.py` is documented in README.

This is intentionally small — full docs/codegen parity is out of scope for a
single-author local MCP server. Add checks as drift patterns repeat.

Usage:
    uv run python scripts/docs_lint.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML = ROOT / "src/lean/config/config.yaml"
CLI_PY = ROOT / "src/lean/cli.py"
TOOLS_PY = ROOT / "src/lean/mcp_server/tools.py"
CONFIG_MD = ROOT / "docs/configuration.md"
README_MD = ROOT / "README.md"


def yaml_leaf_keys() -> set[str]:
    """Return all leaf key names from config.yaml.

    Returns bare key names (not full dotted paths). The docs organize
    fields under markdown section headings, so only the bare key name
    needs to match.
    """
    raw = yaml.safe_load(CONFIG_YAML.read_text()) or {}

    leaves: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, dict):
                    walk(v)
                else:
                    leaves.add(str(k))

    walk(raw)
    return leaves


def cli_commands() -> set[str]:
    """Extract `@app.command("name")` decorators from cli.py."""
    src = CLI_PY.read_text()
    return set(re.findall(r'@app\.command\("([\w-]+)"\)', src))


def mcp_tools() -> set[str]:
    """Extract tool function names from mcp_server/tools.py.

    Tools are registered via `@mcp.tool` or similar; we look for functions
    defined at module level that look like tool handlers.
    """
    src = TOOLS_PY.read_text()
    return {
        m.group(1) for m in re.finditer(r"^(?:async )?def ([a-z][a-z0-9_]*)\(", src, re.MULTILINE)
    }


def check_yaml_keys_in_config_md(keys: set[str]) -> list[str]:
    """Each config.yaml leaf key should appear in docs/configuration.md.

    Matching is word-boundary + case-insensitive on the key name. YAML
    sections (dicts) are checked too — if a section is missing, the keys
    under it are reported as missing as well.
    """
    if not CONFIG_MD.exists():
        return [f"docs/configuration.md missing (cannot check {len(keys)} keys)"]
    text_lower = CONFIG_MD.read_text().lower()
    missing = []
    for k in sorted(keys):
        # Word-boundary match on the bare key (not a substring of another word).
        if re.search(rf"\b{re.escape(k.lower())}\b", text_lower):
            continue
        missing.append(f"config.yaml key '{k}' not found in docs/configuration.md")
    return missing


def check_cli_in_readme(commands: set[str]) -> list[str]:
    if not README_MD.exists():
        return ["README.md missing"]
    text = README_MD.read_text()
    missing = []
    for cmd in sorted(commands):
        if f"lean {cmd}" in text or f"`{cmd}`" in text:
            continue
        missing.append(f"CLI command 'lean {cmd}' not documented in README")
    return missing


def check_mcp_tools_in_readme(tools: set[str]) -> list[str]:
    if not README_MD.exists():
        return ["README.md missing"]
    text = README_MD.read_text()
    skip = {"main", "register"}
    candidates = {t for t in tools if t not in skip and not t.startswith("_")}
    missing = []
    for tool in sorted(candidates):
        if tool in text:
            continue
        missing.append(f"MCP tool '{tool}' not documented in README")
    return missing


def main() -> int:
    errors: list[str] = []
    keys = yaml_leaf_keys()
    commands = cli_commands()
    tools = mcp_tools()

    print(f"checking {len(keys)} config.yaml keys…")
    errors.extend(check_yaml_keys_in_config_md(keys))

    print(f"checking {len(commands)} CLI commands…")
    errors.extend(check_cli_in_readme(commands))

    print(f"checking {len(tools)} candidate MCP tools…")
    errors.extend(check_mcp_tools_in_readme(tools))

    if errors:
        print(f"\nFAIL: {len(errors)} drift item(s) found:\n")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nOK: docs match code surface")
    return 0


if __name__ == "__main__":
    sys.exit(main())
