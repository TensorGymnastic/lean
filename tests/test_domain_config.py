"""Tests for domain config schema contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lean.core.config.domain_config import ToolsConfig


def test_tools_config_rejects_enabled_and_disabled() -> None:
    with pytest.raises(ValidationError):
        ToolsConfig(module="x", enabled=["a"], disabled=["b"])


def test_tools_config_accepts_module_only() -> None:
    tc = ToolsConfig(module="lean.domains.pdf_lss.tools")
    assert tc.module == "lean.domains.pdf_lss.tools"
