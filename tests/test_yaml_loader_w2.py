"""Coverage tests for the YAML loader + domain_config pipeline.

Complements ``tests/test_yaml_loader_w1.py`` (security + error paths) by
exercising the happy paths and the full ``build_from_yaml`` machinery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lean.core.config.domain_config import (
    DomainConfig,
    DomainMetadata,
    ExtractorRef,
    MetadataConfig,
    ToolsConfig,
    VLMConfig,
)
from lean.core.config.settings import _apply_settings_overrides
from lean.core.transports.yaml_loader import (
    ALLOWED_ADAPTER_PREFIXES,
    _import_object,
)

# ---------- DomainConfig ----------


def test_domain_metadata_defaults_version_to_0_1_0() -> None:
    """DomainMetadata.version defaults to ``0.1.0``."""
    m = DomainMetadata(name="x")
    assert m.version == "0.1.0"
    assert m.description == ""


def test_extractor_ref_config_defaults_to_empty_dict() -> None:
    """ExtractorRef.config defaults to an empty dict (kwargs passed to constructor)."""
    r = ExtractorRef(adapter="a.b.C")
    assert r.config == {}


def test_vlm_config_disabled_by_default() -> None:
    """VLMConfig.enabled defaults to False."""
    v = VLMConfig()
    assert v.enabled is False
    assert v.image_heading_format == "Image {n}"


def test_tools_config_disabled_defaults_to_empty() -> None:
    """ToolsConfig.disabled defaults to an empty list."""
    t = ToolsConfig(module=None)
    assert t.enabled == []
    assert t.disabled == []


def test_metadata_config_extractor_defaults_to_none() -> None:
    """MetadataConfig.extractor defaults to None (no custom PDF metadata extractor)."""
    m = MetadataConfig()
    assert m.extractor is None


def test_domain_config_empty_extractors_raises() -> None:
    """A YAML with zero extractors raises a validation error."""
    with pytest.raises(ValueError, match="at least one extractor"):
        DomainConfig(
            domain=DomainMetadata(name="x"),
            extractors=[],
        )


def test_domain_config_from_yaml_extras_go_to_settings(tmp_path: Path) -> None:
    """Unknown top-level YAML keys are forwarded to ``settings``."""
    yaml = tmp_path / "x.yaml"
    yaml.write_text(
        "domain: {name: x, version: 0.1.0}\n"
        "extractors:\n"
        "  - adapter: lean.core.extraction.markitdown.MarkitdownExtractor\n"
        "tools: {module: x, enabled: [], disabled: []}\n"
        "embedding:\n"  # not a DomainConfig field
        "  model: my-model\n"
    )
    cfg = DomainConfig.from_yaml(yaml)
    assert "embedding" in cfg.settings
    assert cfg.settings["embedding"]["model"] == "my-model"


# ---------- Settings overlay ----------


def test_apply_settings_overrides_flat_key() -> None:
    """A flat YAML key that matches a CoreSettings field is set on the instance."""
    settings = _make_settings()
    _apply_settings_overrides(settings, {"corpus_root": "/nonexistent/data"})
    assert settings.corpus_root == "/nonexistent/data"


def test_apply_settings_overrides_composite_key() -> None:
    """A nested YAML key becomes ``f"{section}_{key}"`` if that field exists."""
    settings = _make_settings()
    _apply_settings_overrides(settings, {"embedding": {"model": "my-embedder"}})
    assert settings.embedding_model == "my-embedder"


def test_apply_settings_overrides_unknown_bucket() -> None:
    """A nested YAML key whose leaf matches a Settings field flows through directly.

    After C-1, ``ocr.dpi`` maps to ``settings.ocr_dpi`` (composite).
    Truly unknown leaves go into ``settings.domain_config``.
    """
    settings = _make_settings()
    _apply_settings_overrides(settings, {"ocr": {"dpi": 600}})
    assert settings.ocr_dpi == 600


def test_apply_settings_overrides_does_not_clobber_existing_domain_config() -> None:
    """Truly unknown leaves land under ``domain_config`` without clobbering siblings."""
    settings = _make_settings()
    settings.domain_config["my_custom"] = {"keep": True}
    _apply_settings_overrides(settings, {"my_custom": {"add": "yes"}})
    assert settings.domain_config["my_custom"]["keep"] is True
    assert settings.domain_config["my_custom"]["add"] == "yes"


def _make_settings():
    """Use a real CoreSettings — _apply_settings_overlays touches actual fields."""
    import os

    os.environ.setdefault("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    os.environ.setdefault("HF_TOKEN", "test")
    os.environ.setdefault("LEAN_MCP_API_KEY", "x" * 32)
    os.environ.setdefault("MCP_HTTP_PORT", "8765")
    os.environ.setdefault("API_PORT", "8766")
    from lean.core.config.settings import CoreSettings, clear_settings_cache

    clear_settings_cache()
    s = CoreSettings()
    return s


# ---------- _import_object ----------


def test_allowed_adapter_prefixes_includes_both_families() -> None:
    """Both ``lean.core.`` and ``lean.domains.`` are allowed."""
    assert any(p.startswith("lean.core.") for p in ALLOWED_ADAPTER_PREFIXES)
    assert any(p.startswith("lean.domains.") for p in ALLOWED_ADAPTER_PREFIXES)


def test_import_object_with_colon_syntax() -> None:
    """``module:attr`` syntax resolves the attribute on the module."""
    obj = _import_object("lean.core.adapters:mcp_tool")
    from lean.core.adapters import mcp_tool

    assert obj is mcp_tool


def test_import_object_rejects_non_lean_paths() -> None:
    """Paths outside the whitelist raise ValueError (RCE protection)."""
    for path in ["os.system", "subprocess.run", "shutil.rmtree", "pickle.loads"]:
        with pytest.raises(ValueError, match=r"not in allowed prefixes"):
            _import_object(path)


def test_import_object_rejects_path_with_dots_in_attribute() -> None:
    """An empty attribute part is rejected."""
    with pytest.raises(ValueError, match=r"invalid dotted path"):
        _import_object("lean.core.adapters.")
