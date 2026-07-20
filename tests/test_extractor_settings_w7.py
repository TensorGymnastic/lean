"""Marker/OCR settings flow from YAML's ``settings.marker.*`` / ``settings.ocr.*``
to the extractor constructor — the settings block is the single source of truth.

C-1 regression: previously the ``settings.marker.remote_url`` declaration was
silently dropped (no CoreSettings field matched) and the extractor config block
held an empty string — so MarkerExtractor always ran on local CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _default_transport_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_HTTP_PORT", "8765")
    monkeypatch.setenv("API_PORT", "8766")


def _build() -> object:
    from lean.core.transports.yaml_loader import build_from_yaml

    return build_from_yaml(Path("configs/lean-pdf-lss.yaml"))


def test_marker_remote_url_from_settings_reaches_extractor() -> None:
    """``settings.marker.remote_url`` is the source of truth — the extractor picks it up."""
    from lean.core.extraction.base import get_pipeline

    _build()
    pipeline = get_pipeline()
    marker = pipeline._extractors[0]
    assert marker._remote_url == "http://192.168.2.37:8000"


def test_marker_force_ocr_default_from_settings() -> None:
    """``settings.marker.force_ocr`` flows through even when the extractor config omits it."""
    from lean.core.extraction.base import get_pipeline

    _build()
    pipeline = get_pipeline()
    marker = pipeline._extractors[0]
    assert marker._force_ocr is False


def test_ocr_settings_from_settings_block() -> None:
    """``settings.ocr.model`` / ``settings.ocr.dpi`` flow into the OCR adapter."""
    from lean.core.extraction.base import get_pipeline

    _build()
    pipeline = get_pipeline()
    ocr = pipeline._extractors[1]
    assert ocr._model == "baidu/Unlimited-OCR"
    assert ocr._dpi == 300


def test_ocr_defaults_when_settings_unset() -> None:
    """Empty settings fall back to CoreSettings Python defaults."""
    import os

    os.environ.pop("OCR_BASE_URL", None)
    from lean.core.config.settings import (
        clear_settings_cache,
        get_settings,
    )

    clear_settings_cache()
    s = get_settings()
    assert s.ocr_model == "baidu/Unlimited-OCR"
    assert s.ocr_dpi == 300
    assert s.marker_remote_url == ""


def test_ocr_dpi_validator_rejects_out_of_range() -> None:
    """``ocr_dpi`` outside [50, 1200] is rejected at construction time."""
    from lean.core.config.settings import CoreSettings

    with pytest.raises(Exception, match="ocr_dpi"):
        CoreSettings(
            mcp_http_port=8765,
            api_port=8766,
            ocr_dpi=10,
        )


def test_ocr_timeout_validator_rejects_non_positive() -> None:
    """``ocr_timeout_s <= 0`` is rejected."""
    from lean.core.config.settings import CoreSettings

    with pytest.raises(Exception, match="ocr_timeout_s"):
        CoreSettings(
            mcp_http_port=8765,
            api_port=8766,
            ocr_timeout_s=0,
        )
