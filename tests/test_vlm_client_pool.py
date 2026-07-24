"""Tests for VLM client per-process singleton caching (WP-8)."""

from __future__ import annotations

from unittest.mock import patch

from lean.core.transports.yaml_loader import _get_vlm_client


def test_only_one_vlm_client_created_for_n_calls() -> None:
    _get_vlm_client.cache_clear()
    with patch("lean.core.vlm.OpenAICompatibleVLM") as mock_cls:
        mock_cls.return_value = mock_cls
        _get_vlm_client("url", "model", "key", 30.0, "high", True)
        _get_vlm_client("url", "model", "key", 30.0, "high", True)
        _get_vlm_client("url", "model", "key", 30.0, "high", True)
        assert mock_cls.call_count == 1


def test_close_not_called_on_cached_client() -> None:
    _get_vlm_client.cache_clear()
    with patch("lean.core.vlm.OpenAICompatibleVLM") as mock_cls:
        client = mock_cls.return_value
        _get_vlm_client("url2", "model", "key", 30.0, "high", True)
        _get_vlm_client("url2", "model", "key", 30.0, "high", True)
        _get_vlm_client("url2", "model", "key", 30.0, "high", True)
        assert client.close.call_count == 0
