"""Tests for VLM client per-process singleton caching (WP-8)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from lean.core.transports.yaml_loader import _build_vlm_hooks, _get_vlm_client


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


def test_vlm_describe_one_is_coroutine() -> None:
    """_build_vlm_hooks must produce an async describe_one (describe_images awaits it)."""
    cfg = MagicMock()
    cfg.vlm.enabled = True
    cfg.vlm.parser = None
    cfg.vlm.image_heading_format = "Image {n}"
    cfg.vlm.prompt_inline = "describe this"
    cfg.vlm.prompt_file = None

    with patch("lean.core.transports.yaml_loader._resolve_prompt", return_value="prompt"):
        hooks = _build_vlm_hooks(cfg)

    assert hooks.describe_one is not None
    assert inspect.iscoroutinefunction(hooks.describe_one)
