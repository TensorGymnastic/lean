"""Tests for VLM hook construction."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from lean.core.transports.yaml_loader import _build_vlm_hooks


def test_vlm_describe_one_is_coroutine() -> None:
    """_build_vlm_hooks must produce an async describe_one (describe_images awaits it)."""
    cfg = MagicMock()
    cfg.vlm.enabled = True
    cfg.vlm.parser = None
    cfg.vlm.image_heading_format = "Image {n}"
    cfg.vlm.prompt_inline = "describe this"
    cfg.vlm.prompt_file = None

    settings = MagicMock()
    settings.vlm_base_url = "http://localhost:11434"
    settings.vlm_model = "test-model"
    settings.vlm_api_key = ""
    settings.vlm_timeout_s = 30.0
    settings.vlm_detail = "high"
    settings.vlm_disable_thinking = True
    settings.vlm_max_tokens = 1000

    with (
        patch("lean.core.transports.yaml_loader._resolve_prompt", return_value="prompt"),
        patch("lean.core.vlm.OpenAICompatibleVLM"),
    ):
        hooks = _build_vlm_hooks(cfg, settings)

    assert hooks.describe_one is not None
    assert inspect.iscoroutinefunction(hooks.describe_one)
