"""Unit tests for infrastructure/embedder.py and llm/base.py factory branches.

Verifies that ``get_embedder`` and ``get_llm`` select the correct implementation
class with the correct constructor arguments based on settings — not just that
they return non-None.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_get_embedder_remote_ollama(monkeypatch_env):
    """Remote URL set: RemoteOllamaEmbedder constructed with all 6 kwargs."""
    from lean.infrastructure.embedder import get_embedder

    get_embedder.cache_clear()
    with patch("lean.infrastructure.embedder.get_settings") as mock_settings:
        s = MagicMock()
        s.embedding_remote_url = "http://gpu:11434"
        s.embedding_remote_model = "lfm2.5-embed-32k"
        s.embedding_dim = 1024
        s.embedding_num_ctx = 32768
        s.embedding_http_timeout = 30.0
        s.embedding_max_retries = 3
        mock_settings.return_value = s
        with patch("lean.embeddings.remote_ollama.RemoteOllamaEmbedder") as mock_cls:
            mock_cls.return_value = MagicMock(dim=1024)
            get_embedder()

    mock_cls.assert_called_once_with(
        base_url="http://gpu:11434",
        model="lfm2.5-embed-32k",
        dim=1024,
        num_ctx=32768,
        timeout=30.0,
        max_retries=3,
    )
    get_embedder.cache_clear()


def test_get_embedder_local_cpu(monkeypatch_env):
    """When remote_url is empty, LiquidLMFEmbedder is constructed with all 5 kwargs."""
    from lean.infrastructure.embedder import get_embedder

    get_embedder.cache_clear()
    with patch("lean.infrastructure.embedder.get_settings") as mock_settings:
        s = MagicMock()
        s.embedding_remote_url = ""
        s.embedding_remote_model = ""
        s.embedding_model = "LiquidAI/LFM2.5-Embedding-350M"
        s.hf_token = "hf-token-123"
        s.embedding_device = "cpu"
        s.embedding_dim = 1024
        s.embedding_model_revision = "f35ae2c9"
        mock_settings.return_value = s
        with patch("lean.embeddings.liquid_lmf.LiquidLMFEmbedder") as mock_cls:
            mock_cls.return_value = MagicMock(dim=1024)
            get_embedder()

    mock_cls.assert_called_once_with(
        model="LiquidAI/LFM2.5-Embedding-350M",
        hf_token="hf-token-123",
        device="cpu",
        dim=1024,
        revision="f35ae2c9",
    )
    get_embedder.cache_clear()


def test_get_llm_minimax(monkeypatch_env):
    """When minimax_api_key is set, OpenAICompatibleLLM is constructed with MiniMax config."""
    from lean.llm.base import get_llm

    get_llm.cache_clear()
    with patch("lean.llm.base.get_settings") as mock_settings:
        s = MagicMock()
        s.minimax_api_key = "minimax-key-12345"
        s.llm_minimax_base_url = "https://api.minimax.io"
        s.llm_minimax_model = "MiniMax-Text-01"
        s.llm_timeout_s = 60.0
        s.llm_ollama_url = ""
        s.llm_ollama_model = ""
        mock_settings.return_value = s
        with patch("lean.llm.openai_compatible.OpenAICompatibleLLM") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_llm()

    mock_cls.assert_called_once_with(
        base_url="https://api.minimax.io",
        model="MiniMax-Text-01",
        api_key="minimax-key-12345",
        timeout=60.0,
    )
    get_llm.cache_clear()


def test_get_llm_ollama(monkeypatch_env):
    """Ollama URL set (no minimax key): OpenAICompatibleLLM with Ollama config."""
    from lean.llm.base import get_llm

    get_llm.cache_clear()
    with patch("lean.llm.base.get_settings") as mock_settings:
        s = MagicMock()
        s.minimax_api_key = ""
        s.llm_ollama_url = "http://gpu:11434"
        s.llm_ollama_model = "qwen3.5:9b"
        s.llm_timeout_s = 60.0
        mock_settings.return_value = s
        with patch("lean.llm.openai_compatible.OpenAICompatibleLLM") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_llm()

    mock_cls.assert_called_once_with(
        base_url="http://gpu:11434",
        model="qwen3.5:9b",
        api_key="",
        timeout=60.0,
    )
    get_llm.cache_clear()


def test_get_llm_none_when_unconfigured(monkeypatch_env):
    """When neither minimax nor ollama is configured, get_llm returns None."""
    from lean.llm.base import get_llm

    get_llm.cache_clear()
    with patch("lean.llm.base.get_settings") as mock_settings:
        s = MagicMock()
        s.minimax_api_key = ""
        s.llm_ollama_url = ""
        mock_settings.return_value = s
        result = get_llm()

    assert result is None
    get_llm.cache_clear()


def test_get_llm_minimax_takes_precedence_over_ollama(monkeypatch_env):
    """When both minimax and ollama are configured, MiniMax wins (checked first)."""
    from lean.llm.base import get_llm

    get_llm.cache_clear()
    with patch("lean.llm.base.get_settings") as mock_settings:
        s = MagicMock()
        s.minimax_api_key = "minimax-key"
        s.llm_minimax_base_url = "https://api.minimax.io"
        s.llm_minimax_model = "MiniMax-Text-01"
        s.llm_timeout_s = 60.0
        s.llm_ollama_url = "http://gpu:11434"
        s.llm_ollama_model = "qwen3.5:9b"
        mock_settings.return_value = s
        with patch("lean.llm.openai_compatible.OpenAICompatibleLLM") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_llm()

    # Should be called with MiniMax config, not Ollama
    mock_cls.assert_called_once_with(
        base_url="https://api.minimax.io",
        model="MiniMax-Text-01",
        api_key="minimax-key",
        timeout=60.0,
    )
    get_llm.cache_clear()
