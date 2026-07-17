"""Unit tests for infrastructure/embedder.py and llm/base.py factory branches."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

_VALID_KEY = "x" * 32


@pytest.fixture
def monkeypatch_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)


def test_get_embedder_remote_ollama(monkeypatch_env):
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
            embedder = get_embedder()
    assert embedder is not None
    get_embedder.cache_clear()


def test_get_embedder_local_cpu(monkeypatch_env):
    from lean.infrastructure.embedder import get_embedder

    get_embedder.cache_clear()
    with patch("lean.infrastructure.embedder.get_settings") as mock_settings:
        s = MagicMock()
        s.embedding_remote_url = ""
        s.embedding_remote_model = ""
        s.embedding_model = "test-model"
        s.hf_token = ""
        s.embedding_device = "cpu"
        s.embedding_dim = 1024
        s.embedding_model_revision = "abc123"
        mock_settings.return_value = s
        with patch("lean.embeddings.liquid_lmf.LiquidLMFEmbedder") as mock_cls:
            mock_cls.return_value = MagicMock(dim=1024)
            embedder = get_embedder()
    assert embedder is not None
    get_embedder.cache_clear()


def test_get_llm_minimax(monkeypatch_env):
    from lean.llm.base import get_llm

    get_llm.cache_clear()
    with patch("lean.llm.base.get_settings") as mock_settings:
        s = MagicMock()
        s.minimax_api_key = "test-key"
        s.llm_minimax_base_url = "https://api.minimax.io"
        s.llm_minimax_model = "MiniMax-Text-01"
        s.llm_timeout_s = 60.0
        s.llm_ollama_url = ""
        s.llm_ollama_model = ""
        mock_settings.return_value = s
        with patch("lean.llm.openai_compatible.OpenAICompatibleLLM") as mock_cls:
            mock_cls.return_value = MagicMock()
            llm = get_llm()
    assert llm is not None
    get_llm.cache_clear()


def test_get_llm_ollama(monkeypatch_env):
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
            llm = get_llm()
    assert llm is not None
    get_llm.cache_clear()


def test_get_llm_none_when_unconfigured(monkeypatch_env):
    from lean.llm.base import get_llm

    get_llm.cache_clear()
    with patch("lean.llm.base.get_settings") as mock_settings:
        s = MagicMock()
        s.minimax_api_key = ""
        s.llm_ollama_url = ""
        mock_settings.return_value = s
        llm = get_llm()
    assert llm is None
    get_llm.cache_clear()
