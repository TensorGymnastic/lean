"""Tests for the remote Ollama embedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from lean.embeddings.remote_ollama import RemoteOllamaEmbedder


@patch("lean.embeddings.remote_ollama.httpx.Client")
def test_embed_query_calls_ollama_api(MockClient: MagicMock) -> None:
    mock_client = MockClient.return_value
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_resp.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_resp

    embedder = RemoteOllamaEmbedder(
        base_url="http://test:11434",
        model="test-model",
        dim=3,
    )
    result = embedder.embed_query("hello")

    assert result == [0.1, 0.2, 0.3]
    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args
    assert args[0] == "http://test:11434/api/embeddings"
    assert kwargs["json"]["model"] == "test-model"
    assert kwargs["json"]["prompt"] == "hello"


@patch("lean.embeddings.remote_ollama.httpx.Client")
def test_embed_documents_parallel(MockClient: MagicMock) -> None:
    mock_client = MockClient.return_value
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embedding": [0.5]}
    mock_resp.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_resp

    embedder = RemoteOllamaEmbedder(
        base_url="http://test:11434",
        model="test-model",
        dim=1,
    )
    results = embedder.embed_documents(["a", "b", "c"])

    assert len(results) == 3
    assert all(r == [0.5] for r in results)
    assert mock_client.post.call_count == 3


def test_dim_property() -> None:
    with patch("lean.embeddings.remote_ollama.httpx.Client"):
        embedder = RemoteOllamaEmbedder(
            base_url="http://test:11434",
            model="m",
            dim=768,
        )
        assert embedder.dim == 768


@patch("lean.embeddings.remote_ollama.httpx.Client")
def test_empty_documents_returns_empty(MockClient: MagicMock) -> None:
    with patch("lean.embeddings.remote_ollama.httpx.Client"):
        embedder = RemoteOllamaEmbedder(
            base_url="http://test:11434",
            model="m",
            dim=10,
        )
        assert embedder.embed_documents([]) == []
