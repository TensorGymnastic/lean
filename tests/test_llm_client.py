"""Tests for the OpenAI-compatible LLM client."""

from __future__ import annotations

import httpx
import respx


@respx.mock
def test_minimax_generate_success() -> None:
    """MiniMax-style API call returns the generated text."""
    respx.post("https://api.minimax.io/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "DMAIC is a methodology."}}],
            },
        )
    )

    from lean.llm.openai_compatible import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(
        base_url="https://api.minimax.io",
        model="MiniMax-Text-01",
        api_key="test-key",
    )
    result = llm.generate("What is DMAIC?")
    assert result == "DMAIC is a methodology."


@respx.mock
def test_ollama_generate_success() -> None:
    """Ollama OpenAI-compatible endpoint works without API key."""
    respx.post("http://gpu-host:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Context: chapter on quality."}}],
            },
        )
    )

    from lean.llm.openai_compatible import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(
        base_url="http://gpu-host:11434",
        model="qwen3.5:9b",
    )
    result = llm.generate("Generate context", system="You are a helpful assistant.")
    assert "quality" in result


@respx.mock
def test_generate_with_system_prompt() -> None:
    """System prompt is included in the request."""
    route = respx.post("https://api.test.io/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Response"}}]},
        )
    )

    from lean.llm.openai_compatible import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(base_url="https://api.test.io", model="test-model")
    llm.generate("Hello", system="You are a robot")

    request_body = route.calls[0].request.read().decode()
    assert "system" in request_body
    assert "robot" in request_body


@respx.mock
def test_generate_raises_on_error() -> None:
    """Non-200 raises HTTPStatusError."""
    respx.post("https://api.test.io/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    from lean.llm.openai_compatible import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(base_url="https://api.test.io", model="test-model")
    try:
        llm.generate("test")
        raise AssertionError("Should have raised")
    except httpx.HTTPStatusError:
        pass
