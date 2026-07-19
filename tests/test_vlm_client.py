"""Tests for the VLM (vision-language model) client."""

from __future__ import annotations

import base64
import io
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from PIL import Image

from lean.vlm.client import VLMClient, VLMError
from lean.vlm.prompts import CHART_EXTRACTION_PROMPT, parse_description


def _make_image(width: int = 100, height: int = 100) -> Image.Image:
    return Image.new("RGB", (width, height), color="white")


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    resp.raise_for_status = MagicMock()
    return resp


class TestVLMClientInit:
    def test_creates_client_with_api_key(self):
        client = VLMClient(
            base_url="http://gpu:11434/v1",
            model="gemma3:27b",
            api_key="test-key",
        )
        assert client._model == "gemma3:27b"
        assert "Authorization" in client._headers

    def test_creates_client_without_api_key(self):
        client = VLMClient(
            base_url="http://gpu:11434/v1",
            model="gemma3:27b",
        )
        assert "Authorization" not in client._headers

    def test_strips_trailing_slash_from_base_url(self):
        client = VLMClient(base_url="http://gpu:11434/v1/", model="m")
        assert client._base_url == "http://gpu:11434/v1"


class TestDescribeImage:
    def test_sends_image_and_returns_text(self):
        client = VLMClient(base_url="http://gpu:11434/v1", model="gemma3:27b")
        mock_resp = _mock_response("A bar chart showing Q1-Q4 revenue.")
        with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
            result = client.describe_image(_make_image(), prompt="Describe this.")
        assert "bar chart" in result
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "gemma3:27b"
        content = body["messages"][-1]["content"]
        assert any(p["type"] == "image_url" for p in content)
        assert any(p["type"] == "text" for p in content)

    def test_image_is_base64_png_data_uri(self):
        client = VLMClient(base_url="http://gpu:11434/v1", model="m")
        mock_resp = _mock_response("desc")
        with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
            client.describe_image(_make_image(50, 50), prompt="x")
        body = mock_post.call_args.kwargs["json"]
        image_part = next(p for p in body["messages"][-1]["content"] if p["type"] == "image_url")
        url = image_part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        b64_data = url.split(",", 1)[1]
        decoded = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (50, 50)

    def test_includes_detail_in_image_url(self):
        client = VLMClient(base_url="http://gpu:11434/v1", model="m", detail="high")
        mock_resp = _mock_response("desc")
        with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
            client.describe_image(_make_image(), prompt="x")
        body = mock_post.call_args.kwargs["json"]
        image_part = next(p for p in body["messages"][-1]["content"] if p["type"] == "image_url")
        assert image_part["image_url"]["detail"] == "high"

    def test_uses_custom_max_tokens(self):
        client = VLMClient(base_url="http://gpu:11434/v1", model="m")
        mock_resp = _mock_response("desc")
        with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
            client.describe_image(_make_image(), prompt="x", max_tokens=500)
        body = mock_post.call_args.kwargs["json"]
        assert body["max_tokens"] == 500

    def test_disable_thinking_adds_field(self):
        client = VLMClient(
            base_url="http://gpu:11434/v1", model="MiniMax-M3", disable_thinking=True
        )
        mock_resp = _mock_response("desc")
        with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
            client.describe_image(_make_image(), prompt="x")
        body = mock_post.call_args.kwargs["json"]
        assert body["thinking"] == {"type": "disabled"}

    def test_thinking_not_in_body_by_default(self):
        client = VLMClient(base_url="http://gpu:11434/v1", model="m")
        mock_resp = _mock_response("desc")
        with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
            client.describe_image(_make_image(), prompt="x")
        body = mock_post.call_args.kwargs["json"]
        assert "thinking" not in body

    def test_raises_on_http_error(self):
        client = VLMClient(base_url="http://gpu:11434/v1", model="m")
        with (
            patch.object(client._client, "post", side_effect=httpx.ConnectError("refused")),
            pytest.raises(VLMError, match="VLM request failed"),
        ):
            client.describe_image(_make_image(), prompt="x")

    def test_raises_on_empty_content(self):
        client = VLMClient(base_url="http://gpu:11434/v1", model="m")
        mock_resp = _mock_response("")
        with (
            patch.object(client._client, "post", return_value=mock_resp),
            pytest.raises(VLMError, match="empty response"),
        ):
            client.describe_image(_make_image(), prompt="x")

    def test_raises_on_missing_choices(self):
        client = VLMClient(base_url="http://gpu:11434/v1", model="m")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"error": "model not found"}
        mock_resp.raise_for_status = MagicMock()
        with (
            patch.object(client._client, "post", return_value=mock_resp),
            pytest.raises(VLMError, match="unexpected response"),
        ):
            client.describe_image(_make_image(), prompt="x")

    def test_posts_to_v1_chat_completions(self):
        """The VLM client must POST to ``/v1/chat/completions`` to match the
        OCR client and the OpenAI-compatible LLM client. Previously the path
        was ``/chat/completions`` (no /v1 prefix), which 404s on providers
        like MiniMax unless the base_url already includes ``/v1`` (audit
        finding 4.3 in docs/audit-architecture-2026-07-19.md).
        """
        client = VLMClient(base_url="http://gpu:11434", model="m")
        mock_resp = _mock_response("desc")
        with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
            client.describe_image(_make_image(), prompt="x")
        url = mock_post.call_args.args[0]
        assert url == "http://gpu:11434/v1/chat/completions"


class TestChartExtractionPrompt:
    def test_prompt_contains_json_instruction(self):
        assert "JSON" in CHART_EXTRACTION_PROMPT.upper()

    def test_prompt_lists_required_fields(self):
        for field in ("chart_type", "description", "key_data_points"):
            assert field in CHART_EXTRACTION_PROMPT


class TestParseDescription:
    def test_parses_valid_json(self):
        raw = json.dumps(
            {
                "chart_type": "bar",
                "description": "Revenue by quarter",
                "key_data_points": ["Q1: $1M"],
            }
        )
        result = parse_description(raw)
        assert result["chart_type"] == "bar"
        assert result["description"] == "Revenue by quarter"

    def test_extracts_json_from_markdown_fence(self):
        raw = '```json\n{"chart_type": "line", "description": "Trend"}\n```'
        result = parse_description(raw)
        assert result["chart_type"] == "line"

    def test_returns_raw_text_if_not_json(self):
        raw = "This is a bar chart showing revenue growth."
        result = parse_description(raw)
        assert result["chart_type"] == "unknown"
        assert result["description"] == raw

    def test_handles_partial_json(self):
        raw = '{"chart_type": "pie"'
        result = parse_description(raw)
        assert result["chart_type"] == "unknown"
        assert raw in result["description"]
