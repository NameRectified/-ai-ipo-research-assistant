"""Tests for the LLMClient — provider fallback and error handling."""

from typing import Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.services.llm_client import LLMClient


def _mock_response(status_code: int = 200, json_data: Optional[dict] = None) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {
        "choices": [{"message": {"content": "test response"}}]
    }
    resp.raise_for_status = MagicMock()
    if status_code >= 400 and status_code != 429:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    return resp


class TestLLMClientInit:
    """Tests for provider initialization."""

    @patch("app.services.llm_client.settings")
    def test_no_providers_configured(self, mock_settings: MagicMock) -> None:
        mock_settings.llm_provider_priority = "groq,gemini,openrouter"
        mock_settings.groq_api_key = ""
        mock_settings.gemini_api_key = ""
        mock_settings.openrouter_api_key = ""
        client = LLMClient()
        assert client.available is False

    @patch("app.services.llm_client.settings")
    def test_groq_provider_configured(self, mock_settings: MagicMock) -> None:
        mock_settings.llm_provider_priority = "groq"
        mock_settings.groq_api_key = "test-key"
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        client = LLMClient()
        assert client.available is True
        assert len(client._providers) == 1
        assert client._providers[0].name == "groq"

    @patch("app.services.llm_client.settings")
    def test_multiple_providers_priority_order(self, mock_settings: MagicMock) -> None:
        mock_settings.llm_provider_priority = "gemini,groq"
        mock_settings.groq_api_key = "groq-key"
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.gemini_api_key = "gemini-key"
        mock_settings.gemini_model = "gemini-2.5-flash"
        mock_settings.openrouter_api_key = ""
        mock_settings.openrouter_models = ""
        client = LLMClient()
        assert len(client._providers) == 2
        assert client._providers[0].name == "gemini"
        assert client._providers[1].name == "groq"


class TestLLMClientGenerate:
    """Tests for the generate method with provider fallback."""

    @patch("app.services.llm_client.settings")
    @patch("httpx.Client.post")
    def test_first_provider_succeeds(
        self, mock_post: MagicMock, mock_settings: MagicMock
    ) -> None:
        mock_settings.llm_provider_priority = "groq"
        mock_settings.groq_api_key = "test-key"
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_post.return_value = _mock_response(200)
        client = LLMClient()
        result = client.generate("system", "user")
        assert result == "test response"
        assert mock_post.call_count == 1

    @patch("app.services.llm_client.settings")
    @patch("httpx.Client.post")
    def test_fallback_to_second_provider(
        self, mock_post: MagicMock, mock_settings: MagicMock
    ) -> None:
        mock_settings.llm_provider_priority = "groq,gemini"
        mock_settings.groq_api_key = "groq-key"
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.gemini_api_key = "gemini-key"
        mock_settings.gemini_model = "gemini-2.5-flash"
        mock_settings.openrouter_api_key = ""
        mock_settings.openrouter_models = ""
        mock_post.side_effect = [
            _mock_response(500),
            _mock_response(200, {"candidates": [{"content": {"parts": [{"text": "gemini response"}]}}]}),
        ]
        client = LLMClient()
        result = client.generate("system", "user")
        assert result == "gemini response"
        assert mock_post.call_count == 2

    @patch("app.services.llm_client.settings")
    @patch("httpx.Client.post")
    def test_rate_limit_returns_none_fallback(
        self, mock_post: MagicMock, mock_settings: MagicMock
    ) -> None:
        mock_settings.llm_provider_priority = "groq,gemini"
        mock_settings.groq_api_key = "groq-key"
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.gemini_api_key = "gemini-key"
        mock_settings.gemini_model = "gemini-2.5-flash"
        mock_settings.openrouter_api_key = ""
        mock_settings.openrouter_models = ""
        mock_post.side_effect = [
            _mock_response(429),
            _mock_response(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}),
        ]
        client = LLMClient()
        result = client.generate("system", "user")
        assert result == "ok"

    @patch("app.services.llm_client.settings")
    @patch("httpx.Client.post")
    def test_all_providers_fail_raises(
        self, mock_post: MagicMock, mock_settings: MagicMock
    ) -> None:
        mock_settings.llm_provider_priority = "groq"
        mock_settings.groq_api_key = "groq-key"
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_post.return_value = _mock_response(500)
        client = LLMClient()
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            client.generate("system", "user")

    @patch("app.services.llm_client.settings")
    @patch("httpx.Client.post")
    def test_no_providers_raises(self, mock_post: MagicMock, mock_settings: MagicMock) -> None:
        mock_settings.llm_provider_priority = "groq"
        mock_settings.groq_api_key = ""
        mock_settings.gemini_api_key = ""
        mock_settings.openrouter_api_key = ""
        client = LLMClient()
        with pytest.raises(RuntimeError, match="No LLM providers configured"):
            client.generate("system", "user")
