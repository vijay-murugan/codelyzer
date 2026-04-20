"""Tests for Ollama LLM client configuration helpers."""

from unittest.mock import MagicMock, patch

from codelyzer.config import Settings
from codelyzer.llm.client import OllamaClient, _is_ollama_cloud_api_host, _ollama_auth_client_kwargs


def test_ollama_auth_client_kwargs_empty() -> None:
    cfg = Settings(ollama_api_key="")
    assert _ollama_auth_client_kwargs(cfg) == {}


def test_is_ollama_cloud_api_host() -> None:
    assert _is_ollama_cloud_api_host("https://ollama.com") is True
    assert _is_ollama_cloud_api_host("https://ollama.com/") is True
    assert _is_ollama_cloud_api_host("http://localhost:11434") is False
    assert _is_ollama_cloud_api_host("") is False


def test_ollama_auth_client_kwargs_bearer() -> None:
    cfg = Settings(ollama_api_key="  secret  ")
    assert _ollama_auth_client_kwargs(cfg) == {
        "client_kwargs": {"headers": {"Authorization": "Bearer secret"}}
    }


def test_ollama_client_cloud_uses_cloud_model_and_url() -> None:
    cfg = Settings(
        ollama_use_cloud=True,
        ollama_cloud_base_url="https://ollama.com",
        ollama_cloud_model="gpt-oss:120b",
        ollama_api_key="k",
        ollama_model="ignored-local",
    )
    with patch("codelyzer.llm.client.settings", cfg):
        with patch("codelyzer.llm.client.ChatOllama") as mock_chat:
            mock_chat.return_value = MagicMock()
            OllamaClient()
    mock_chat.assert_called_once()
    call_kw = mock_chat.call_args.kwargs
    assert call_kw["model"] == "gpt-oss:120b"
    assert call_kw["base_url"] == "https://ollama.com"
    assert call_kw["client_kwargs"]["headers"]["Authorization"] == "Bearer k"


def test_ollama_client_inferred_cloud_from_ollama_base_url() -> None:
    cfg = Settings(
        ollama_use_cloud=False,
        ollama_base_url="https://ollama.com",
        ollama_model="gpt-oss:120b",
        ollama_api_key="k",
    )
    with patch("codelyzer.llm.client.settings", cfg):
        with patch("codelyzer.llm.client.ChatOllama") as mock_chat:
            mock_chat.return_value = MagicMock()
            OllamaClient()
    mock_chat.assert_called_once()
    call_kw = mock_chat.call_args.kwargs
    assert call_kw["model"] == "gpt-oss:120b"
    assert call_kw["base_url"] == "https://ollama.com"
    assert call_kw["client_kwargs"]["headers"]["Authorization"] == "Bearer k"


def test_ollama_client_local_uses_base_url_and_select_model() -> None:
    cfg = Settings(
        ollama_use_cloud=False,
        ollama_base_url="http://localhost:11434",
        ollama_model="mymodel",
        ollama_api_key="",
    )
    with patch("codelyzer.llm.client.settings", cfg):
        with patch("codelyzer.llm.client.ChatOllama") as mock_chat:
            mock_chat.return_value = MagicMock()
            with patch.object(OllamaClient, "_select_available_model", return_value="resolved"):
                OllamaClient()
    mock_chat.assert_called_once()
    call_kw = mock_chat.call_args.kwargs
    assert call_kw["model"] == "resolved"
    assert call_kw["base_url"] == "http://localhost:11434"
    assert "client_kwargs" not in call_kw
