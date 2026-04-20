from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar, Type
from urllib.parse import urlparse

from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import subprocess
import structlog

from codelyzer.config import Settings, settings

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


def _ollama_auth_client_kwargs(cfg: Settings) -> dict:
    """Bearer auth for Ollama Cloud (and compatible hosts). Uses settings field so .env works without relying on os.environ alone."""
    if not (cfg.ollama_api_key or "").strip():
        return {}
    return {"client_kwargs": {"headers": {"Authorization": f"Bearer {cfg.ollama_api_key.strip()}"}}}


def _is_ollama_cloud_api_host(url: str | None) -> bool:
    """True when OLLAMA_BASE_URL points at ollama.com (hosted API); skip local model discovery in that case."""
    if not (url or "").strip():
        return False
    try:
        u = urlparse(url.strip())
    except ValueError:
        return False
    if (u.scheme or "").lower() != "https":
        return False
    host = (u.hostname or "").lower()
    return host in ("ollama.com", "www.ollama.com")


class BaseLLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_structured(self, prompt: ChatPromptTemplate, input_vars: Dict[str, Any],
                           output_schema: Type[T]) -> T:
        """Generate structured output matching given Pydantic schema."""
        pass

    @abstractmethod
    def generate_text(self, prompt: ChatPromptTemplate, input_vars: Dict[str, Any]) -> str:
        """Generate plain text response."""
        pass


class OllamaClient(BaseLLMClient):
    def __init__(self):
        auth_kw = _ollama_auth_client_kwargs(settings)
        inferred_cloud = _is_ollama_cloud_api_host(settings.ollama_base_url) and not settings.ollama_use_cloud

        if settings.ollama_use_cloud:
            model_name = settings.ollama_cloud_model
            base_url = settings.ollama_cloud_base_url.rstrip("/")
            if not (settings.ollama_api_key or "").strip():
                logger.warning(
                    "Ollama Cloud is enabled but ollama_api_key is empty; set OLLAMA_API_KEY "
                    "or ollama_api_key in .env (https://ollama.com/settings/keys)"
                )
            self.model = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=0.1,
                **auth_kw,
            )
            logger.info(
                "Initialized Ollama Cloud LLM client",
                model=model_name,
                base_url=base_url,
            )
            return

        if inferred_cloud:
            model_name = settings.ollama_model
            base_url = settings.ollama_base_url.rstrip("/")
            if not (settings.ollama_api_key or "").strip():
                logger.warning(
                    "OLLAMA_BASE_URL points at ollama.com but ollama_api_key is empty; "
                    "set OLLAMA_API_KEY (https://ollama.com/settings/keys)"
                )
            self.model = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=0.1,
                **auth_kw,
            )
            logger.info(
                "Initialized Ollama Cloud LLM client (from OLLAMA_BASE_URL)",
                model=model_name,
                base_url=base_url,
            )
            return

        selected_model = self._select_available_model(settings.ollama_model)
        self.model = ChatOllama(
            model=selected_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
            **auth_kw,
        )
        logger.info("Initialized Ollama LLM client", model=selected_model, base_url=settings.ollama_base_url)

    def _list_local_models(self) -> list[str]:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return []

        if result.returncode != 0:
            return []

        models: list[str] = []
        for idx, line in enumerate(result.stdout.splitlines()):
            if idx == 0:
                continue
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models

    def _select_available_model(self, preferred_model: str) -> str:
        local_models = self._list_local_models()
        if not local_models:
            return preferred_model

        if preferred_model in local_models:
            return preferred_model

        non_embedding_models = [
            model for model in local_models
            if "embed" not in model.lower() and "embedding" not in model.lower()
        ]
        fallback = non_embedding_models[0] if non_embedding_models else local_models[0]
        logger.warning(
            "Configured Ollama model not found locally, falling back to installed model",
            preferred_model=preferred_model,
            fallback_model=fallback,
        )
        return fallback

    def generate_structured(self, prompt: ChatPromptTemplate, input_vars: Dict[str, Any],
                           output_schema: Type[T]) -> T:
        chain = prompt | self.model.with_structured_output(output_schema)
        return chain.invoke(input_vars)

    def generate_text(self, prompt: ChatPromptTemplate, input_vars: Dict[str, Any]) -> str:
        chain = prompt | self.model
        result = chain.invoke(input_vars)
        return result.content


def get_llm_client() -> BaseLLMClient:
    """Factory function to get configured LLM client."""
    return OllamaClient()
