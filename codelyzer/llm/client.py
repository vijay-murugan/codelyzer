from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Type
from pydantic import BaseModel
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import subprocess
import structlog

from codelyzer.config import settings

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


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
        # If an API key is configured we assume the cloud/remote Ollama service
        # should be used and therefore prefer the configured model without
        # probing local `ollama` binaries. Otherwise, fall back to checking
        # locally-installed models.
        if getattr(settings, "ollama_api_key", None):

            selected_model = settings.ollama_model
            print(f"Selected Ollama model: {selected_model}")
        else:
            selected_model = self._select_available_model(settings.ollama_model)
        
        client_kwargs = {}
        api_key = getattr(settings, "ollama_api_key", None)
        # print(f"Using Ollama API key: {api_key}")
        if api_key:
            # Pass API key as a Bearer token header to the underlying httpx client.
            client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}

        self.llm = ChatOpenAI(
            model=settings.ollama_model,
            api_key=settings.ollama_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0
        )

        print(f"Initialized Ollama LLM client with model '{selected_model}' at '{settings.ollama_base_url}'")
        logger.info(
            "Initialized Ollama LLM client",
            base_url=self.llm.base_url,
            model=selected_model,
            using_api_key=bool(api_key),
        )

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
        chain = prompt | self.llm.with_structured_output(output_schema)
        return chain.invoke(input_vars)

    def generate_text(self, prompt: ChatPromptTemplate, input_vars: Dict[str, Any]) -> str:
        chain = prompt | self.llm
        result = chain.invoke(input_vars)
        return result.content


def get_llm_client() -> BaseLLMClient:
    """Factory function to get configured LLM client."""
    return OllamaClient()
