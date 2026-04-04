from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str = ""
    ollama_model: str = "gemma4"

    # Retrieval Configuration
    chroma_persist_directory: Path = Path.home() / ".cache" / "codelyzer" / "chroma"
    embedding_model: str = "nomic-embed-text"
    indexing_batch_size: int = 2000
    chunk_size: int = 1024
    chunk_overlap: int = 200

    # Workflow Configuration
    max_context_tokens: int = 128000
    parallel_steps: bool = True
    summary_max_files: int = 30
    summary_max_chars_per_file: int = 3500

    # Git Configuration
    git_binary: str = "git"


settings = Settings()