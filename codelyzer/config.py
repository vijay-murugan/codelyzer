from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_file_paths() -> tuple[Path, Path]:
    """Repo-root .env then cwd .env (later wins) so `codelyzer analyze /other/repo` can use a stable config."""
    return _REPO_ROOT / ".env", Path.cwd() / ".env"


# Merge .env into os.environ before Settings() — avoids import-order bugs when other modules import config first.
load_dotenv(_REPO_ROOT / ".env", override=False)
load_dotenv(Path.cwd() / ".env", override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_file_paths(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Configuration (local Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str = ""
    ollama_model: str = "gemma4"

    # Ollama Cloud (https://ollama.com/api) — used for chat/structured LLM when enabled.
    # Embeddings / `codelyzer index` still use ollama_base_url unless you change that separately.
    ollama_use_cloud: bool = False
    ollama_cloud_base_url: str = "https://ollama.com"
    ollama_cloud_model: str = "gpt-oss:120b"

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