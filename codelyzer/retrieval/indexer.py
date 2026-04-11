from pathlib import Path
import subprocess
from git import Repo
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import Language
import structlog
from typing import List, Dict, Any

from codelyzer.config import settings

logger = structlog.get_logger(__name__)


class RepositoryIndexer:
    """Semantic code indexer for git repositories using vector database."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path.resolve()
        self.repository_id = self._get_repo_id()
        self.reused_existing_index = False
        self.embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
        )
        self.vector_store = None
        self._initialize_store()

    def _get_repo_id(self) -> str:
        """Generate unique identifier for repository."""
        repo = Repo(self.repo_path)
        return repo.head.object.hexsha[:8] if repo.head.is_valid() else str(hash(self.repo_path))

    def _initialize_store(self):
        """Initialize Chroma vector store for this repository."""
        persist_dir = settings.chroma_persist_directory / self.repository_id
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.vector_store = Chroma(
            collection_name=f"codelyzer-{self.repository_id}",
            embedding_function=self.embeddings,
            persist_directory=str(persist_dir)
        )
        logger.debug("Initialized vector store", repo=self.repo_path, persist_dir=persist_dir)

    def _pull_embedding_model(self) -> None:
        """Pull the configured Ollama embedding model if it is missing locally."""
        logger.warning("Embedding model missing locally, pulling it", model=settings.embedding_model)

        try:
            result = subprocess.run(
                ["ollama", "pull", settings.embedding_model],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Ollama CLI was not found. Install Ollama or run `ollama pull <model>` manually."
            ) from exc

        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"Failed to pull embedding model '{settings.embedding_model}': {details}"
            )

    def _should_pull_model(self, error: Exception) -> bool:
        message = str(error).lower()
        return "404" in message and "model" in message or "not found" in message

    def _add_documents_with_retry(self, docs: List[Any]) -> None:
        """Add a document batch and retry once after auto-pulling a missing model."""
        try:
            self.vector_store.add_documents(docs)
        except Exception as exc:
            if self._should_pull_model(exc):
                self._pull_embedding_model()
                self.vector_store.add_documents(docs)
            else:
                raise

    def _index_in_batches(self, chunks: List[Any]) -> None:
        """Insert chunks in bounded batches to avoid vector store batch-size errors."""
        batch_size = max(1, settings.indexing_batch_size)
        total = len(chunks)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self._add_documents_with_retry(chunks[start:end])
            logger.info("Indexed chunk batch", start=start, end=end, total=total)

    def _existing_embedding_count(self) -> int:
        """Return number of stored vectors for this repository collection, if available."""
        collection = getattr(self.vector_store, "_collection", None)
        if collection is None or not hasattr(collection, "count"):
            return 0

        try:
            return int(collection.count())
        except Exception:
            return 0

    def index_repository(self, file_types: List[str] = None) -> int:
        """Index all code files in the repository."""
        existing_count = self._existing_embedding_count()
        if existing_count > 0:
            self.reused_existing_index = True
            logger.info("Using existing repository embeddings", vectors=existing_count)
            return existing_count

        if file_types is None:
            file_types = ['.py']

        documents = []

        for ext in file_types:
            loader = GenericLoader.from_filesystem(
                str(self.repo_path),
                glob=f"**/*{ext}",
                exclude=[ "**/__pycache__/**", "**/.git/**"],
                suffixes=[ext],
                parser=LanguageParser(language=Language.PYTHON, parser_threshold=1000)
            )
            docs = loader.load()
            documents.extend(docs)

        logger.info("Loaded source files", count=len(documents))

        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        chunks = python_splitter.split_documents(documents)
        logger.info("Split into code chunks", chunks=len(chunks))

        self._index_in_batches(chunks)
        logger.debug("Chroma handles persistence through configured persist_directory")

        logger.info("Repository indexing complete", total_chunks=len(chunks))
        return len(chunks)

    def search_relevant_code(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for semantically relevant code fragments."""
        results = self.vector_store.similarity_search_with_score(query, k=limit)

        matches = []
        for doc, score in results:
            matches.append({
                "content": doc.page_content,
                "file_path": doc.metadata.get("source"),
                "line_start": doc.metadata.get("line_start"),
                "line_end": doc.metadata.get("line_end"),
                "relevance_score": float(score)
            })

        return matches

    def search_tests_for_file(self, file_path: Path, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve existing test files related to a given source file."""
        test_queries = [f"test {file_path.name}", f"tests for {file_path.stem}"]

        all_matches = []
        for query in test_queries:
            matches = self.vector_store.similarity_search(query, k=limit, filter={"source": {"$contains": "test"}})
            all_matches.extend(matches)

        unique = {doc.metadata["source"]: doc for doc in all_matches}

        return [{
            "content": doc.page_content,
            "file_path": doc.metadata.get("source"),
        } for doc in unique.values()]