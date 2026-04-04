# Codelyzer

Intelligent code analysis, PR summarization and automated test generation for git repositories.

## Architecture

Codelyzer uses a 3-engine architecture:

1.  **LLM Engine** - Pluggable interface for OpenAI, Anthropic and local Ollama models with structured output validation
2.  **Orchestration Engine** - LangGraph based stateful workflow system with checkpoint persistence
3.  **Retrieval Engine** - Semantic code search using Chroma vector database

## Features

✅ Works with **any git repository** without modifications
✅ Zero side effects on target repositories
✅ PR summary generation
✅ Release note generation
✅ Automated pytest generation
✅ Semantic code context retrieval
✅ Resumable workflows with checkpointing

## Installation

```bash
git clone <repo>
cd codelyzer
pip install -e .
```

## Configuration

Create `.env` file:
```ini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4
EMBEDDING_MODEL=nomic-embed-text
```

If the embedding model is not installed yet, the index command will try to pull it automatically. You can also pull it yourself with `ollama pull nomic-embed-text`.

## Usage

```bash
# Analyze uncommitted changes in repository
codelyzer analyze /path/to/your/repo

# Analyze diff between two branches
codelyzer analyze /path/to/your/repo --base main --target feature-branch

# Pre-build semantic index for repository
codelyzer index /path/to/your/repo
```

## Storage

All cache, indexes and checkpoints are stored outside target repositories:
```
~/.cache/codelyzer/
├── chroma/          # Vector database indexes
└── checkpoints/     # Workflow persistence