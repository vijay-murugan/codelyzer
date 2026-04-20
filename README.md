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

### Ollama Cloud (hosted LLM)

For chat / structured outputs against [Ollama Cloud](https://docs.ollama.com/cloud) instead of a local daemon, set an [API key](https://ollama.com/settings/keys) and either enable cloud mode **or** point the LLM host at `https://ollama.com`. Indexing embeddings still use `OLLAMA_BASE_URL` when you use the split config below (local base URL + cloud flag).

**Option A — explicit cloud fields (keep local `OLLAMA_BASE_URL` for embeddings):**

```ini
OLLAMA_USE_CLOUD=true
OLLAMA_API_KEY=your_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CLOUD_BASE_URL=https://ollama.com
OLLAMA_CLOUD_MODEL=gpt-oss:120b
```

**Option B — single host for the LLM (no `OLLAMA_USE_CLOUD` needed):**

```ini
OLLAMA_API_KEY=your_key
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_MODEL=gpt-oss:120b
```

`.env` is loaded from the **codelyzer repo root** and then the **current working directory** (cwd wins on duplicate keys), so flags apply even if you run `codelyzer analyze /some/other/repo` from another directory.

## Usage

```bash
# Analyze uncommitted changes in repository
codelyzer analyze /path/to/your/repo

# Analyze diff between two branches
codelyzer analyze /path/to/your/repo --base main --target feature-branch

# Pre-build semantic index for repository
codelyzer index /path/to/your/repo

# Generate pytest tests for changed Python files (requires a non-empty diff)
codelyzer analyze /path/to/your/repo --generate-tests

# Same, with auto-validation (research, review, rerun, coverage report)
codelyzer analyze /path/to/your/repo --generate-tests --auto-validate-tests --cov-target . --qa-report-path /path/to/your/repo/reports/qa_report.md
```

## Troubleshooting: no test files generated

- Pass **`--generate-tests`**. The `analyze` command does not write tests by default.
- The git diff must be **non-empty**: default is uncommitted changes vs `HEAD`. If everything is committed, you will see a warning. Compare branches instead, e.g. `codelyzer analyze /path/to/repo --base main --target feature --generate-tests`.
- Only **non-test** `.py` files are targets (paths under `tests/`, `test_*.py`, or `*_test.py` layouts are skipped). A diff of only config, markdown, or test files will not create new production tests.
- If tests are **appended** to an existing file and every generated `test_*` name already exists, nothing is written; the report will say all names already exist (**no file change**).

## Storage

All cache, indexes and checkpoints are stored outside target repositories:
```
~/.cache/codelyzer/
├── chroma/          # Vector database indexes
└── checkpoints/     # Workflow persistence