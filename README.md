# Codelyzer

Intelligent code analysis, PR summarization and automated test generation for git repositories.

## Architecture

Codelyzer uses a 3-engine architecture:

1.  **LLM Engine** - Pluggable interface for OpenAI, Anthropic, Groq and local Ollama models with structured output validation
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
✅ **GitHub Actions PR check** — auto-analyze and comment on PRs
✅ **Groq cloud LLM support** for CI environments

## Installation

```bash
git clone <repo>
cd codelyzer
pip install -e .

# For CI/GitHub Actions (Groq cloud LLM provider):
pip install -e ".[ci]"
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

### CLI — Local Analysis

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

### CLI — PR Analysis (for CI)

```bash
# Run analysis between two git refs and write a PR comment markdown file
codelyzer pr-analyze /path/to/repo --base main --target feature-branch --output pr_comment.md

# Same, but also post the comment to a GitHub PR
GITHUB_TOKEN=ghp_... PR_NUMBER=42 GITHUB_REPOSITORY=owner/repo \
  codelyzer pr-analyze . --base main --target feature-branch --post-comment

# Include test generation in the PR comment
codelyzer pr-analyze . --base main --target feature-branch --generate-tests --output pr_comment.md
```

## GitHub Actions — Automated PR Checks

Codelyzer ships with a GitHub Actions workflow that **automatically analyzes every pull request** and posts a rich comment with:

- 📊 Diff statistics (files changed, insertions, deletions)
- 📝 Per-file LLM analysis (what changed, why, and what it does)
- 🧪 Optional test generation results
- 📚 Retrieval context summary
- ⚠️ Errors and warnings

### Setup

1. **Add your Groq API key** as a repository secret:

   Go to **Settings → Secrets and variables → Actions → New repository secret**

   | Secret Name   | Value                     |
   |---------------|---------------------------|
   | `GROQ_API_KEY`| Your Groq API key (`gsk_...`) |

2. **The workflow is already configured** in `.github/workflows/codelyzer-pr.yml`. It will run automatically on every PR to `main` or `develop`.

3. **`GITHUB_TOKEN`** is provided automatically by GitHub Actions — no additional setup needed for posting comments.

### How It Works

```
PR opened/updated → GitHub Actions triggers
  → codelyzer pr-analyze . --base <base-sha> --target <head-sha> --post-comment
    → Parses git diff between PR base and head
    → LLM analyzes each changed file via Groq cloud API
    → Formats results as rich markdown
    → Posts (or updates) a comment on the PR
```

### LLM Provider Selection

| Environment | Provider | Configuration |
|-------------|----------|---------------|
| Local dev   | Ollama   | `OLLAMA_MODEL` in `.env` |
| GitHub Actions (CI) | Groq | `GROQ_API_KEY` secret |

The LLM client auto-selects: if `GROQ_API_KEY` is set, it uses Groq; otherwise it falls back to local Ollama.

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
```