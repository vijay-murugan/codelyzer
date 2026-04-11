from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import structlog
from langchain_core.prompts import ChatPromptTemplate

from codelyzer.config import settings
from codelyzer.diff.parser import FileDiff
from codelyzer.llm.client import BaseLLMClient, get_llm_client
from codelyzer.retrieval.indexer import RepositoryIndexer
from codelyzer.workflow.state import GeneratedTest, WorkflowState

logger = structlog.get_logger(__name__)

TEST_NAME_PATTERN = re.compile(r"^\s*def\s+(test_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)
IMPORT_PATTERN = re.compile(
    r"^\s*(?:from\s+[A-Za-z0-9_\.]+\s+import\s+.+|import\s+[A-Za-z0-9_\.]+(?:\s+as\s+\w+)?)\s*$"
)


@dataclass
class TestGenerationCandidate:
    file_diff: FileDiff
    reason: str | None = None


def _is_test_like_path(file_path: Path) -> bool:
    parts = {part.lower() for part in file_path.parts}
    name = file_path.name.lower()
    stem = file_path.stem.lower()
    return "tests" in parts or name.startswith("test_") or stem.endswith("_test")


def _build_diff_payload(file_diff: FileDiff, max_chars: int) -> str:
    chunks: List[str] = []
    for idx, hunk in enumerate(file_diff.hunks, start=1):
        chunks.append(
            f"@@ hunk {idx} -{hunk.start_line_old},{hunk.lines_old} +{hunk.start_line_new},{hunk.lines_new}\n"
            f"{hunk.content}"
        )
    payload = "\n".join(chunks)
    if len(payload) > max_chars:
        return payload[:max_chars] + "\n[TRUNCATED]"
    return payload


def _clean_generated_code(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_test_names(code: str) -> List[str]:
    return TEST_NAME_PATTERN.findall(code)


def _render_context_snippets(matches: List[dict], max_chars: int = 5000) -> str:
    if not matches:
        return "None"

    snippets: List[str] = []
    total = 0
    for match in matches:
        file_path = match.get("file_path", "unknown")
        content = str(match.get("content", "")).strip()
        snippet = f"File: {file_path}\n{content}"
        total += len(snippet)
        if total > max_chars:
            break
        snippets.append(snippet)
    return "\n\n".join(snippets) if snippets else "None"


def _fallback_test_code(file_diff: FileDiff) -> str:
    module_name = file_diff.file_path.stem
    return (
        "import pytest\n\n"
        f"@pytest.mark.skip(reason=\"Insufficient context to generate reliable tests for {module_name}\")\n"
        f"def test_{module_name}_change_requires_manual_completion():\n"
        "    raise NotImplementedError(\"Complete this generated scaffold with project-specific assertions\")\n"
    )


def _pick_existing_test_path(file_path: Path, related_tests: List[dict]) -> Path | None:
    candidates: List[tuple[tuple[int, int, str], Path]] = []
    for item in related_tests:
        raw_path = item.get("file_path")
        if not raw_path:
            continue
        candidate = Path(raw_path)
        name_distance = 0 if file_path.stem in candidate.stem else 1
        score = (name_distance, len(candidate.parts), str(candidate))
        candidates.append((score, candidate))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _default_test_path(repo_path: Path, source_file: Path) -> Path:
    return repo_path / "tests" / f"test_{source_file.stem}.py"


def _resolve_test_path(repo_path: Path, source_file: Path, related_tests: List[dict]) -> tuple[Path, bool]:
    existing = _pick_existing_test_path(source_file, related_tests)
    if existing is not None:
        return repo_path / existing, True
    return _default_test_path(repo_path, source_file), False


def _filter_append_block(existing_content: str, generated_code: str) -> tuple[str, List[str]]:
    existing_tests = set(_extract_test_names(existing_content))
    existing_imports = {
        line.strip()
        for line in existing_content.splitlines()
        if IMPORT_PATTERN.match(line.strip())
    }

    kept_lines: List[str] = []
    skip_block = False
    current_test_name: str | None = None
    kept_tests: List[str] = []

    for line in generated_code.splitlines():
        stripped = line.strip()
        import_match = IMPORT_PATTERN.match(stripped)
        test_match = TEST_NAME_PATTERN.match(line)

        if test_match:
            current_test_name = test_match.group(1)
            skip_block = current_test_name in existing_tests
            if not skip_block:
                kept_tests.append(current_test_name)

        if import_match and stripped in existing_imports:
            continue

        if skip_block:
            if stripped and not line.startswith((" ", "\t")) and not stripped.startswith("@"):
                skip_block = False
                current_test_name = None
            else:
                continue

        kept_lines.append(line)
        if import_match:
            existing_imports.add(stripped)

        if current_test_name and stripped == "":
            current_test_name = None

    filtered = "\n".join(kept_lines).strip()
    return filtered, kept_tests


def _append_generated_tests(target_path: Path, generated_code: str) -> tuple[str, List[str], str]:
    existing_content = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    filtered_code, test_names = _filter_append_block(existing_content, generated_code)
    if not filtered_code:
        return existing_content, [], "No non-duplicate tests were generated"

    separator = "\n\n# Generated by codelyzer\n"
    if existing_content and not existing_content.endswith("\n"):
        existing_content += "\n"
    new_content = f"{existing_content}{separator}{filtered_code}\n"
    return new_content, test_names, ""


def _create_test_file_content(generated_code: str) -> tuple[str, List[str]]:
    cleaned = generated_code.strip() + "\n"
    return cleaned, _extract_test_names(cleaned)


def _candidate_for_file(file_diff: FileDiff) -> TestGenerationCandidate:
    if file_diff.change_type == "deleted":
        return TestGenerationCandidate(file_diff, "Deleted files do not need generated tests")
    if file_diff.file_path.suffix != ".py":
        return TestGenerationCandidate(file_diff, "Only Python source files are supported")
    if _is_test_like_path(file_diff.file_path):
        return TestGenerationCandidate(file_diff, "Changed test files are not generation targets")
    return TestGenerationCandidate(file_diff)


def _build_generation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You generate pytest unit tests for changed Python code. "
                "Return only valid Python test code, with no markdown fences and no explanation. "
                "Prefer unit tests over integration tests. "
                "Add mocks or monkeypatching when code touches I/O, subprocesses, network, databases, or environment. "
                "Cover newly added branches, changed return values, changed error handling, and visible public behavior. "
                "If context is incomplete, emit the smallest runnable pytest scaffold and mark it skipped."
            ),
            (
                "human",
                "Repository path: {repo_path}\n"
                "Source file: {file_path}\n"
                "Git diff:\n{diff_payload}\n\n"
                "Current source code:\n{source_code}\n\n"
                "Relevant repository context:\n{code_context}\n\n"
                "Existing related tests:\n{existing_tests}\n\n"
                "Requirements:\n"
                "- Use pytest style only.\n"
                "- Create test functions with deterministic names.\n"
                "- Import only what is needed.\n"
                "- Do not explain the tests.\n"
                "- Return only Python code.\n"
            ),
        ]
    )


def _generate_test_code(
    llm: BaseLLMClient | None,
    repo_path: Path,
    file_diff: FileDiff,
    source_code: str,
    code_context: List[dict],
    existing_tests: List[dict],
) -> tuple[str, bool, str | None]:
    if llm is None:
        return _fallback_test_code(file_diff), True, "LLM unavailable; generated scaffold only"

    prompt = _build_generation_prompt()
    diff_payload = _build_diff_payload(file_diff, max_chars=max(1000, settings.summary_max_chars_per_file))
    try:
        response = llm.generate_text(
            prompt,
            {
                "repo_path": str(repo_path),
                "file_path": str(file_diff.file_path),
                "diff_payload": diff_payload,
                "source_code": source_code,
                "code_context": _render_context_snippets(code_context),
                "existing_tests": _render_context_snippets(existing_tests, max_chars=3000),
            },
        )
    except Exception as exc:
        logger.warning("LLM test generation failed, falling back to scaffold", file=str(file_diff.file_path), error=str(exc))
        return _fallback_test_code(file_diff), True, f"LLM generation failed; scaffold only ({exc})"
    cleaned = _clean_generated_code(response)
    if not cleaned:
        return _fallback_test_code(file_diff), True, "LLM returned empty test content"

    partial = "pytest.mark.skip" in cleaned
    reason = "Generated scaffold requires manual completion" if partial else None
    return cleaned, partial, reason


def generate_tests_for_changes(state: WorkflowState, indexer: RepositoryIndexer | None = None) -> WorkflowState:
    if not state.structured_diff or not state.structured_diff.files:
        return state

    llm: BaseLLMClient | None = None
    try:
        llm = get_llm_client()
    except Exception as exc:
        logger.warning("LLM client unavailable for test generation", error=str(exc))

    for file_diff in state.structured_diff.files:
        candidate = _candidate_for_file(file_diff)
        source_key = str(file_diff.file_path)
        if candidate.reason:
            state.generated_tests[source_key] = GeneratedTest(
                source_file_path=source_key,
                test_file_path="",
                test_code="",
                test_names=[],
                status="skipped",
                reason=candidate.reason,
                partial=False,
            )
            continue

        source_path = state.repo_path / file_diff.file_path
        if not source_path.exists():
            state.generated_tests[source_key] = GeneratedTest(
                source_file_path=source_key,
                test_file_path="",
                test_code="",
                test_names=[],
                status="failed",
                reason="Changed source file was not found on disk",
                partial=False,
            )
            continue

        try:
            source_code = source_path.read_text(encoding="utf-8")
            related_tests = state.existing_tests.get(source_key, [])
            if indexer is not None:
                query = f"pytest tests for {file_diff.file_path.name} changed behavior"
                code_context = indexer.search_relevant_code(query, limit=5)
                state.code_context[source_key] = code_context
                if not related_tests:
                    related_tests = indexer.search_tests_for_file(file_diff.file_path)
                    if related_tests:
                        state.existing_tests[source_key] = related_tests
            else:
                code_context = state.code_context.get(source_key, [])

            generated_code, partial, reason = _generate_test_code(
                llm=llm,
                repo_path=state.repo_path,
                file_diff=file_diff,
                source_code=source_code,
                code_context=code_context,
                existing_tests=related_tests,
            )

            test_path, appending = _resolve_test_path(state.repo_path, file_diff.file_path, related_tests)
            test_path.parent.mkdir(parents=True, exist_ok=True)

            if appending:
                rendered_content, test_names, dedupe_reason = _append_generated_tests(test_path, generated_code)
                if not test_names and dedupe_reason:
                    state.generated_tests[source_key] = GeneratedTest(
                        source_file_path=source_key,
                        test_file_path=str(test_path),
                        test_code="",
                        test_names=[],
                        status="skipped",
                        reason=dedupe_reason,
                        partial=partial,
                    )
                    continue
                test_path.write_text(rendered_content, encoding="utf-8")
                status = "appended"
                persisted_code = generated_code
            else:
                rendered_content, test_names = _create_test_file_content(generated_code)
                test_path.write_text(rendered_content, encoding="utf-8")
                status = "generated"
                persisted_code = rendered_content

            state.generated_tests[source_key] = GeneratedTest(
                source_file_path=source_key,
                test_file_path=str(test_path),
                test_code=persisted_code,
                test_names=test_names,
                status=status,
                reason=reason,
                partial=partial,
            )
        except Exception as exc:
            state.generated_tests[source_key] = GeneratedTest(
                source_file_path=source_key,
                test_file_path="",
                test_code="",
                test_names=[],
                status="failed",
                reason=str(exc),
                partial=False,
            )
            state.add_error(f"Test generation failed for {source_key}: {exc}")

    return state
