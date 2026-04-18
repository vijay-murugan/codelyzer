from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import structlog
from langchain_core.prompts import ChatPromptTemplate

from codelyzer.config import settings
from codelyzer.diff.parser import FileDiff, StructuredDiff
from codelyzer.llm.client import BaseLLMClient, get_llm_client
from codelyzer.retrieval.indexer import RepositoryIndexer
from codelyzer.workflow.state import GeneratedTest, WorkflowState

logger = structlog.get_logger(__name__)

TEST_NAME_PATTERN = re.compile(r"^\s*def\s+(test_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)
IMPORT_PATTERN = re.compile(
    r"^\s*(?:from\s+[A-Za-z0-9_\.]+\s+import\s+.+|import\s+[A-Za-z0-9_\.]+(?:\s+as\s+\w+)?)\s*$"
)
_FULL_LINE_COMMENT_RE = re.compile(r"^\s+#")
_WHITESPACE_ONLY_RE = re.compile(r"^\s*$")
_PROTECTED_COMMENT_RE = re.compile(r"#.*\b(noqa|pragma|type:\s*ignore)\b", re.IGNORECASE)
_DEF_ADDED_IN_DIFF_RE = re.compile(r"^\+\s*(?:async\s+)?def\s+(\w+)\s*\(")
_IO_HINT_RES = (
    re.compile(r"\brequests\.(get|post|put|delete|patch|head|request)\b"),
    re.compile(r"\bhttpx\."),
    re.compile(r"\burllib\."),
    re.compile(r"\baiohttp\b"),
    re.compile(r"\bboto3?\b"),
    re.compile(r"\bfirebase\b"),
    re.compile(r"\bsqlite3\b"),
    re.compile(r"\bpsycopg"),
    re.compile(r"\bpymongo\b"),
    re.compile(r"\bsubprocess\."),
    re.compile(r"\burllib3\b"),
    re.compile(r"\bgrpc\."),
    re.compile(r"\bredis\b"),
    re.compile(r"\bmysql\b"),
    re.compile(r"\bcassandra\b"),
)
_MOCK_HINT_RES = (
    re.compile(r"\bpatch\s*\("),
    re.compile(r"\.patch\s*\("),
    re.compile(r"\bMonkeyPatch\b"),
    re.compile(r"\bMagicMock\b"),
    re.compile(r"\bMock\s*\("),
    re.compile(r"\bAsyncMock\b"),
    re.compile(r"unittest\.mock"),
    re.compile(r"@patch\b"),
    re.compile(r"\.return_value\b"),
    re.compile(r"\.side_effect\b"),
    re.compile(r"pytest\.MonkeyPatch\b"),
)

_META_HEDGE_RE = re.compile(
    r"\b("
    r"we need to|since we|let's\b|let us\b|focus on|cannot easily|for this specific test|"
    r"re-?running|if we were|simulate the|global state|complex setup|we focus on|"
    r"we test the|testing the function|for this test|here we are|"
    r"we cannot|we let|attempts to initialize|provided code structure"
    r")\b",
    re.IGNORECASE,
)
_PLACEHOLDER_META_RE = re.compile(
    r"\b("
    r"since we cannot|i will stop|placeholder structure|simulate the test structure|"
    r"cannot proceed without|assuming the file is|in a real scenario"
    r")\b",
    re.IGNORECASE,
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


def _strip_llm_meta_comments(code: str) -> str:
    """Remove consecutive full-line comment blocks that look like LLM hedging."""

    lines = code.splitlines()
    result: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FULL_LINE_COMMENT_RE.match(line) and not _WHITESPACE_ONLY_RE.match(line):
            run: List[str] = []
            j = i
            while j < len(lines):
                cur = lines[j]
                if _WHITESPACE_ONLY_RE.match(cur):
                    break
                if not _FULL_LINE_COMMENT_RE.match(cur):
                    break
                run.append(cur)
                j += 1
            if len(run) >= 3 and not any(_PROTECTED_COMMENT_RE.search(r) for r in run):
                meta_hits = sum(1 for r in run if _META_HEDGE_RE.search(r))
                if meta_hits > len(run) / 2:
                    i = j
                    continue
            result.extend(run)
            i = j
            continue
        result.append(line)
        i += 1
    return "\n".join(result)


def _decorator_chain_marks_skip(dec: ast.expr) -> bool:
    parts: List[str] = []
    cur: ast.expr | None = dec
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    chain = ".".join(reversed(parts)).lower()
    return "skip" in chain


def _function_is_skipped(node: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if _decorator_chain_marks_skip(target):
            return True
    return False


def _is_pytest_raises_context(expr: ast.expr) -> bool:
    if not isinstance(expr, ast.Call):
        return False
    func = expr.func
    parts: List[str] = []
    cur: ast.expr | None = func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    chain = ".".join(reversed(parts))
    return chain.endswith("raises") or chain == "raises"


def _call_is_mock_assert(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr.startswith("assert_called"):
        return True
    return False


def _stmt_list_has_check(stmts: List[ast.stmt]) -> bool:
    for stmt in stmts:
        if isinstance(stmt, ast.Assert):
            return True
        if isinstance(stmt, ast.With):
            for item in stmt.items:
                if _is_pytest_raises_context(item.context_expr):
                    return True
            if _stmt_list_has_check(stmt.body):
                return True
        if isinstance(stmt, ast.If):
            if _stmt_list_has_check(stmt.body) or _stmt_list_has_check(stmt.orelse):
                return True
        if isinstance(stmt, (ast.For, ast.While)):
            if _stmt_list_has_check(stmt.body) or _stmt_list_has_check(stmt.orelse):
                return True
        if isinstance(stmt, ast.Try):
            if _stmt_list_has_check(stmt.body):
                return True
            for handler in stmt.handlers:
                if _stmt_list_has_check(handler.body):
                    return True
            if _stmt_list_has_check(stmt.orelse) or _stmt_list_has_check(stmt.finalbody):
                return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            if _call_is_mock_assert(stmt.value):
                return True
    return False


def _analyze_generated_test_quality(tree: ast.Module) -> tuple[str | None, str | None]:
    """Return (fallback_reason, partial_warn). Fallback replaces output; partial_warn sets partial flag."""

    tests: List[ast.AsyncFunctionDef | ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            tests.append(node)

    if not tests:
        return None, None

    vacuous = [
        node.name
        for node in tests
        if not _function_is_skipped(node) and not _stmt_list_has_check(node.body)
    ]
    if not vacuous:
        return None, None

    has_good = any(
        not _function_is_skipped(n) and _stmt_list_has_check(n.body)
        for n in tests
    )
    if has_good:
        return None, f"Some generated tests lack assertions: {', '.join(vacuous)}"
    return f"Generated tests lack assertions: {', '.join(vacuous)}", None


def _collect_defined_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _find_missing_symbol_hints(tree: ast.Module) -> list[str]:
    defined = _collect_defined_names(tree)
    missing: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        ident = node.id
        if ident in defined:
            continue
        if ident in {"pytest", "patch", "Mock", "MagicMock", "AsyncMock", "monkeypatch"}:
            continue
        if ident and ident[0].isupper():
            missing.append(ident)
    deduped: list[str] = []
    for name in missing:
        if name not in deduped:
            deduped.append(name)
    return deduped[:5]


def _contains_placeholder_meta_comments(code: str) -> bool:
    comment_lines = [ln.strip() for ln in code.splitlines() if ln.strip().startswith("#")]
    if not comment_lines:
        return False
    text = "\n".join(comment_lines)
    hits = sum(1 for ln in comment_lines if _PLACEHOLDER_META_RE.search(ln))
    return hits >= 2 or _PLACEHOLDER_META_RE.search(text) is not None


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
        return (
            existing_content,
            [],
            f"No file change: all generated test_* names already exist in {target_path}; nothing appended",
        )

    separator = "\n\n# Generated by codelyzer\n"
    if existing_content and not existing_content.endswith("\n"):
        existing_content += "\n"
    new_content = f"{existing_content}{separator}{filtered_code}\n"
    return new_content, test_names, ""


def _create_test_file_content(generated_code: str) -> tuple[str, List[str]]:
    cleaned = generated_code.strip() + "\n"
    return cleaned, _extract_test_names(cleaned)


def extract_unit_under_test_from_diff(file_diff: FileDiff) -> Tuple[str, str]:
    """Derive function names from added lines in the diff; return label + scope rules for the prompt."""

    names: List[str] = []
    for hunk in file_diff.hunks:
        for line in hunk.content.splitlines():
            if not line.startswith("+"):
                continue
            if line.startswith("+++"):
                continue
            match = _DEF_ADDED_IN_DIFF_RE.match(line)
            if match:
                names.append(match.group(1))
    ordered: List[str] = []
    for name in names:
        if name not in ordered:
            ordered.append(name)
    if not ordered:
        label = "entire module (no new/changed def lines detected in diff)"
        scope = (
            "Focus tests on behavior exercised by the changed code. Call public entry points in this file. "
            "Mock external I/O (HTTP, DB, subprocess, cloud SDKs) at the name bound in this module—do not hit real services. "
            "Do not assert behavior of unrelated helpers unless the diff shows direct coupling."
        )
        return label, scope
    label = ", ".join(ordered)
    scope = (
        f"Primary units to test: {label}. "
        "Tests should exercise these functions (or the smallest public wrapper that calls them per the source). "
        "Patch or MonkeyPatch dependencies at the import path as used inside this file (where the name is looked up). "
        "Configure mocks with return_value/side_effect (and nested attributes like .json.return_value) to mirror what "
        "the function under test actually calls. No real HTTP, DB writes, or subprocesses. "
        "Avoid assertions that depend on code paths outside these units unless strictly required by the diff."
    )
    return label, scope


def source_suggests_external_io(source_code: str) -> bool:
    """Heuristic: source likely performs network/DB/subprocess I/O."""

    return any(rx.search(source_code) for rx in _IO_HINT_RES)


def mock_helpers_in_test_code(test_code: str) -> bool:
    """Heuristic: test code appears to isolate I/O via mocking."""

    return any(rx.search(test_code) for rx in _MOCK_HINT_RES)


def _candidate_for_file(file_diff: FileDiff) -> TestGenerationCandidate:
    if file_diff.change_type == "deleted":
        return TestGenerationCandidate(file_diff, "Deleted files do not need generated tests")
    if file_diff.file_path.suffix != ".py":
        return TestGenerationCandidate(file_diff, "Only Python source files are supported")
    if _is_test_like_path(file_diff.file_path):
        return TestGenerationCandidate(file_diff, "Changed test files are not generation targets")
    return TestGenerationCandidate(file_diff)


def summarize_test_generation_preflight(structured_diff: StructuredDiff | None) -> Dict[str, Any]:
    """Count changed files vs eligible Python generation targets for CLI warnings."""

    if not structured_diff or not structured_diff.files:
        return {
            "changed_files": 0,
            "eligible": 0,
            "skip_counts": {},
        }
    skip_counts: Dict[str, int] = {}
    eligible = 0
    for file_diff in structured_diff.files:
        candidate = _candidate_for_file(file_diff)
        if candidate.reason:
            skip_counts[candidate.reason] = skip_counts.get(candidate.reason, 0) + 1
        else:
            eligible += 1
    return {
        "changed_files": len(structured_diff.files),
        "eligible": eligible,
        "skip_counts": skip_counts,
    }


def _build_generation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You generate high-quality pytest unit tests for changed Python code.\n"
                "\n"
                "OUTPUT RULES:\n"
                "- Return only valid Python test code.\n"
                "- No markdown fences.\n"
                "- No prose, explanations, or narration.\n"
                "- No TODOs, placeholders, pseudo-code, or unfinished scaffolding.\n"
                "- No large comment blocks. At most one short '#' line if unavoidable.\n"
                "- Every def test_* must contain at least one assert, pytest.raises(...), or mock assert_called*.\n"
                "- If context is insufficient, emit the smallest runnable scaffold marked with "
                "@pytest.mark.skip(reason='insufficient context').\n"
                "\n"
                "TEST PHILOSOPHY:\n"
                "- Prefer unit tests over integration tests.\n"
                "- Test observable behavior, not implementation trivia.\n"
                "- Cover changed logic first.\n"
                "- Cover public APIs, changed branches, changed return values, changed exceptions, changed side effects.\n"
                "- Add regression tests for bugs implied by the diff.\n"
                "- Keep tests deterministic and isolated.\n"
                "\n"
                "MANDATORY COVERAGE CHECKLIST:\n"
                "1. Happy path / expected success case.\n"
                "2. At least one edge case.\n"
                "3. At least one error/failure path if code can fail.\n"
                "4. Boundary values where relevant (0, empty, None, min/max, single item).\n"
                "5. Branches introduced or modified in the diff.\n"
                "6. Return type / output shape validation when relevant.\n"
                "\n"
                "EDGE CASES TO CONSIDER:\n"
                "- Empty string/list/dict/set.\n"
                "- None inputs.\n"
                "- Missing keys.\n"
                "- Duplicate values.\n"
                "- Large values.\n"
                "- Negative numbers.\n"
                "- Whitespace-only strings.\n"
                "- Single-element collections.\n"
                "- Already-normalized inputs.\n"
                "- Unexpected but valid types.\n"
                "\n"
                "EXTERNAL I/O RULES:\n"
                "- Never call real HTTP, DB, cloud SDKs, subprocesses, sockets, filesystem writes, queues, or network.\n"
                "- Patch where symbol is resolved in module under test.\n"
                "- Use unittest.mock.patch / MagicMock / Mock or pytest monkeypatch.\n"
                "- Configure mocks to match production usage:\n"
                "  return_value, side_effect, nested attrs, status_code, json(), text, headers, fetchone(), etc.\n"
                "\n"
                "PATCHING RULE:\n"
                "- If app/service.py does 'import requests', patch 'app.service.requests.get'.\n"
                "- If app/service.py does 'from requests import get', patch 'app.service.get'.\n"
                "- Patch names exactly as bound in the source file.\n"
                "\n"
                "IMPORT PATH DERIVATION RULES:\n"
                "- Derive imports from repo_path + file_path.\n"
                "- Convert file path to dotted module path.\n"
                "- Remove repository root prefix.\n"
                "- Remove .py suffix.\n"
                "- Ignore tests/ prefixes when importing production code.\n"
                "- Example: repo=/repo, file=/repo/pkg/util/helpers.py -> from pkg.util.helpers import ...\n"
                "- Example: repo=/w, file=/w/src/app/api.py -> from src.app.api import ...\n"
                "- Example: file=/repo/package/mod.py -> import package.mod as mod\n"
                "\n"
                "STYLE RULES:\n"
                "- Use pytest only.\n"
                "- Use simple fixtures when repetition helps.\n"
                "- Use parametrize for multiple similar inputs.\n"
                "- Deterministic test names: test_<unit>_<behavior>.\n"
                "- Import only what is needed.\n"
                "- Keep tests concise but complete.\n"
                "\n"
                "FULL GOOD EXAMPLES:\n"
                "\n"
                "Example 1:\n"
                "from unittest.mock import patch, Mock\n"
                "from app.api import fetch_user\n"
                "\n"
                "def test_fetch_user_success():\n"
                "    resp = Mock(status_code=200)\n"
                "    resp.json.return_value = {'id': 1}\n"
                "    with patch('app.api.requests.get', return_value=resp) as m:\n"
                "        result = fetch_user(1)\n"
                "    assert result == {'id': 1}\n"
                "    m.assert_called_once()\n"
                "\n"
                "def test_fetch_user_http_error():\n"
                "    with patch('app.api.requests.get', side_effect=TimeoutError):\n"
                "        with pytest.raises(TimeoutError):\n"
                "            fetch_user(1)\n"
                "\n"
                "Example 2:\n"
                "import pytest\n"
                "from pkg.maths import divide\n"
                "\n"
                "@pytest.mark.parametrize(\n"
                "    'a,b,expected', [(6,3,2), (5,2,2.5)]\n"
                ")\n"
                "def test_divide_values(a,b,expected):\n"
                "    assert divide(a,b) == expected\n"
                "\n"
                "def test_divide_zero_denominator():\n"
                "    with pytest.raises(ZeroDivisionError):\n"
                "        divide(1,0)\n"
                "\n"
                "BAD OUTPUT:\n"
                "- Real network calls.\n"
                "- Recreating production code inside tests.\n"
                "- Missing asserts.\n"
                "- Narrative comments.\n"
                "- Incorrect patch path.\n"
                "- Random sleeps / nondeterminism.\n"
            ),
            (
                "human",
                "Repository path: {repo_path}\n"
                "Source file: {file_path}\n"
                "Unit under test (from diff): {unit_under_test}\n"
                "Scope rules:\n{scope_rules}\n\n"
                "Git diff:\n{diff_payload}\n\n"
                "Current source code:\n{source_code}\n\n"
                "Relevant repository context:\n{code_context}\n\n"
                "Existing related tests:\n{existing_tests}\n\n"
                "Research guidance for this file:\n{research_guidance}\n\n"
                "Generate pytest tests for the changed code only.\n"
                "Prioritize diff coverage first, then nearby regressions.\n"
                "Return only Python code.\n"
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
    research_guidance: str,
    unit_under_test: str,
    scope_rules: str,
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
                "unit_under_test": unit_under_test,
                "scope_rules": scope_rules,
                "diff_payload": diff_payload,
                "source_code": source_code,
                "code_context": _render_context_snippets(code_context),
                "existing_tests": _render_context_snippets(existing_tests, max_chars=3000),
                "research_guidance": research_guidance or "None",
            },
        )
    except Exception as exc:
        logger.warning("LLM test generation failed, falling back to scaffold", file=str(file_diff.file_path), error=str(exc))
        return _fallback_test_code(file_diff), True, f"LLM generation failed; scaffold only ({exc})"
    cleaned = _clean_generated_code(response)
    if not cleaned:
        return _fallback_test_code(file_diff), True, "LLM returned empty test content"

    stripped = _strip_llm_meta_comments(cleaned)
    try:
        tree = ast.parse(stripped)
    except SyntaxError as exc:
        logger.warning(
            "LLM test generation produced invalid Python",
            file=str(file_diff.file_path),
            error=str(exc),
        )
        return _fallback_test_code(file_diff), True, f"LLM returned invalid Python ({exc})"

    fallback_reason, partial_warn = _analyze_generated_test_quality(tree)
    if fallback_reason:
        return _fallback_test_code(file_diff), True, fallback_reason
    if _contains_placeholder_meta_comments(stripped):
        return (
            _fallback_test_code(file_diff),
            True,
            "Generated output contained placeholder/narrative comments instead of runnable tests",
        )
    missing_symbols = _find_missing_symbol_hints(tree)
    if missing_symbols:
        return (
            _fallback_test_code(file_diff),
            True,
            f"Generated tests reference unresolved symbols without imports: {', '.join(missing_symbols)}",
        )

    partial = "pytest.mark.skip" in stripped or bool(partial_warn)
    if partial_warn:
        reason: str | None = partial_warn
    elif partial:
        reason = "Generated scaffold requires manual completion"
    else:
        reason = None

    if (
        source_suggests_external_io(source_code)
        and not mock_helpers_in_test_code(stripped)
        and "pytest.mark.skip" not in stripped
    ):
        partial = True
        io_msg = (
            "Heuristic: source may use external I/O; tests show no patch/Mock/MonkeyPatch—verify isolation or add mocks "
            "matching call-site behavior."
        )
        reason = f"{reason}; {io_msg}" if reason else io_msg

    return stripped, partial, reason


def _research_guidance_for_file(state: WorkflowState, source_key: str) -> str:
    matched: List[str] = []
    for item in state.pre_generation_research:
        target = str(item.get("target_file", ""))
        if target == source_key or target.endswith(source_key):
            rationale = str(item.get("rationale", "")).strip()
            suggested = str(item.get("suggested_test_name", "")).strip()
            if suggested:
                matched.append(f"{suggested}: {rationale}")
            elif rationale:
                matched.append(rationale)
    return "\n".join(matched) if matched else "None"


def generate_tests_for_changes(
    state: WorkflowState,
    indexer: RepositoryIndexer | None = None,
) -> WorkflowState:
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

            unit_label, scope_rules = extract_unit_under_test_from_diff(file_diff)
            generated_code, partial, reason = _generate_test_code(
                llm=llm,
                repo_path=state.repo_path,
                file_diff=file_diff,
                source_code=source_code,
                code_context=code_context,
                existing_tests=related_tests,
                research_guidance=_research_guidance_for_file(state, source_key),
                unit_under_test=unit_label,
                scope_rules=scope_rules,
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
                trace_context={
                    "change_type": file_diff.change_type,
                    "used_research_guidance": _research_guidance_for_file(state, source_key) != "None",
                    "unit_under_test": unit_label,
                    "source_suggests_io": source_suggests_external_io(source_code),
                    "test_uses_mocks": mock_helpers_in_test_code(
                        persisted_code if isinstance(persisted_code, str) else ""
                    ),
                },
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
