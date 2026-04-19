"""Tests for mock-scoped unit test generation helpers."""

from pathlib import Path

from codelyzer.diff.parser import DiffHunk, FileDiff
from codelyzer.testing import generator as generator_mod
from codelyzer.diff.parser import StructuredDiff
from codelyzer.testing.generator import (
    extract_unit_under_test_from_diff,
    mock_helpers_in_test_code,
    source_suggests_external_io,
    summarize_test_generation_preflight,
)


def test_extract_unit_under_test_from_added_defs() -> None:
    diff = FileDiff(
        file_path=Path("pkg/mod.py"),
        change_type="modified",
        hunks=[
            DiffHunk(
                start_line_old=1,
                start_line_new=1,
                lines_old=0,
                lines_new=5,
                content="@@\n+def fetch_user(x):\n+    return x\n",
            )
        ],
    )
    label, scope = extract_unit_under_test_from_diff(diff)
    assert "fetch_user" in label
    assert "Primary units to test" in scope
    assert "patch" in scope.lower() or "MonkeyPatch" in scope


def test_extract_unit_under_test_fallback_when_no_defs() -> None:
    diff = FileDiff(
        file_path=Path("pkg/mod.py"),
        change_type="modified",
        hunks=[
            DiffHunk(
                start_line_old=1,
                start_line_new=1,
                lines_old=0,
                lines_new=1,
                content="@@\n+FOO = 1\n",
            )
        ],
    )
    label, scope = extract_unit_under_test_from_diff(diff)
    assert "entire module" in label.lower()
    assert "mock" in scope.lower()


def test_source_suggests_external_io() -> None:
    assert source_suggests_external_io("def x():\n    return requests.get('https://a')\n")
    assert not source_suggests_external_io("def x():\n    return 1 + 1\n")


def test_summarize_test_generation_preflight_empty() -> None:
    assert summarize_test_generation_preflight(None) == {
        "changed_files": 0,
        "eligible": 0,
        "skip_counts": {},
    }
    diff = StructuredDiff(
        base_commit="a",
        target_commit="b",
        total_files_changed=0,
        total_insertions=0,
        total_deletions=0,
        files=[],
    )
    assert summarize_test_generation_preflight(diff)["changed_files"] == 0


def test_append_empty_generated_yields_no_file_change_message(tmp_path: Path) -> None:
    existing = "def test_foo():\n    assert True\n"
    target = tmp_path / "test_mod.py"
    target.write_text(existing, encoding="utf-8")
    _, names, reason = generator_mod._append_generated_tests(target, "")
    assert names == []
    assert "already exist" in reason
    assert "No file change" in reason


def test_mock_helpers_in_test_code() -> None:
    assert mock_helpers_in_test_code(
        "from unittest.mock import patch\n\ndef test_a():\n    with patch('m.x') as m:\n        m.return_value = 1\n        assert True\n"
    )
    assert not mock_helpers_in_test_code("def test_a():\n    assert 1 == 1\n")
