"""Tests for test-generation post-processing and quality gate."""

import ast

from codelyzer.testing.generator import (
    _analyze_generated_test_quality,
    _contains_placeholder_meta_comments,
    _find_missing_symbol_hints,
    _strip_llm_meta_comments,
)


def test_strip_llm_meta_comments_removes_hedge_block() -> None:
    raw = '''import pytest

def test_firebase():
        # We need to ensure the actual implementation of init_firebase is tested,
        # but since we are testing the file directly, we focus on the exception handling and flow.
        # Since we are testing the function logic, we let it run against the mocks.
        # We need to patch the actual module where init_firebase resides if we were testing imports,
        # but here we are testing the function implementation directly.
    pass
'''
    out = _strip_llm_meta_comments(raw)
    assert "we need to ensure" not in out.lower()
    assert "since we are testing" not in out.lower()
    assert "def test_firebase" in out


def test_strip_llm_meta_comments_keeps_short_runs() -> None:
    raw = """def test_x():
    # we need to do one thing
    # second line only
    assert True
"""
    out = _strip_llm_meta_comments(raw)
    assert "we need to" in out


def test_strip_llm_meta_comments_keeps_non_hedge_blocks() -> None:
    raw = """def test_x():
    # edge case: empty input
    # expect ValueError from validator
    # see issue 123
    assert True
"""
    out = _strip_llm_meta_comments(raw)
    assert "edge case" in out
    assert "ValueError" in out


def test_strip_llm_meta_comments_preserves_protected_block() -> None:
    raw = """def test_x():
    # noqa: F401 — we need to import for side effects
    # since we register hooks at import time
    # focus on import path only
    assert True
"""
    out = _strip_llm_meta_comments(raw)
    assert "noqa" in out
    assert "since we register" in out


def test_analyze_quality_no_tests() -> None:
    tree = ast.parse("x = 1\n")
    assert _analyze_generated_test_quality(tree) == (None, None)


def test_analyze_quality_good_assert() -> None:
    tree = ast.parse(
        """
def test_ok():
    assert 1 == 1
"""
    )
    assert _analyze_generated_test_quality(tree) == (None, None)


def test_analyze_quality_pytest_raises() -> None:
    tree = ast.parse(
        """
import pytest

def test_raises():
    with pytest.raises(ValueError):
        raise ValueError("x")
"""
    )
    assert _analyze_generated_test_quality(tree) == (None, None)


def test_analyze_quality_mock_assert() -> None:
    tree = ast.parse(
        """
from unittest.mock import MagicMock

def test_mock():
    m = MagicMock()
    m()
    m.assert_called_once()
"""
    )
    assert _analyze_generated_test_quality(tree) == (None, None)


def test_analyze_quality_all_vacuous_fallback() -> None:
    tree = ast.parse(
        """
def test_a():
    pass

def test_b():
    x = 1
"""
    )
    fb, partial = _analyze_generated_test_quality(tree)
    assert fb is not None
    assert "lack assertions" in fb
    assert "test_a" in fb and "test_b" in fb
    assert partial is None


def test_analyze_quality_skipped_exempt() -> None:
    tree = ast.parse(
        """
import pytest

@pytest.mark.skip(reason="integration")
def test_a():
    pass
"""
    )
    assert _analyze_generated_test_quality(tree) == (None, None)


def test_analyze_quality_mixed_partial_warn() -> None:
    tree = ast.parse(
        """
def test_good():
    assert True

def test_bad():
    pass
"""
    )
    fb, partial = _analyze_generated_test_quality(tree)
    assert fb is None
    assert partial is not None
    assert "test_bad" in partial
    assert "Some generated tests lack assertions" in partial


def test_find_missing_symbol_hints_detects_unimported_class() -> None:
    tree = ast.parse(
        """
def test_model():
    obj = BillCreate(title="x")
    assert obj.title == "x"
"""
    )
    missing = _find_missing_symbol_hints(tree)
    assert "BillCreate" in missing


def test_contains_placeholder_meta_comments_detects_narrative() -> None:
    code = """
def test_x():
    # Since we cannot run this file here, I will stop.
    # In a real scenario, this would patch dependencies.
    assert True
"""
    assert _contains_placeholder_meta_comments(code) is True
