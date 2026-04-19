"""Ordered step log for QA validation (Level B trajectory tests)."""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import Any, Iterator

_trajectory_log: ContextVar[list[dict[str, Any]] | None] = ContextVar("codelyzer_eval_trajectory", default=None)


def validation_step_names() -> list[str]:
    """Integrated-validation stages including optional prune + second pytest pass."""
    return [
        "review_generated_tests_agent",
        "auto_fix_generated_tests_agent",
        "run_test_and_coverage",
        "append_pytest_failure_findings",
        "remove_inappropriate_tests_agent",
        "run_test_and_coverage_retry",
        "append_pytest_failure_findings_retry",
        "test_research_agent",
        "write_qa_markdown_report",
    ]


def integrated_validation_core_subsequence() -> list[str]:
    """Minimal ordered steps always expected on the happy path (pytest passes first run)."""
    return [
        "review_generated_tests_agent",
        "auto_fix_generated_tests_agent",
        "run_test_and_coverage",
        "append_pytest_failure_findings",
        "test_research_agent",
        "write_qa_markdown_report",
    ]


def record_validation_step(step: str, **meta: Any) -> None:
    log = _trajectory_log.get()
    if log is None:
        return
    log.append({"step": step, "t": time.perf_counter(), **meta})


def current_trajectory() -> list[dict[str, Any]] | None:
    return _trajectory_log.get()


def trajectory_is_subsequence(
    recorded: list[dict[str, Any]],
    required: list[str],
) -> bool:
    """True if ``required`` appears in order within ``recorded`` step names (extras allowed)."""
    names = [item.get("step", "") for item in recorded]
    it: Iterator[str] = iter(required)
    need = next(it, None)
    for name in names:
        if need is None:
            return True
        if name == need:
            need = next(it, None)
    return need is None


class TrajectoryRecorder(AbstractContextManager[list[dict[str, Any]]]):
    """Context manager: while active, ``record_validation_step`` appends to the returned list."""

    def __init__(self) -> None:
        self._log: list[dict[str, Any]] = []
        self._token: Token[list[dict[str, Any]] | None] | None = None

    def __enter__(self) -> list[dict[str, Any]]:
        self._token = _trajectory_log.set(self._log)
        return self._log

    def __exit__(self, *args: object) -> None:
        if self._token is not None:
            _trajectory_log.reset(self._token)
            self._token = None
