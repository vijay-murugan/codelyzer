"""Split scorecards for PR-summary quality signals vs test-generation / QA signals."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from codelyzer.workflow.state import WorkflowState


class PRSummaryScorecard(BaseModel):
    """Heuristic / structural metrics for per-file PR-style summaries (not LLM-judge scores)."""

    summary_present: bool = False
    files_in_diff: int = 0
    fallback_phrase_hits: int = Field(
        0,
        description="Count of 'Could not generate LLM rationale' in pr_summary (per-file fallback used).",
    )
    parse_failure_proxy: int = Field(
        0,
        description="Same as fallback_phrase_hits for current pipeline (LLM errors route to fallback text).",
    )


class TestGenerationScorecard(BaseModel):
    """Counts and rates for generated tests and validation."""

    generated: int = 0
    appended: int = 0
    skipped: int = 0
    failed: int = 0
    partial_count: int = 0
    pytest_passed: bool | None = None
    coverage_total: int | None = None
    final_validation: str | None = None
    prune_events: int = 0
    refinement_fixes: int = 0


class E2ESuccess(BaseModel):
    """End-outcome booleans for dashboards (define which gates you require per run)."""

    tests_green: bool = False
    coverage_met: bool = True
    no_errors: bool = False
    summary_ok: bool = False
    overall: bool = False


def compute_pr_summary_scorecard(state: WorkflowState) -> PRSummaryScorecard:
    text = state.pr_summary or ""
    fallback_marker = "Could not generate LLM rationale"
    files = state.structured_diff.total_files_changed if state.structured_diff else 0
    hits = text.count(fallback_marker) if text else 0
    return PRSummaryScorecard(
        summary_present=bool(text.strip()),
        files_in_diff=files,
        fallback_phrase_hits=hits,
        parse_failure_proxy=hits,
    )


def compute_test_generation_scorecard(state: WorkflowState) -> TestGenerationScorecard:
    g = a = s = f = partial_c = 0
    for r in state.generated_tests.values():
        if r.status == "generated":
            g += 1
        elif r.status == "appended":
            a += 1
        elif r.status == "skipped":
            s += 1
        elif r.status == "failed":
            f += 1
        if r.partial:
            partial_c += 1

    tr = state.test_run_report or {}
    status = tr.get("status")
    pytest_passed: bool | None = None
    if status == "passed":
        pytest_passed = True
    elif status in {"failed", "timeout"}:
        pytest_passed = False

    cov = state.coverage_report or {}
    total = cov.get("total_percent")
    total_int: int | None = int(total) if isinstance(total, int) else None

    return TestGenerationScorecard(
        generated=g,
        appended=a,
        skipped=s,
        failed=f,
        partial_count=partial_c,
        pytest_passed=pytest_passed,
        coverage_total=total_int,
        final_validation=state.final_validation_status,
        prune_events=len(state.removed_tests),
        refinement_fixes=len(state.refinement_fixes),
    )


def compute_e2e_success(
    state: WorkflowState,
    *,
    require_tests: bool = False,
    min_coverage: int | None = None,
    require_summary: bool = True,
) -> E2ESuccess:
    """
    Task completion from final state only (Level C).

    ``require_tests``: if True, treat success as pytest passed when a test run was recorded.
    """
    tr = state.test_run_report or {}
    tests_green = tr.get("status") == "passed"
    if not require_tests:
        tests_green = True

    cov = state.coverage_report.get("total_percent")
    coverage_met = True
    if min_coverage is not None and isinstance(cov, int):
        coverage_met = cov >= min_coverage

    no_errors = len(state.errors) == 0

    sc = compute_pr_summary_scorecard(state)
    summary_ok = (not require_summary) or (sc.summary_present and sc.fallback_phrase_hits == 0)

    overall = tests_green and coverage_met and no_errors and summary_ok
    return E2ESuccess(
        tests_green=bool(tests_green),
        coverage_met=coverage_met,
        no_errors=no_errors,
        summary_ok=summary_ok,
        overall=overall,
    )


def scorecard_dict_for_jsonl(
    state: WorkflowState,
    *,
    require_tests: bool = False,
    min_coverage: int | None = None,
    require_summary: bool = True,
) -> dict[str, Any]:
    return {
        "pr_summary": compute_pr_summary_scorecard(state).model_dump(),
        "test_generation": compute_test_generation_scorecard(state).model_dump(),
        "e2e": compute_e2e_success(
            state,
            require_tests=require_tests,
            min_coverage=min_coverage,
            require_summary=require_summary,
        ).model_dump(),
    }
