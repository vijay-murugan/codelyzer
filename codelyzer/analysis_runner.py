"""Headless analyze pipeline for batch eval and scripting (no Click I/O)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from codelyzer.changed_files_summary import render_changed_files_summary
from codelyzer.diff.parser import GitDiffParser
from codelyzer.eval.tracing import traceable_step
from codelyzer.retrieval.indexer import RepositoryIndexer
from codelyzer.testing.generator import generate_tests_for_changes
from codelyzer.testing.qa_agents import (
    research_before_generation_agent,
    run_integrated_generation_validation,
    run_qa_agents,
)
from codelyzer.workflow.state import WorkflowState

logger = structlog.get_logger(__name__)


@traceable_step("codelyzer_run_analyze_workflow")
def run_analyze_workflow(
    repo_path: Path,
    base: str = "HEAD",
    target: str | None = None,
    *,
    generate_tests: bool = False,
    auto_validate_tests: bool = False,
    run_qa: bool = False,
    qa_report_path: Path | None = None,
    cov_target: str = "codelyzer",
    min_coverage: int | None = None,
    pytest_targets: list[str] | None = None,
    use_system_python: bool = False,
    coverage_scope: str = "runtime",
) -> WorkflowState:
    """
    Run the same logic as ``codelyzer analyze`` and return final ``WorkflowState``.

    ``pytest_targets``: when set, used as pytest paths if no generated/appended test files
    define the run set; otherwise defaults to ``tests``.

    ``use_system_python``: when True, QA uses ``sys.executable`` and does not create
    ``.codelyzer_venv`` or run ``pip install``.

    ``coverage_scope``: ``runtime`` keeps current behavior; ``full_source`` forces
    coverage source enumeration against ``cov_target`` so unexecuted files count as 0%.

    Raises on fatal diff parse failure (caller may catch).
    """
    state = WorkflowState(
        repo_path=repo_path,
        base_ref=base,
        target_ref=target,
        pytest_targets=pytest_targets,
        use_system_python=use_system_python,
        coverage_scope="full_source" if coverage_scope == "full_source" else "runtime",
    )

    parser = GitDiffParser(repo_path)
    state.structured_diff = parser.parse_diff(base, target)
    state.pr_summary = render_changed_files_summary(state.structured_diff)

    indexer: RepositoryIndexer | None = None
    try:
        indexer = RepositoryIndexer(repo_path)
        indexer.index_repository()
    except Exception as exc:
        logger.warning("Indexing failed", error=str(exc))
        state.add_error(f"Indexing warning: {exc}")

    try:
        if indexer is not None and state.structured_diff:
            for file_diff in state.structured_diff.files:
                tests = indexer.search_tests_for_file(file_diff.file_path)
                if tests:
                    state.existing_tests[str(file_diff.file_path)] = tests
    except Exception as exc:
        state.add_error(f"Test search warning: {exc}")

    if generate_tests:
        if auto_validate_tests:
            state = research_before_generation_agent(state)
        state = generate_tests_for_changes(state, indexer=indexer)
        if auto_validate_tests:
            default_report_path = repo_path / "reports" / "qa_report.md"
            report_path = qa_report_path or default_report_path
            state = run_integrated_generation_validation(
                state,
                report_path=report_path,
                cov_target=cov_target,
                min_coverage=min_coverage,
                coverage_scope=state.coverage_scope,
            )
    elif auto_validate_tests:
        state.add_error("--auto-validate-tests requires generate_tests in workflow")

    if run_qa:
        default_report_path = repo_path / "reports" / "qa_report.md"
        report_path = qa_report_path or default_report_path
        state = run_qa_agents(
            state,
            report_path=report_path,
            cov_target=cov_target,
            min_coverage=min_coverage,
            coverage_scope=state.coverage_scope,
        )

    return state


def workflow_to_eval_record(state: WorkflowState, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten WorkflowState into a JSON-serializable row for batch eval logs.

    Pytest capture tails mirror ``state.test_run_report`` (same truncation as QA:
    last ~4000 chars stdout, ~2000 stderr). If the workflow ran pytest multiple times,
    ``test_run_report`` is whatever the final step left on ``state`` (typically the
    last ``run_test_and_coverage`` call).
    """
    from codelyzer.eval.diff_coverage import compute_diff_touch_coverage
    from codelyzer.eval.scorecards import scorecard_dict_for_jsonl

    meta = dict(extra or {})
    require_tests = bool(meta.pop("require_tests_for_e2e", False))
    min_cov = meta.pop("min_coverage_for_e2e", None)
    min_cov_int: int | None = None
    if isinstance(min_cov, int):
        min_cov_int = min_cov
    elif isinstance(min_cov, str) and min_cov.strip().isdigit():
        min_cov_int = int(min_cov.strip())

    cov = state.coverage_report or {}
    test_run = state.test_run_report or {}
    sc = scorecard_dict_for_jsonl(
        state,
        require_tests=require_tests,
        min_coverage=min_cov_int,
    )
    row: dict[str, Any] = {
        "repo_path": str(state.repo_path),
        "base_ref": state.base_ref,
        "target_ref": state.target_ref,
        "pytest_targets": state.pytest_targets,
        "pytest_targets_used": test_run.get("targets"),
        "pytest_return_code": test_run.get("return_code"),
        "pytest_command": test_run.get("command"),
        "pytest_stdout_tail": test_run.get("stdout_tail"),
        "pytest_stderr_tail": test_run.get("stderr_tail"),
        "use_system_python": state.use_system_python,
        "coverage_scope": state.coverage_scope,
        "qa_report_path": state.qa_report_path,
        "files_changed": state.structured_diff.total_files_changed if state.structured_diff else 0,
        "pytest_status": test_run.get("status"),
        "coverage_total_percent": cov.get("total_percent"),
        "final_validation_status": state.final_validation_status,
        "error_count": len(state.errors),
        "errors": list(state.errors),
        "review_findings_count": len(state.review_findings),
        "refinement_fixes_count": len(state.refinement_fixes),
        "removed_tests_count": len(state.removed_tests),
        "research_suggestions_count": len(state.research_suggestions),
        "scorecards": sc,
        "diff_touch_coverage": compute_diff_touch_coverage(state),
    }
    if meta:
        row.update(meta)
    return row
