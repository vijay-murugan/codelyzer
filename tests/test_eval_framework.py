"""Tests for eval scorecards, trajectory, diff-touch coverage, and single-step QA/summary behavior."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from codelyzer.analysis_runner import workflow_to_eval_record
from codelyzer.changed_files_summary import summarize_file_with_llm
from codelyzer.diff.parser import DiffHunk, FileDiff, StructuredDiff
from codelyzer.eval.diff_coverage import (
    collect_diff_touch_lines,
    compute_diff_touch_coverage,
)
from codelyzer.eval.scorecards import (
    compute_e2e_success,
    compute_pr_summary_scorecard,
    compute_test_generation_scorecard,
)
from codelyzer.eval.trajectory import (
    TrajectoryRecorder,
    integrated_validation_core_subsequence,
    trajectory_is_subsequence,
)
from codelyzer.testing.qa_agents import review_generated_tests_agent, run_integrated_generation_validation
from codelyzer.workflow.state import GeneratedTest, WorkflowState


def test_pr_summary_scorecard_detects_fallback_phrase() -> None:
    state = WorkflowState(repo_path=Path("."), base_ref="HEAD", target_ref=None)
    state.pr_summary = "What changed: x\nCould not generate LLM rationale for this file (bad)."
    sc = compute_pr_summary_scorecard(state)
    assert sc.fallback_phrase_hits >= 1


def test_test_generation_scorecard_counts_statuses() -> None:
    state = WorkflowState(repo_path=Path("."), base_ref="HEAD", target_ref=None)
    state.generated_tests["a.py"] = GeneratedTest(
        source_file_path="a.py",
        test_file_path="tests/test_a.py",
        test_code="def test_x(): assert 1",
        test_names=["test_x"],
        status="generated",
        partial=True,
    )
    state.test_run_report = {"status": "passed"}
    state.coverage_report = {"total_percent": 88}
    state.final_validation_status = "passed"
    sc = compute_test_generation_scorecard(state)
    assert sc.generated == 1 and sc.partial_count == 1 and sc.pytest_passed is True


def test_e2e_success_respects_require_tests() -> None:
    state = WorkflowState(repo_path=Path("."), base_ref="HEAD", target_ref=None)
    state.pr_summary = "ok summary without fallback phrase"
    state.test_run_report = {"status": "failed"}
    loose = compute_e2e_success(state, require_tests=False)
    assert loose.tests_green is True
    strict = compute_e2e_success(state, require_tests=True)
    assert strict.tests_green is False and strict.overall is False


def test_trajectory_subsequence() -> None:
    rec = [
        {"step": "review_generated_tests_agent"},
        {"step": "noise"},
        {"step": "auto_fix_generated_tests_agent"},
        {"step": "run_test_and_coverage"},
    ]
    assert trajectory_is_subsequence(rec, ["review_generated_tests_agent", "run_test_and_coverage"])
    assert not trajectory_is_subsequence(rec, ["run_test_and_coverage", "review_generated_tests_agent"])


def test_integrated_core_subsequence_matches_recorded_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run_test(state: WorkflowState, **kwargs: object) -> WorkflowState:
        state.test_run_report = {"status": "passed"}
        state.coverage_report = {
            "status": "available",
            "total_percent": 90,
            "low_coverage_files": [],
        }
        state.final_validation_status = "passed"
        return state

    monkeypatch.setattr("codelyzer.testing.qa_agents.run_test_and_coverage", fake_run_test)
    monkeypatch.setattr("codelyzer.testing.qa_agents._get_llm_client_safe", lambda: None)

    state = WorkflowState(repo_path=tmp_path, base_ref="HEAD", target_ref=None)
    state.generated_tests["m.py"] = GeneratedTest(
        source_file_path="m.py",
        test_file_path=str(tmp_path / "tests" / "test_m.py"),
        test_code="def test_ok():\n    assert 1\n",
        test_names=["test_ok"],
        status="generated",
        partial=False,
    )
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "test_m.py").write_text("def test_ok():\n    assert 1\n", encoding="utf-8")
    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")

    report = tmp_path / "qa_report.md"
    with TrajectoryRecorder() as log:
        run_integrated_generation_validation(state, report_path=report, cov_target="codelyzer")

    assert trajectory_is_subsequence(log, integrated_validation_core_subsequence())


def test_collect_diff_touch_lines_counts_additions() -> None:
    hunk = DiffHunk(
        start_line_old=1,
        start_line_new=1,
        lines_old=1,
        lines_new=2,
        content=" line0\n-old\n+new\n",
    )
    fd = FileDiff(file_path=Path("pkg/mod.py"), change_type="modified", hunks=[hunk])
    diff = StructuredDiff(
        base_commit="a",
        target_commit="b",
        total_files_changed=1,
        total_insertions=1,
        total_deletions=1,
        files=[fd],
    )
    m = collect_diff_touch_lines(diff)
    assert "pkg/mod.py" in m
    assert m["pkg/mod.py"] == {2}


def test_compute_diff_touch_coverage_matches_xml(tmp_path: Path) -> None:
    xml = """<?xml version="1.0" ?>
<coverage version="7.0">
  <packages>
    <package>
      <classes>
        <class filename="src/x.py" line-rate="1" branch-rate="1">
          <lines>
            <line number="2" hits="1"/>
            <line number="3" hits="0"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""
    (tmp_path / "coverage.xml").write_text(xml, encoding="utf-8")
    hunk = DiffHunk(
        start_line_old=1,
        start_line_new=1,
        lines_old=1,
        lines_new=2,
        content=" a\n+b\n+c\n",
    )
    fd = FileDiff(file_path=Path("src/x.py"), change_type="modified", hunks=[hunk])
    diff = StructuredDiff(
        base_commit="a",
        target_commit="b",
        total_files_changed=1,
        total_insertions=2,
        total_deletions=0,
        files=[fd],
    )
    state = WorkflowState(repo_path=tmp_path, base_ref="HEAD", target_ref=None)
    state.structured_diff = diff
    out = compute_diff_touch_coverage(state)
    assert out["status"] == "ok"
    assert out["touch_lines"] == 2
    assert out["covered_lines"] == 1
    assert out["percent"] == 50.0


def test_summarize_file_with_llm_parses_three_lines() -> None:
    class StubLLM:
        def generate_text(self, prompt, input_vars):  # type: ignore[no-untyped-def]
            return (
                "What changed: Added guard.\n"
                "Why it was changed: Prevent crash.\n"
                "What it does: Returns early on bad input.\n"
            )

    fd = FileDiff(
        file_path=Path("f.py"),
        change_type="modified",
        hunks=[
            DiffHunk(
                start_line_old=1,
                start_line_new=1,
                lines_old=1,
                lines_new=1,
                content=" a\n",
            )
        ],
    )
    w, y, z = summarize_file_with_llm(StubLLM(), fd, 5000)
    assert "guard" in w.lower() and "crash" in y.lower()


def test_review_generated_tests_agent_flags_failed_generation() -> None:
    state = WorkflowState(repo_path=Path("."), base_ref="HEAD", target_ref=None)
    state.generated_tests["z.py"] = GeneratedTest(
        source_file_path="z.py",
        test_file_path="",
        test_code="",
        test_names=[],
        status="failed",
        reason="boom",
        partial=False,
    )
    review_generated_tests_agent(state)
    assert any("boom" in str(f.get("message", "")) for f in state.review_findings)


def test_workflow_to_eval_record_roundtrip_keys() -> None:
    state = WorkflowState(repo_path=Path("/tmp"), base_ref="HEAD", target_ref=None)
    state.structured_diff = StructuredDiff(
        base_commit="a",
        target_commit="b",
        total_files_changed=0,
        total_insertions=0,
        total_deletions=0,
        files=[],
    )
    state.pr_summary = "ok"
    row = workflow_to_eval_record(state, extra={"case_label": "t"})
    assert row["case_label"] == "t"
    assert "scorecards" in row and "diff_touch_coverage" in row
    assert row.get("pytest_targets") is None
    assert row.get("use_system_python") is False
    assert row.get("coverage_scope") == "runtime"

    state.pytest_targets = ["app/tests"]
    row2 = workflow_to_eval_record(state, extra={"case_label": "t2"})
    assert row2["pytest_targets"] == ["app/tests"]
    state.coverage_scope = "full_source"
    row2b = workflow_to_eval_record(state, extra={"case_label": "t2b"})
    assert row2b["coverage_scope"] == "full_source"

    state.test_run_report = {
        "status": "failed",
        "return_code": 1,
        "command": ["python", "-m", "pytest", "tests"],
        "targets": ["tests"],
        "stdout_tail": "collected 1 item\nFAILED tests/test_x.py\n",
        "stderr_tail": "ImportError: no module named foo\n",
    }
    row3 = workflow_to_eval_record(state, extra={"case_label": "t3"})
    assert row3["pytest_stdout_tail"] == "collected 1 item\nFAILED tests/test_x.py\n"
    assert row3["pytest_stderr_tail"] == "ImportError: no module named foo\n"


def test_eval_batch_cli_writes_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from codelyzer.eval import batch as batch_mod
    seen_kwargs: list[dict[str, object]] = []

    def fake_workflow(repo_path: Path, **kwargs: object) -> WorkflowState:
        seen_kwargs.append(kwargs)
        s = WorkflowState(repo_path=tmp_path, base_ref="HEAD", target_ref=None)
        s.structured_diff = StructuredDiff(
            base_commit="a",
            target_commit="b",
            total_files_changed=0,
            total_insertions=0,
            total_deletions=0,
            files=[],
        )
        s.pr_summary = "summary"
        return s

    monkeypatch.setattr(batch_mod, "run_analyze_workflow", fake_workflow)

    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps([{"repo_path": str(tmp_path), "base": "HEAD", "label": "unit"}]),
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(batch_mod.eval_cli, ["batch", str(cfg), "--output", str(out)])
    assert result.exit_code == 0, result.output
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["case_label"] == "unit"
    assert "error" not in row
    assert "scorecards" in row
    assert seen_kwargs and seen_kwargs[0].get("coverage_scope") == "runtime"


def test_eval_batch_cli_passes_full_source_coverage_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from codelyzer.eval import batch as batch_mod

    seen_kwargs: list[dict[str, object]] = []

    def fake_workflow(repo_path: Path, **kwargs: object) -> WorkflowState:
        seen_kwargs.append(kwargs)
        s = WorkflowState(repo_path=tmp_path, base_ref="HEAD", target_ref=None)
        s.structured_diff = StructuredDiff(
            base_commit="a",
            target_commit="b",
            total_files_changed=0,
            total_insertions=0,
            total_deletions=0,
            files=[],
        )
        s.pr_summary = "summary"
        return s

    monkeypatch.setattr(batch_mod, "run_analyze_workflow", fake_workflow)

    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps([{"repo_path": str(tmp_path), "base": "HEAD", "coverage_scope": "full_source"}]),
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(batch_mod.eval_cli, ["batch", str(cfg), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert seen_kwargs and seen_kwargs[0].get("coverage_scope") == "full_source"


def test_eval_batch_cli_verbose_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from codelyzer.eval import batch as batch_mod

    def fake_workflow(repo_path: Path, **kwargs: object) -> WorkflowState:
        s = WorkflowState(repo_path=tmp_path, base_ref="HEAD", target_ref=None)
        s.structured_diff = StructuredDiff(
            base_commit="a",
            target_commit="b",
            total_files_changed=0,
            total_insertions=0,
            total_deletions=0,
            files=[],
        )
        s.pr_summary = "summary"
        s.test_run_report = {
            "status": "passed",
            "return_code": 0,
            "targets": ["app/tests"],
            "command": ["/x/python", "-m", "pytest", "app/tests"],
        }
        s.qa_report_path = str(tmp_path / "reports" / "qa.md")
        return s

    monkeypatch.setattr(batch_mod, "run_analyze_workflow", fake_workflow)

    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps([{"repo_path": str(tmp_path), "base": "HEAD", "label": "v"}]),
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(batch_mod.eval_cli, ["batch", str(cfg), "--output", str(out), "--verbose"])
    assert result.exit_code == 0, result.output
    assert "pytest command:" in result.output
    assert "targets_used=" in result.output


def test_human_rubric_paths_exist() -> None:
    from codelyzer.eval.human_rubric import EXAMPLE_LABELS_PATH, SCHEMA_PATH

    assert SCHEMA_PATH.is_file()
    assert EXAMPLE_LABELS_PATH.is_file()
