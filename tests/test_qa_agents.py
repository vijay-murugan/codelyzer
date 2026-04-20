from pathlib import Path
import pytest

from codelyzer.testing.qa_agents import (
    QARunArtifacts,
    _detect_dependency_file,
    _detect_pythonpath_override,
    _pytest_target_paths,
    _remove_vacuous_tests_from_content,
    run_test_and_coverage,
    parse_coverage_output,
    parse_test_runner_output,
    write_qa_markdown_report,
)
from codelyzer.workflow.state import GeneratedTest, WorkflowState
from codelyzer.diff.parser import StructuredDiff, FileDiff


def _state(tmp_path: Path) -> WorkflowState:
    return WorkflowState(repo_path=tmp_path, base_ref="HEAD", target_ref=None)


def test_pytest_target_paths_default_tests(tmp_path: Path) -> None:
    state = WorkflowState(repo_path=tmp_path, base_ref="HEAD", target_ref=None)
    assert _pytest_target_paths(state) == ["tests"]


def test_pytest_target_paths_uses_configured_when_no_generated(tmp_path: Path) -> None:
    state = WorkflowState(
        repo_path=tmp_path,
        base_ref="HEAD",
        target_ref=None,
        pytest_targets=["app/tests", "  "],
    )
    assert _pytest_target_paths(state) == ["app/tests"]


def test_pytest_target_paths_generated_wins_over_configured(tmp_path: Path) -> None:
    state = WorkflowState(
        repo_path=tmp_path,
        base_ref="HEAD",
        target_ref=None,
        pytest_targets=["app/tests"],
    )
    state.generated_tests["m.py"] = GeneratedTest(
        source_file_path="m.py",
        test_file_path=str(tmp_path / "tests" / "test_m.py"),
        test_code="def test_ok():\n    assert 1\n",
        test_names=["test_ok"],
        status="generated",
        partial=False,
    )
    assert _pytest_target_paths(state) == [str(tmp_path / "tests" / "test_m.py")]


def test_remove_vacuous_tests_from_content() -> None:
    content = """
def test_ok():
    assert 1 == 1

def test_bad():
    pass

def test_weak():
    x = 1
"""
    fixed, removed = _remove_vacuous_tests_from_content(content)
    assert "test_bad" in removed
    assert "test_weak" in removed
    assert "def test_ok" in fixed
    assert "def test_bad" not in fixed


def test_parse_test_runner_output_collects_failure_sections() -> None:
    artifacts = QARunArtifacts(
        command=["pytest"],
        stdout=(
            "collected 3 items\n"
            "== FAILURES ==\n"
            "__ test_a __\nAssertionError\n"
            "== 1 failed, 2 passed in 0.10s ==\n"
        ),
        stderr="",
        return_code=1,
    )
    report = parse_test_runner_output(artifacts)
    assert report["status"] == "failed"
    assert report["collected"] == 3
    assert report["failures"]


def test_parse_coverage_output_from_terminal() -> None:
    artifacts = QARunArtifacts(
        command=["pytest"],
        stdout=(
            "Name Stmts Miss Cover\n"
            "codelyzer/cli.py 10 2 80%\n"
            "TOTAL 10 2 80%\n"
        ),
        stderr="",
        return_code=0,
    )
    coverage = parse_coverage_output(artifacts, Path("."))
    assert coverage["status"] == "available"
    assert coverage["total_percent"] == 80
    assert coverage["files"][0]["file"] == "codelyzer/cli.py"


def test_write_qa_markdown_report_includes_refinement_sections(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.generated_tests["module.py"] = GeneratedTest(
        source_file_path="module.py",
        test_file_path=str(tmp_path / "tests" / "test_module.py"),
        test_code="def test_ok():\n    assert True\n",
        test_names=["test_ok"],
        status="generated",
        partial=False,
    )
    state.test_run_report = {
        "status": "passed",
        "return_code": 0,
        "collected": 1,
        "counts": {"passed": 1},
        "bootstrap": {
            "status": "ok",
            "venv_path": str(tmp_path / ".codelyzer_venv"),
            "system_python": False,
            "dependency_file": str(tmp_path / "backend" / "requirements-dev.txt"),
            "pythonpath_override": ".",
        },
    }
    state.coverage_report = {"status": "available", "total_percent": 75, "low_coverage_files": []}
    state.pre_generation_research = [
        {"priority": "high", "target_file": "module.py", "suggested_test_name": "test_module_edge", "rationale": "edge"}
    ]
    state.refinement_fixes = [
        {"action": "remove_vacuous_tests", "test_file": "tests/test_module.py", "tests": ["test_bad"], "reason": "no asserts"}
    ]
    state.removed_tests = [
        {"test_file": "tests/test_old.py", "tests": ["test_old"], "reason": "still invalid"}
    ]
    state.final_validation_status = "passed"

    report_path = tmp_path / "reports" / "qa_report.md"
    write_qa_markdown_report(state, report_path)
    text = report_path.read_text(encoding="utf-8")
    assert "## Pre-Generation Research Used" in text
    assert "## Fixes Applied" in text
    assert "## Removed Tests" in text
    assert "## Generated test review" in text
    assert "### Test Environment Bootstrap" in text
    assert "System Python (no venv): False" in text
    assert "PYTHONPATH override: ." in text


def test_detect_dependency_file_prefers_backend_requirements(tmp_path: Path) -> None:
    backend = tmp_path / "backend"
    backend.mkdir(parents=True, exist_ok=True)
    req = backend / "requirements-dev.txt"
    req.write_text("pytest\n", encoding="utf-8")
    assert _detect_dependency_file(tmp_path) == req


def test_detect_pythonpath_override_for_app_layout(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir(parents=True, exist_ok=True)
    assert _detect_pythonpath_override(tmp_path) == "."


def test_detect_pythonpath_override_for_src_layout(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    assert _detect_pythonpath_override(tmp_path) == "src"


def test_detect_pythonpath_override_none_without_app_or_src_dir(tmp_path: Path) -> None:
    assert _detect_pythonpath_override(tmp_path) is None


def test_bootstrap_system_python_uses_sys_executable(tmp_path: Path) -> None:
    from codelyzer.testing import qa_agents as qa

    info = qa._bootstrap_test_environment(tmp_path, 120, use_system_python=True)
    assert info["status"] == "ok"
    assert info["system_python"] is True
    assert info["venv_python"] == __import__("sys").executable
    assert info["install_command"] == []


def test_run_test_and_coverage_runtime_has_no_cov_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from codelyzer.testing import qa_agents as qa

    state = WorkflowState(repo_path=tmp_path, base_ref="HEAD", target_ref=None, coverage_scope="runtime")
    seen_cmd: list[str] = []

    def fake_bootstrap(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "status": "ok",
            "venv_python": "python3",
            "pythonpath_override": None,
            "system_python": False,
        }

    def fake_run(cmd: list[str], **kwargs: object):  # type: ignore[no-untyped-def]
        seen_cmd[:] = cmd
        return __import__("subprocess").CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="TOTAL 1 0 100%\n",
            stderr="",
        )

    monkeypatch.setattr(qa, "_bootstrap_test_environment", fake_bootstrap)
    monkeypatch.setattr(qa.subprocess, "run", fake_run)

    out = run_test_and_coverage(state, cov_target="app", coverage_scope="runtime")
    assert out.test_run_report.get("status") == "passed"
    assert "--cov=app" in seen_cmd
    assert not any(str(arg).startswith("--cov-config=") for arg in seen_cmd)


def test_run_test_and_coverage_full_source_adds_cov_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from codelyzer.testing import qa_agents as qa

    state = WorkflowState(repo_path=tmp_path, base_ref="HEAD", target_ref=None, coverage_scope="full_source")
    seen_cmd: list[str] = []

    def fake_bootstrap(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "status": "ok",
            "venv_python": "python3",
            "pythonpath_override": None,
            "system_python": False,
        }

    def fake_run(cmd: list[str], **kwargs: object):  # type: ignore[no-untyped-def]
        seen_cmd[:] = cmd
        return __import__("subprocess").CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="TOTAL 1 0 100%\n",
            stderr="",
        )

    monkeypatch.setattr(qa, "_bootstrap_test_environment", fake_bootstrap)
    monkeypatch.setattr(qa.subprocess, "run", fake_run)

    out = run_test_and_coverage(state, cov_target="app", coverage_scope="full_source")
    assert out.test_run_report.get("status") == "passed"
    assert "--cov" in seen_cmd
    assert "--cov=app" not in seen_cmd
    assert any(str(arg).startswith("--cov-config=") for arg in seen_cmd)


def test_run_test_and_coverage_diff_files_adds_cov_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from codelyzer.testing import qa_agents as qa

    state = WorkflowState(repo_path=tmp_path, base_ref="HEAD", target_ref=None, coverage_scope="diff_files")
    state.structured_diff = StructuredDiff(
        base_commit="a",
        target_commit="b",
        total_files_changed=1,
        total_insertions=1,
        total_deletions=0,
        files=[FileDiff(file_path=Path("src/mod.py"), change_type="modified", hunks=[])],
    )
    seen_cmd: list[str] = []
    seen_cov_cfg_text: dict[str, str] = {}

    def fake_bootstrap(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "status": "ok",
            "venv_python": "python3",
            "pythonpath_override": None,
            "system_python": False,
        }

    def fake_run(cmd: list[str], **kwargs: object):  # type: ignore[no-untyped-def]
        seen_cmd[:] = cmd
        cov_cfg = next((str(arg) for arg in cmd if str(arg).startswith("--cov-config=")), "")
        if cov_cfg:
            cfg_path = Path(cov_cfg.split("=", 1)[1])
            seen_cov_cfg_text["text"] = cfg_path.read_text(encoding="utf-8")
        return __import__("subprocess").CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="TOTAL 1 0 100%\n",
            stderr="",
        )

    monkeypatch.setattr(qa, "_bootstrap_test_environment", fake_bootstrap)
    monkeypatch.setattr(qa.subprocess, "run", fake_run)

    out = run_test_and_coverage(state, cov_target="app", coverage_scope="diff_files")
    assert out.test_run_report.get("status") == "passed"
    assert "--cov" in seen_cmd
    assert "--cov=app" not in seen_cmd
    cfg_text = seen_cov_cfg_text.get("text", "")
    assert "include =" in cfg_text
    assert str((tmp_path / "src/mod.py").resolve()) in cfg_text
