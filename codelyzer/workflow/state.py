from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
import structlog

from codelyzer.diff.parser import StructuredDiff

logger = structlog.get_logger(__name__)


@dataclass
class WorkflowState:
    """Shared state object passed through workflow graph steps."""

    # Repository context
    repo_path: Path
    base_ref: str
    target_ref: Optional[str] = None
    pytest_targets: Optional[list[str]] = None
    #: When True, QA runs ``sys.executable -m pytest`` (no ``.codelyzer_venv``, no pip install).
    use_system_python: bool = False
    #: Coverage aggregation mode for QA runs.
    #: "runtime" (default), "full_source", or "diff_files" (changed Python files only).
    coverage_scope: Literal["runtime", "full_source", "diff_files"] = "runtime"

    # Intermediate processing data
    raw_diff: Optional[str] = None
    structured_diff: Optional[StructuredDiff] = None
    code_context: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    existing_tests: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Generated artifacts
    pr_summary: Optional[str] = None
    release_notes: Optional[str] = None
    generated_tests: Dict[str, "GeneratedTest"] = field(default_factory=dict)
    test_run_report: Dict[str, Any] = field(default_factory=dict)
    coverage_report: Dict[str, Any] = field(default_factory=dict)
    review_findings: List[Dict[str, Any]] = field(default_factory=list)
    research_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    pre_generation_research: List[Dict[str, Any]] = field(default_factory=list)
    refinement_fixes: List[Dict[str, Any]] = field(default_factory=list)
    removed_tests: List[Dict[str, Any]] = field(default_factory=list)
    final_validation_status: Optional[str] = None
    qa_report_path: Optional[str] = None

    # Status tracking
    errors: List[str] = field(default_factory=list)
    step_results: Dict[str, bool] = field(default_factory=dict)

    def mark_step_complete(self, step_name: str, success: bool = True):
        """Mark workflow step completion status."""
        self.step_results[step_name] = success
        logger.debug(f"Workflow step completed", step=step_name, success=success)

    def add_error(self, message: str):
        """Add error message to state."""
        self.errors.append(message)
        logger.error(message)

    def is_complete(self) -> bool:
        """Check if the currently implemented analysis artifacts are available."""
        return all([
            self.structured_diff is not None,
            self.pr_summary is not None,
        ]) and len(self.errors) == 0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current state."""
        generated_count = sum(
            1
            for result in self.generated_tests.values()
            if result.status in {"generated", "appended"}
        )
        qa_findings_count = len(self.review_findings)
        research_count = len(self.research_suggestions)
        fix_count = len(self.refinement_fixes)
        removed_count = len(self.removed_tests)
        return {
            "repo_path": str(self.repo_path),
            "files_changed": self.structured_diff.total_files_changed if self.structured_diff else 0,
            "context_retrieved": len(self.code_context),
            "tests_found": len(self.existing_tests),
            "tests_generated": generated_count,
            "qa_findings": qa_findings_count,
            "research_suggestions": research_count,
            "fixes_applied": fix_count,
            "tests_removed": removed_count,
            "final_validation_status": self.final_validation_status,
            "qa_report_path": self.qa_report_path,
            "coverage_scope": self.coverage_scope,
            "errors": len(self.errors),
            "complete": self.is_complete()
        }


class PRSummary(BaseModel):
    title: str
    overview: str
    changes_by_file: List[Dict[str, str]]
    impact: str
    breaking_changes: bool
    risk_level: str


class ReleaseNote(BaseModel):
    category: str
    summary: str
    description: str
    issues_resolved: List[str] = []


class GeneratedTest(BaseModel):
    source_file_path: str
    test_file_path: str
    test_code: str
    test_names: List[str]
    status: Literal["generated", "appended", "skipped", "failed"]
    reason: Optional[str] = None
    partial: bool = False
    trace_context: Dict[str, Any] = Field(default_factory=dict)
