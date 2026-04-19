"""Evaluation helpers: scorecards, trajectory logging, batch runs, diff-touch coverage."""

from codelyzer.eval.scorecards import (
    PRSummaryScorecard,
    TestGenerationScorecard,
    compute_e2e_success,
    compute_pr_summary_scorecard,
    compute_test_generation_scorecard,
)
from codelyzer.eval.trajectory import (
    TrajectoryRecorder,
    integrated_validation_core_subsequence,
    trajectory_is_subsequence,
    validation_step_names,
)

__all__ = [
    "PRSummaryScorecard",
    "TestGenerationScorecard",
    "compute_e2e_success",
    "compute_pr_summary_scorecard",
    "compute_test_generation_scorecard",
    "TrajectoryRecorder",
    "integrated_validation_core_subsequence",
    "trajectory_is_subsequence",
    "validation_step_names",
]
