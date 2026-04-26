"""
Format WorkflowState results into a rich GitHub PR comment.

Produces a collapsible, badge-decorated markdown comment that summarises
code-analysis results and, when available, test-generation + QA outcomes.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from codelyzer.workflow.state import WorkflowState


# ── Helpers ──────────────────────────────────────────────────────────────────

def _status_badge(label: str, value: str, colour: str = "blue") -> str:
    """Return a shields.io static badge in markdown."""
    safe_label = label.replace(" ", "%20").replace("-", "--")
    safe_value = value.replace(" ", "%20").replace("-", "--")
    return f"![{label}: {value}](https://img.shields.io/badge/{safe_label}-{safe_value}-{colour})"


def _risk_colour(risk: str) -> str:
    risk_lower = risk.lower()
    if risk_lower in ("low", "none"):
        return "brightgreen"
    elif risk_lower == "medium":
        return "yellow"
    elif risk_lower in ("high", "critical"):
        return "red"
    return "blue"


def _status_emoji(status: str) -> str:
    mapping = {
        "generated": "✅",
        "appended": "📎",
        "skipped": "⏭️",
        "failed": "❌",
        "pass": "✅",
        "fail": "❌",
        "error": "⚠️",
        "unknown": "❓",
    }
    return mapping.get(status.lower(), "•")


def _truncate(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …"


# ── Section renderers ────────────────────────────────────────────────────────

def _render_header(state: WorkflowState) -> str:
    summary = state.get_summary()
    files_changed = summary["files_changed"]
    errors = summary["errors"]

    status = "✅ Analysis Passed" if errors == 0 else f"⚠️ Analysis completed with {errors} error(s)"

    lines = [
        "## 🔬 Codelyzer Analysis Report",
        "",
        f"> {status}",
        "",
        "  ".join([
            _status_badge("files changed", str(files_changed), "blue"),
            _status_badge("errors", str(errors), "brightgreen" if errors == 0 else "red"),
        ]),
        "",
    ]
    return "\n".join(lines)


def _render_diff_stats(state: WorkflowState) -> str:
    diff = state.structured_diff
    if not diff:
        return ""

    lines = [
        "### 📊 Diff Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Files changed | **{diff.total_files_changed}** |",
        f"| Insertions | **+{diff.total_insertions}** |",
        f"| Deletions | **-{diff.total_deletions}** |",
        f"| Base ref | `{diff.base_commit}` |",
        f"| Target ref | `{diff.target_commit}` |",
        "",
    ]
    return "\n".join(lines)


def _render_file_analysis(state: WorkflowState) -> str:
    """Render the per-file LLM analysis into a collapsible section."""
    if not state.pr_summary:
        return ""

    lines = [
        "<details>",
        "<summary><strong>📝 Per-File Analysis</strong> (click to expand)</summary>",
        "",
    ]

    # Parse the pr_summary which has lines like:
    # - path/to/file.py
    #   What changed: ...
    #   Why it was changed: ...
    #   What it does: ...
    current_file = None
    file_details: Dict[str, Dict[str, str]] = {}

    for raw_line in state.pr_summary.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("- ") and not stripped.startswith("- [omitted"):
            current_file = stripped[2:].strip()
            file_details[current_file] = {}
        elif current_file and ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key in ("what changed", "why it was changed", "what it does"):
                file_details[current_file][key] = value

    if file_details:
        lines.append("| File | What Changed | Why | Effect |")
        lines.append("|------|-------------|-----|--------|")
        for fpath, details in file_details.items():
            what = _truncate(details.get("what changed", "—"), 200)
            why = _truncate(details.get("why it was changed", "—"), 200)
            effect = _truncate(details.get("what it does", "—"), 200)
            lines.append(f"| `{fpath}` | {what} | {why} | {effect} |")
        lines.append("")
    else:
        # Fallback: render the raw summary
        lines.append("```")
        lines.append(state.pr_summary)
        lines.append("```")
        lines.append("")

    # Handle omitted files
    for raw_line in state.pr_summary.splitlines():
        if "[omitted" in raw_line:
            lines.append(f"> ℹ️ {raw_line.strip()}")
            lines.append("")

    lines.extend(["</details>", ""])
    return "\n".join(lines)


def _render_test_generation(state: WorkflowState) -> str:
    """Render test-generation outcomes (if any)."""
    if not state.generated_tests:
        return ""

    counts: Dict[str, int] = {"generated": 0, "appended": 0, "skipped": 0, "failed": 0}
    for result in state.generated_tests.values():
        counts[result.status] += 1

    total = len(state.generated_tests)
    success = counts["generated"] + counts["appended"]

    lines = [
        "### 🧪 Test Generation",
        "",
        "  ".join([
            _status_badge("candidates", str(total), "blue"),
            _status_badge("generated", str(counts["generated"]), "brightgreen" if counts["generated"] else "lightgrey"),
            _status_badge("appended", str(counts["appended"]), "blue" if counts["appended"] else "lightgrey"),
            _status_badge("skipped", str(counts["skipped"]), "yellow" if counts["skipped"] else "lightgrey"),
            _status_badge("failed", str(counts["failed"]), "red" if counts["failed"] else "lightgrey"),
        ]),
        "",
        "<details>",
        "<summary>Test generation details</summary>",
        "",
        "| Source File | Status | Test File | Details |",
        "|------------|--------|-----------|---------|",
    ]

    for result in state.generated_tests.values():
        emoji = _status_emoji(result.status)
        test_dest = f"`{result.test_file_path}`" if result.test_file_path else "—"
        detail = _truncate(result.reason or "", 150)
        partial_tag = " 🟡 partial" if result.partial else ""
        lines.append(
            f"| `{result.source_file_path}` | {emoji} {result.status}{partial_tag} | {test_dest} | {detail} |"
        )

    lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def _render_qa_report(state: WorkflowState) -> str:
    """Render QA agent findings if available."""
    test_run = state.test_run_report
    coverage = state.coverage_report

    if not test_run and not coverage:
        return ""

    lines = ["### 🛡️ QA Report", ""]

    # Test run results
    if test_run and isinstance(test_run, dict):
        run_status = test_run.get("status", "unknown")
        counts = test_run.get("counts", {})
        emoji = _status_emoji(run_status)
        lines.append(f"**Test Run:** {emoji} {run_status}")
        if counts:
            parts = [f"{k}: {v}" for k, v in counts.items()]
            lines.append(f"  Counts: {', '.join(parts)}")
        lines.append("")

    # Coverage results
    if coverage and isinstance(coverage, dict):
        cov_status = coverage.get("status", "unknown")
        total_pct = coverage.get("total_percent", "n/a")
        lines.extend([
            f"**Coverage:** {total_pct}%",
            "",
        ])

    # Review findings
    if state.review_findings:
        lines.extend([
            f"**Review Findings:** {len(state.review_findings)} issue(s) flagged",
            "",
        ])

    # Refinement
    if state.refinement_fixes:
        lines.append(f"**Auto-fixes Applied:** {len(state.refinement_fixes)}")
        lines.append("")

    if state.removed_tests:
        lines.append(f"**Tests Removed (failing/invalid):** {len(state.removed_tests)}")
        lines.append("")

    if state.final_validation_status:
        emoji = _status_emoji(state.final_validation_status)
        lines.append(f"**Final Validation:** {emoji} {state.final_validation_status}")
        lines.append("")

    return "\n".join(lines)


def _render_context_summary(state: WorkflowState) -> str:
    """Render retrieval/indexing context info."""
    summary = state.get_summary()
    context_count = summary["context_retrieved"]
    tests_found = summary["tests_found"]

    if context_count == 0 and tests_found == 0:
        return ""

    lines = [
        "<details>",
        "<summary><strong>📚 Retrieval Context</strong></summary>",
        "",
        f"- **Code context chunks retrieved:** {context_count}",
        f"- **Existing test files found:** {tests_found}",
    ]

    if state.research_suggestions:
        lines.append(f"- **Research suggestions:** {len(state.research_suggestions)}")

    lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def _render_errors(state: WorkflowState) -> str:
    """Render errors if any."""
    if not state.errors:
        return ""

    lines = [
        "<details>",
        "<summary><strong>⚠️ Errors & Warnings</strong> ({count})</summary>".format(count=len(state.errors)),
        "",
    ]
    for i, err in enumerate(state.errors, 1):
        lines.append(f"{i}. {_truncate(err, 300)}")

    lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def _render_footer() -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        "\n---\n"
        f"<sub>🤖 Generated by **Codelyzer** at {now} · "
        "[docs](https://github.com/vijay-murugan/codelyzer)</sub>\n"
    )


# ── Public API ───────────────────────────────────────────────────────────────

def format_pr_comment(
    state: WorkflowState,
    *,
    include_tests: bool = True,
    include_qa: bool = True,
) -> str:
    """
    Render a complete GitHub PR comment from a WorkflowState.

    Parameters
    ----------
    state : WorkflowState
        Completed analysis state.
    include_tests : bool
        Include test-generation section (if data is present).
    include_qa : bool
        Include QA report section (if data is present).

    Returns
    -------
    str
        GitHub-flavoured Markdown ready to post as a PR comment.
    """
    sections = [
        _render_header(state),
        _render_diff_stats(state),
        _render_file_analysis(state),
    ]

    if include_tests:
        sections.append(_render_test_generation(state))

    if include_qa:
        sections.append(_render_qa_report(state))

    sections.extend([
        _render_context_summary(state),
        _render_errors(state),
        _render_footer(),
    ])

    # Remove empty sections and join
    return "\n".join(s for s in sections if s)
