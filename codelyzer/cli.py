from pathlib import Path

from dotenv import load_dotenv

# Load .env before any `codelyzer` import — `codelyzer.config` builds `settings` at import time.
_repo_root = Path(__file__).resolve().parent.parent
load_dotenv(_repo_root / ".env", override=False)
load_dotenv(Path.cwd() / ".env", override=True)

import click
import structlog

from codelyzer import __version__
from codelyzer.changed_files_summary import render_changed_files_summary
from codelyzer.diff.parser import GitDiffParser
from codelyzer.retrieval.indexer import RepositoryIndexer
from codelyzer.testing.generator import generate_tests_for_changes, summarize_test_generation_preflight
from codelyzer.testing.qa_agents import (
    research_before_generation_agent,
    run_integrated_generation_validation,
    run_qa_agents,
)
from codelyzer.eval.batch import eval_cli
from codelyzer.workflow.state import WorkflowState

logger = structlog.get_logger(__name__)


def _render_test_generation_report(state: WorkflowState) -> str:
    if not state.generated_tests:
        return "No test generation results available."

    counts = {
        "generated": 0,
        "appended": 0,
        "skipped": 0,
        "failed": 0,
    }
    lines: list[str] = []
    for result in state.generated_tests.values():
        counts[result.status] += 1
        destination = f" -> {result.test_file_path}" if result.test_file_path else ""
        detail = f" ({result.reason})" if result.reason else ""
        partial = " [partial]" if result.partial else ""
        note = ""
        if (
            result.status == "skipped"
            and result.reason
            and "already exist" in result.reason
            and result.test_file_path
        ):
            note = " [no file change]"
        lines.append(
            f"- {result.source_file_path}: {result.status}{partial}{destination}{detail}{note}"
        )

    header = (
        f"Candidates: {len(state.generated_tests)} | "
        f"generated: {counts['generated']} | "
        f"appended: {counts['appended']} | "
        f"skipped: {counts['skipped']} | "
        f"failed: {counts['failed']}"
    )
    return "\n".join([header, *lines])


def _render_qa_report_summary(state: WorkflowState) -> str:
    test_run = state.test_run_report or {}
    coverage = state.coverage_report or {}
    bootstrap = test_run.get("bootstrap", {}) if isinstance(test_run, dict) else {}
    lines = [
        f"- Test run status: {test_run.get('status', 'unknown')}",
        f"- Test counts: {test_run.get('counts', {})}",
        f"- Bootstrap status: {bootstrap.get('status', 'not-run')}",
        f"- Bootstrap deps: {bootstrap.get('dependency_file', 'none')}",
        f"- Bootstrap PYTHONPATH: {bootstrap.get('pythonpath_override', 'none')}",
        f"- System Python (no venv): {bootstrap.get('system_python', False)}",
        f"- Coverage scope: {state.coverage_scope}",
        f"- Coverage status: {coverage.get('status', 'unknown')}",
        f"- Coverage total: {coverage.get('total_percent', 'n/a')}%",
        f"- Review findings: {len(state.review_findings)}",
        f"- Research suggestions: {len(state.research_suggestions)}",
        f"- Fixes applied: {len(state.refinement_fixes)}",
        f"- Tests removed: {len(state.removed_tests)}",
        f"- Final validation: {state.final_validation_status or 'unknown'}",
        f"- QA report: {state.qa_report_path or 'not written'}",
    ]
    return "\n".join(lines)


@click.group()
@click.version_option(version=__version__)
def cli():
    """
    Codelyzer - Intelligent code analysis for git repositories.

    Analyze git diffs, generate PR summaries, release notes and automated tests.
    Works with any local git repository without modifications.
    """
    pass


cli.add_command(eval_cli)


@cli.command()
@click.argument('repo_path', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--base', default='HEAD', help='Base git reference (default: HEAD)')
@click.option('--target', help='Target git reference (default: working copy)')
@click.option('--generate-tests', is_flag=True, help='Generate pytest tests for changed Python source files')
@click.option('--run-qa', is_flag=True, help='Run tests + coverage, pytest failure notes, post-run research, QA report (no LLM review of source)')
@click.option('--auto-validate-tests', is_flag=True, help='With --generate-tests: after generation, LLM-review generated tests, auto-fix, pytest+coverage, prune if needed, research, report')
@click.option('--qa-report-path', type=click.Path(path_type=Path), help='Markdown output path for QA report')
@click.option('--cov-target', default='codelyzer', help='Coverage target passed to pytest --cov')
@click.option('--min-coverage', type=int, help='Optional minimum required total coverage percent')
@click.option(
    '--coverage-scope',
    type=click.Choice(['runtime', 'full_source', 'diff_files'], case_sensitive=False),
    default='runtime',
    show_default=True,
    help='Coverage accounting mode: runtime, full_source, or diff_files (changed Python files only).',
)
@click.option(
    '--pytest-target',
    multiple=True,
    help='Path(s) for pytest when no generated test files define the run set (repeatable); default is tests.',
)
@click.option(
    '--qa-system-python',
    is_flag=True,
    help='Run pytest with the current Python (sys.executable); skip .codelyzer_venv and pip install.',
)
def analyze(
    repo_path: Path,
    base: str,
    target: str | None,
    generate_tests: bool,
    run_qa: bool,
    auto_validate_tests: bool,
    qa_report_path: Path | None,
    cov_target: str,
    min_coverage: int | None,
    coverage_scope: str,
    pytest_target: tuple[str, ...],
    qa_system_python: bool,
):
    """Analyze repository changes and generate insights."""
    click.echo(f"🔍 Analyzing repository: {repo_path}")
    click.echo(f"📌 Base ref: {base}" + (f" → Target ref: {target}" if target else ""))

    pytest_targets = [p.strip() for p in pytest_target if p.strip()]
    # Initialize workflow state
    state = WorkflowState(
        repo_path=repo_path,
        base_ref=base,
        target_ref=target,
        pytest_targets=pytest_targets if pytest_targets else None,
        use_system_python=qa_system_python,
        coverage_scope=(
            "full_source"
            if str(coverage_scope).lower() == "full_source"
            else ("diff_files" if str(coverage_scope).lower() == "diff_files" else "runtime")
        ),
    )

    # Parse git diff
    try:
        parser = GitDiffParser(repo_path)
        state.structured_diff = parser.parse_diff(base, target)
        state.pr_summary = render_changed_files_summary(state.structured_diff)

        click.echo(f"✅ Parsed diff: {state.structured_diff.total_files_changed} files changed, "
                   f"+{state.structured_diff.total_insertions} -{state.structured_diff.total_deletions}")
    except Exception as e:
        click.echo(f"❌ Failed to parse diff: {e}", err=True)
        return 1

    indexer: RepositoryIndexer | None = None

    # Index repository
    try:
        click.echo("\n📚 Indexing repository code...")
        indexer = RepositoryIndexer(repo_path)
        chunk_count = indexer.index_repository()
        if indexer.reused_existing_index:
            click.echo(f"✅ Reused existing repository index: {chunk_count} vectors")
        else:
            click.echo(f"✅ Indexed repository: {chunk_count} code chunks")
    except Exception as e:
        click.echo(f"⚠️  Indexing warning: {e}")
        state.add_error(f"Indexing warning: {e}")

    # Find and store existing tests for each changed file
    try:
        if indexer is not None:
            for file_diff in (state.structured_diff.files if state.structured_diff else []):
                tests = indexer.search_tests_for_file(file_diff.file_path)
                if tests:
                    state.existing_tests[str(file_diff.file_path)] = tests
    except Exception as e:
        click.echo(f"⚠️  Test search warning: {e}")
        state.add_error(f"Test search warning: {e}")

    if generate_tests:
        preflight = summarize_test_generation_preflight(state.structured_diff)
        if preflight["changed_files"] == 0:
            click.echo(
                "\n⚠️  No git diff: nothing changed between the base ref and the target "
                "(or the working tree matches HEAD). No test files will be created.\n"
                "   Try: make local edits, or compare branches, e.g.\n"
                "   codelyzer analyze <repo> --base main --target <branch> --generate-tests"
            )
        elif preflight["eligible"] == 0:
            click.echo(
                f"\n⚠️  No eligible Python sources for test generation "
                f"({preflight['changed_files']} file(s) in diff; all skipped). Reasons:"
            )
            for reason, count in sorted(
                preflight["skip_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            ):
                click.echo(f"   - {count}x: {reason}")
            click.echo(
                "   Only non-test .py files generate tests. Use a diff that includes them, or adjust paths."
            )
        if auto_validate_tests:
            state = research_before_generation_agent(state)
        click.echo("\n🧪 Generating unit tests...")
        state = generate_tests_for_changes(state, indexer=indexer)
        click.echo(_render_test_generation_report(state))
        if auto_validate_tests:
            click.echo("\n🛡️ Auto-validating generated tests (review generated tests → fix → pytest + coverage → prune if needed → research + report)...")
            default_report_path = repo_path / "reports" / "qa_report.md"
            report_path = qa_report_path or default_report_path
            state = run_integrated_generation_validation(
                state,
                report_path=report_path,
                cov_target=cov_target,
                min_coverage=min_coverage,
                coverage_scope=state.coverage_scope,
            )
            click.echo(_render_qa_report_summary(state))
    elif auto_validate_tests:
        click.echo("⚠️  --auto-validate-tests requires --generate-tests; skipping auto-validation.")

    if run_qa:
        click.echo("\n🛡️ Running QA agents (tests + coverage + failure notes + research + report)...")
        default_report_path = repo_path / "reports" / "qa_report.md"
        report_path = qa_report_path or default_report_path
        state = run_qa_agents(
            state,
            report_path=report_path,
            cov_target=cov_target,
            min_coverage=min_coverage,
            coverage_scope=state.coverage_scope,
        )
        click.echo(_render_qa_report_summary(state))

    summary = state.get_summary()
    click.echo("\n📝 Changed files summary")
    click.echo(state.pr_summary)

    click.echo("\n📊 Analysis summary")
    click.echo(f"- Repository: {summary['repo_path']}")
    click.echo(f"- Files changed: {summary['files_changed']}")
    click.echo(f"- Context retrieved: {summary['context_retrieved']}")
    click.echo(f"- Existing tests found: {summary['tests_found']}")
    click.echo(f"- Tests generated: {summary['tests_generated']}")
    click.echo(f"- Errors: {summary['errors']}")
    click.echo(f"- Workflow complete: {summary['complete']}")

    click.echo("\n🎉 Analysis complete!")
    return 0


@cli.command()
@click.argument('repo_path', type=click.Path(exists=True, file_okay=False, path_type=Path))
def index(repo_path: Path):
    """Build semantic index for repository."""
    click.echo(f"📚 Building index for: {repo_path}")

    indexer = RepositoryIndexer(repo_path)
    chunks = indexer.index_repository()
    if indexer.reused_existing_index:
        click.echo(f"✅ Index already available: {chunks} vectors")
    else:
        click.echo(f"✅ Index complete: {chunks} code chunks stored")
    return 0


def main():
    """CLI entry point."""
    try:
        return cli(standalone_mode=False)
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    exit(main())
