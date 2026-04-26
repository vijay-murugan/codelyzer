import click
import structlog
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from codelyzer import __version__
from codelyzer.diff.parser import GitDiffParser
from codelyzer.llm.client import get_llm_client
from codelyzer.retrieval.indexer import RepositoryIndexer
from codelyzer.testing.generator import generate_tests_for_changes, summarize_test_generation_preflight
from codelyzer.testing.qa_agents import (
    research_before_generation_agent,
    run_integrated_generation_validation,
    run_qa_agents,
)
from codelyzer.workflow.state import WorkflowState
from codelyzer.config import settings

load_dotenv()
logger = structlog.get_logger(__name__)


def _extract_changed_examples(file_diff, max_examples: int = 3) -> tuple[list[str], list[str]]:
    added: list[str] = []
    removed: list[str] = []
    for hunk in file_diff.hunks:
        for raw_line in hunk.content.splitlines():
            if raw_line.startswith('+') and len(added) < max_examples:
                text = raw_line[1:].strip()
                if text:
                    added.append(text)
            elif raw_line.startswith('-') and len(removed) < max_examples:
                text = raw_line[1:].strip()
                if text:
                    removed.append(text)
    return added, removed


def _build_file_diff_payload(file_diff, max_chars_per_file: int) -> str:
    hunk_chunks: list[str] = []
    for idx, hunk in enumerate(file_diff.hunks, start=1):
        hunk_chunks.append(
            f"@@ hunk {idx} -{hunk.start_line_old},{hunk.lines_old} +{hunk.start_line_new},{hunk.lines_new}\n"
            f"{hunk.content}"
        )

    file_payload = "\n".join(hunk_chunks)
    if len(file_payload) > max_chars_per_file:
        return file_payload[:max_chars_per_file] + "\n[TRUNCATED]"
    return file_payload


def _summarize_file_with_llm(llm, file_diff, max_chars_per_file: int) -> tuple[str, str, str]:
    llm_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert software reviewer. Explain code changes precisely from diff evidence only. "
            "Avoid generic phrases like 'improves maintainability' unless you justify with concrete diff details."
        ),
        (
            "human",
            "File path: {file_path}\n"
            "Change type: {change_type}\n"
            "Git diff:\n{diff_payload}\n\n"
            "Return exactly three lines, one sentence each:\n"
            "What changed: ...\n"
            "Why it was changed: ...\n"
            "What it does: ..."
        ),
    ])

    diff_payload = _build_file_diff_payload(file_diff, max_chars_per_file)
    response = llm.generate_text(
        llm_prompt,
        {
            "file_path": str(file_diff.file_path),
            "change_type": file_diff.change_type,
            "diff_payload": diff_payload,
        },
    )
    if not response or not response.strip():
        raise RuntimeError("LLM returned an empty summary")

    extracted = {
        "what changed": "",
        "why it was changed": "",
        "what it does": "",
    }
    for line in response.splitlines():
        stripped = line.strip().lstrip("- ").strip()
        lower = stripped.lower()
        for key in extracted:
            prefix = f"{key}:"
            if lower.startswith(prefix):
                extracted[key] = stripped[len(prefix):].strip()

    if not extracted["what changed"]:
        raise RuntimeError("LLM response missing 'What changed' field")
    if not extracted["why it was changed"]:
        raise RuntimeError("LLM response missing 'Why it was changed' field")
    if not extracted["what it does"]:
        raise RuntimeError("LLM response missing 'What it does' field")

    return extracted["what changed"], extracted["why it was changed"], extracted["what it does"]


def _fallback_file_summary(file_diff, error_message: str) -> tuple[str, str, str]:
    added, removed = _extract_changed_examples(file_diff)
    added_examples = "; ".join(added) if added else "none"
    removed_examples = "; ".join(removed) if removed else "none"
    change_count = len(file_diff.hunks)

    what_changed = (
        f"{file_diff.change_type} with {change_count} hunks; added examples: {added_examples}; "
        f"removed examples: {removed_examples}."
    )
    why_changed = f"Could not generate LLM rationale for this file ({error_message})."
    what_it_does = "Applies the shown line-level modifications in this file."
    return what_changed, why_changed, what_it_does


def _render_changed_files_summary(structured_diff) -> str:
    if not structured_diff or not structured_diff.files:
        return "No changed files were detected."
    max_files = max(1, settings.summary_max_files)
    max_chars_per_file = max(500, settings.summary_max_chars_per_file)
    llm = None
    try:
        llm = get_llm_client()
    except Exception as exc:
        logger.warning("LLM client unavailable, using per-file fallback summaries", error=str(exc))

    lines: list[str] = []
    for file_diff in structured_diff.files[:max_files]:
        if llm is not None:
            try:
                what_changed, why_changed, effect = _summarize_file_with_llm(
                    llm,
                    file_diff,
                    max_chars_per_file,
                )
            except Exception as exc:
                what_changed, why_changed, effect = _fallback_file_summary(file_diff, str(exc))
        else:
            what_changed, why_changed, effect = _fallback_file_summary(file_diff, "LLM client not available")

        lines.append(f"- {file_diff.file_path}")
        lines.append(f"  What changed: {what_changed}")
        lines.append(f"  Why it was changed: {why_changed}")
        lines.append(f"  What it does: {effect}")

    if len(structured_diff.files) > max_files:
        omitted = len(structured_diff.files) - max_files
        lines.append(f"- [omitted {omitted} files due to summary_max_files limit]")

    return "\n".join(lines)


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
):
    """Analyze repository changes and generate insights."""
    click.echo(f"🔍 Analyzing repository: {repo_path}")
    click.echo(f"📌 Base ref: {base}" + (f" → Target ref: {target}" if target else ""))

    # Initialize workflow state
    state = WorkflowState(
        repo_path=repo_path,
        base_ref=base,
        target_ref=target
    )

    # Parse git diff
    try:
        parser = GitDiffParser(repo_path)
        state.structured_diff = parser.parse_diff(base, target)
        state.pr_summary = _render_changed_files_summary(state.structured_diff)

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


@cli.command("pr-analyze")
@click.argument('repo_path', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--base', required=True, help='Base git reference (PR base branch)')
@click.option('--target', required=True, help='Target git reference (PR head commit)')
@click.option('--output', type=click.Path(path_type=Path), default=None,
              help='Write the PR comment markdown to this file')
@click.option('--generate-tests', is_flag=True, help='Also generate tests (results included in comment)')
@click.option('--post-comment', is_flag=True, help='Post the comment to a GitHub PR via API')
@click.option('--github-token', envvar='GITHUB_TOKEN', help='GitHub token for posting comments')
@click.option('--repo-full-name', envvar='GITHUB_REPOSITORY', help='GitHub repo (owner/name)')
@click.option('--pr-number', type=int, envvar='PR_NUMBER', help='PR number to comment on')
def pr_analyze(
    repo_path: Path,
    base: str,
    target: str,
    output: Path | None,
    generate_tests: bool,
    post_comment: bool,
    github_token: str | None,
    repo_full_name: str | None,
    pr_number: int | None,
):
    """Run analysis for a pull request and generate a PR comment.

    Designed for CI/GitHub Actions. Runs headless analysis, formats results
    as a rich GitHub markdown comment, and optionally posts it to the PR.
    """
    from codelyzer.pr_comment import format_pr_comment

    click.echo(f"🔬 PR Analysis: {repo_path}")
    click.echo(f"📌 {base} → {target}")

    # Initialize workflow state
    state = WorkflowState(
        repo_path=repo_path,
        base_ref=base,
        target_ref=target,
    )

    # Parse git diff
    try:
        parser = GitDiffParser(repo_path)
        state.structured_diff = parser.parse_diff(base, target)
        state.pr_summary = _render_changed_files_summary(state.structured_diff)
        click.echo(f"✅ Parsed diff: {state.structured_diff.total_files_changed} files changed, "
                   f"+{state.structured_diff.total_insertions} -{state.structured_diff.total_deletions}")
    except Exception as e:
        click.echo(f"❌ Failed to parse diff: {e}", err=True)
        return 1

    # Index repository (best-effort, skip if embedding model unavailable in CI)
    indexer = None
    try:
        click.echo("📚 Indexing repository code...")
        indexer = RepositoryIndexer(repo_path)
        chunk_count = indexer.index_repository()
        click.echo(f"✅ Indexed: {chunk_count} code chunks")
    except Exception as e:
        click.echo(f"⚠️  Indexing skipped (OK in CI): {e}")
        state.add_error(f"Indexing warning: {e}")

    # Find existing tests
    try:
        if indexer is not None:
            for file_diff in (state.structured_diff.files if state.structured_diff else []):
                tests = indexer.search_tests_for_file(file_diff.file_path)
                if tests:
                    state.existing_tests[str(file_diff.file_path)] = tests
    except Exception as e:
        state.add_error(f"Test search warning: {e}")

    # Optional: generate tests
    if generate_tests:
        click.echo("🧪 Generating unit tests...")
        state = generate_tests_for_changes(state, indexer=indexer)
        click.echo(_render_test_generation_report(state))

    # Format the PR comment
    comment_md = format_pr_comment(
        state,
        include_tests=generate_tests,
        include_qa=False,
    )

    # Write comment to file
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(comment_md, encoding="utf-8")
        click.echo(f"📝 Comment written to: {output}")
    else:
        # Default output path
        default_output = repo_path / "pr_comment.md"
        default_output.write_text(comment_md, encoding="utf-8")
        click.echo(f"📝 Comment written to: {default_output}")

    # Post to GitHub
    if post_comment:
        if not all([github_token, repo_full_name, pr_number]):
            click.echo("❌ --post-comment requires GITHUB_TOKEN, GITHUB_REPOSITORY, and PR_NUMBER", err=True)
            return 1
        try:
            _post_github_pr_comment(github_token, repo_full_name, pr_number, comment_md)
            click.echo(f"✅ Comment posted to PR #{pr_number}")
        except Exception as e:
            click.echo(f"❌ Failed to post comment: {e}", err=True)
            return 1

    click.echo("🎉 PR analysis complete!")
    return 0


def _post_github_pr_comment(
    token: str,
    repo_full_name: str,
    pr_number: int,
    body: str,
) -> None:
    """Post or update a comment on a GitHub PR.

    If a previous Codelyzer comment exists, it is updated in-place.
    Otherwise a new comment is created.
    """
    import json
    import urllib.request

    api_base = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
        "User-Agent": "Codelyzer-Bot",
    }

    # Check for existing Codelyzer comment to update
    marker = "## 🔬 Codelyzer Analysis Report"
    existing_comment_id = None

    try:
        req = urllib.request.Request(api_base, headers=headers, method="GET")
        with urllib.request.urlopen(req) as resp:
            comments = json.loads(resp.read().decode())
            for comment in comments:
                if marker in (comment.get("body") or ""):
                    existing_comment_id = comment["id"]
                    break
    except Exception:
        pass  # If listing fails, just create a new comment

    payload = json.dumps({"body": body}).encode()

    if existing_comment_id:
        # Update existing comment
        update_url = f"https://api.github.com/repos/{repo_full_name}/issues/comments/{existing_comment_id}"
        req = urllib.request.Request(update_url, data=payload, headers=headers, method="PATCH")
    else:
        # Create new comment
        req = urllib.request.Request(api_base, data=payload, headers=headers, method="POST")

    with urllib.request.urlopen(req) as resp:
        if resp.status not in (200, 201):
            raise RuntimeError(f"GitHub API returned {resp.status}")


def main():
    """CLI entry point."""
    try:
        return cli(standalone_mode=False)
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    exit(main())

