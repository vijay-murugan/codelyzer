"""JSONL batch runner for fixed (repo, base, target) eval matrix."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import click

from codelyzer.analysis_runner import run_analyze_workflow, workflow_to_eval_record


def _pytest_targets_from_case(case: dict[str, Any]) -> list[str] | None:
    raw = case.get("pytest_targets")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out or None


def _coverage_scope_from_case(case: dict[str, Any]) -> str:
    raw = case.get("coverage_scope", "runtime")
    if isinstance(raw, str) and raw.strip() == "full_source":
        return "full_source"
    return "runtime"


@click.group("eval")
def eval_cli() -> None:
    """Evaluation and benchmarking helpers."""


@eval_cli.command("batch")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="JSONL output path (default: eval_batch_results.jsonl in cwd).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Print per-case progress, pytest command, and stderr tail on failure (pytest stdout is still captured).",
)
def eval_batch(config_path: Path, output: Path | None, verbose: bool) -> None:
    """
    Run headless analyze for each case in CONFIG (JSON array or {\"cases\": [...]}).

    Each case: repo_path (required), base, target, generate_tests, auto_validate_tests,
    run_qa, cov_target, min_coverage, qa_report_path, require_tests_for_e2e, pytest_targets,
    use_system_python (run pytest with current Python, no .codelyzer_venv / pip),
    coverage_scope (runtime|full_source),
    label.
    """
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = raw if isinstance(raw, list) else raw["cases"]
    out_path = output or Path("eval_batch_results.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for case in cases:
            label = str(case.get("label", ""))
            repo = Path(case["repo_path"]).resolve()
            base = str(case.get("base", "HEAD"))
            target = case.get("target")
            if target is not None:
                target = str(target)
            gen = bool(case.get("generate_tests", False))
            auto_val = bool(case.get("auto_validate_tests", False))
            run_qa = bool(case.get("run_qa", False))
            cov_target = str(case.get("cov_target", "codelyzer"))
            min_cov = case.get("min_coverage")
            min_cov_int = int(min_cov) if min_cov is not None and str(min_cov).strip().isdigit() else None
            qa_rp = case.get("qa_report_path")
            qa_path = Path(qa_rp) if qa_rp else None
            pytest_targets = _pytest_targets_from_case(case)
            use_system_python = bool(case.get("use_system_python", False))
            coverage_scope = _coverage_scope_from_case(case)

            t0 = time.perf_counter()
            err: str | None = None
            if verbose:
                click.echo(
                    f"[{label or 'case'}] repo={repo} base={base} target={target!r} "
                    f"generate_tests={gen} auto_validate_tests={auto_val} run_qa={run_qa} "
                    f"pytest_targets={pytest_targets!r} use_system_python={use_system_python} "
                    f"coverage_scope={coverage_scope}"
                )
            try:
                state = run_analyze_workflow(
                    repo,
                    base=base,
                    target=target,
                    generate_tests=gen,
                    auto_validate_tests=auto_val,
                    run_qa=run_qa,
                    qa_report_path=qa_path,
                    cov_target=cov_target,
                    min_coverage=min_cov_int,
                    pytest_targets=pytest_targets,
                    use_system_python=use_system_python,
                    coverage_scope=coverage_scope,
                )
            except Exception as exc:
                err = str(exc)
                if verbose:
                    click.echo(f"[{label or 'case'}] ERROR: {err}")
                row = {
                    "case_label": case.get("label", ""),
                    "repo_path": str(repo),
                    "base_ref": base,
                    "target_ref": target,
                    "error": err,
                    "elapsed_seconds": round(time.perf_counter() - t0, 4),
                }
                fh.write(json.dumps(row, default=str) + "\n")
                continue

            elapsed = round(time.perf_counter() - t0, 4)
            extra = {
                "case_label": case.get("label", ""),
                "elapsed_seconds": elapsed,
                "require_tests_for_e2e": bool(case.get("require_tests_for_e2e", gen or auto_val or run_qa)),
                "min_coverage_for_e2e": min_cov_int,
            }
            row = workflow_to_eval_record(state, extra=extra)
            fh.write(json.dumps(row, default=str) + "\n")
            if verbose:
                tr = state.test_run_report or {}
                cmd = tr.get("command") or []
                cmd_s = " ".join(str(x) for x in cmd) if cmd else "(none)"
                click.echo(
                    f"[{label or 'case'}] done in {elapsed}s — pytest_status={tr.get('status')} "
                    f"return_code={tr.get('return_code')} targets_used={tr.get('targets')!r}"
                )
                click.echo(f"[{label or 'case'}] pytest command: {cmd_s}")
                if state.qa_report_path:
                    click.echo(f"[{label or 'case'}] QA report: {state.qa_report_path}")
                if tr.get("status") not in (None, "passed"):
                    tail = (tr.get("stderr_tail") or "").rstrip()
                    if tail:
                        click.echo(f"[{label or 'case'}] pytest stderr (tail):\n{tail[-1200:]}")

    click.echo(f"Wrote {len(cases)} row(s) to {out_path}")


@eval_cli.command("mutation-docs")
def eval_mutation_docs() -> None:
    """Print optional mutmut-based mutation testing workflow for benchmark repos."""
    from codelyzer.eval import mutation_benchmark

    click.echo(mutation_benchmark.MUTATION_EVAL_README)
