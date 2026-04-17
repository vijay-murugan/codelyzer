from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate

from codelyzer.llm.client import BaseLLMClient, get_llm_client
from codelyzer.testing.generator import mock_helpers_in_test_code, source_suggests_external_io
from codelyzer.workflow.state import WorkflowState

logger = structlog.get_logger(__name__)

_SUMMARY_LINE_RE = re.compile(r"=+\s+(\d+)\s+(\w+)")
_COLLECTED_RE = re.compile(r"collected\s+(\d+)\s+items?")
_COVERAGE_TOTAL_RE = re.compile(r"TOTAL\s+\d+\s+\d+\s+(\d+)%")

# Module import name → PyPI package name (for common cases where they differ)
_IMPORT_TO_PYPI: dict[str, str] = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "git": "gitpython",
    "dotenv": "python-dotenv",
    "attr": "attrs",
    "dateutil": "python-dateutil",
    "jose": "python-jose",
    "jwt": "PyJWT",
    "serial": "pyserial",
    "usb": "pyusb",
    "skimage": "scikit-image",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "psycopg2": "psycopg2-binary",
    "grpc": "grpcio",
    "Crypto": "pycryptodome",
    "nacl": "PyNaCl",
    "langchain_core": "langchain-core",
    "langchain_community": "langchain-community",
    "langchain_ollama": "langchain-ollama",
    "langchain_chroma": "langchain-chroma",
    "langchain_text_splitters": "langchain-text-splitters",
}

# Known third-party packages where import name == PyPI name
_KNOWN_THIRD_PARTY: set[str] = {
    "requests", "flask", "django", "fastapi", "numpy", "pandas", "scipy",
    "matplotlib", "seaborn", "plotly", "torch", "tensorflow", "boto3",
    "botocore", "redis", "celery", "sqlalchemy", "alembic", "pydantic",
    "httpx", "aiohttp", "urllib3", "pymongo", "psutil", "click", "typer",
    "rich", "tqdm", "structlog", "loguru", "jinja2", "gunicorn", "uvicorn",
    "starlette", "marshmallow", "dlib", "transformers", "datasets",
    "networkx", "sympy", "lxml", "paramiko", "scrapy", "selenium",
    "hypothesis", "freezegun", "responses", "moto", "toml", "tomli",
    "msgpack", "cryptography", "bcrypt", "passlib", "arrow", "pendulum",
    "pytz", "watchdog", "schedule", "pygments", "chromadb", "langchain",
    "langgraph", "pytest", "openpyxl", "xlrd",
}


@dataclass
class QARunArtifacts:
    command: list[str]
    stdout: str
    stderr: str
    return_code: int


def _pytest_target_paths(state: WorkflowState) -> list[str]:
    generated = [
        result.test_file_path
        for result in state.generated_tests.values()
        if result.status in {"generated", "appended"} and result.test_file_path
    ]
    deduped = sorted(set(generated))
    if deduped:
        return deduped
    return ["tests"]


def _detect_dependency_file(repo_path: Path) -> Path | None:
    preferred = [
        repo_path / "backend" / "requirements-dev.txt",
        repo_path / "backend" / "requirements.txt",
        repo_path / "requirements-dev.txt",
        repo_path / "requirements.txt",
    ]
    for path in preferred:
        if path.exists():
            return path
    return None


def _extract_top_level_imports(file_path: Path) -> set[str]:
    """Parse a Python file via AST and return top-level module names from import statements."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return set()

    modules: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:  # absolute imports only
                modules.add(node.module.split(".")[0])
    return modules


def _is_project_internal(module_name: str, repo_path: Path) -> bool:
    """Check if a module name resolves to a package or file inside the repo."""
    candidate_dir = repo_path / module_name
    if candidate_dir.is_dir() and (candidate_dir / "__init__.py").exists():
        return True
    candidate_file = repo_path / f"{module_name}.py"
    if candidate_file.is_file():
        return True
    return False


def _parse_existing_requirements(req_path: Path) -> set[str]:
    """Parse normalized package names from an existing requirements file."""
    if not req_path.exists():
        return set()
    names: set[str] = set()
    try:
        for line in req_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                continue
            pkg = re.split(r"[><=!~;\[]", stripped, maxsplit=1)[0].strip()
            if pkg:
                names.add(pkg.lower())
    except OSError:
        pass
    return names


def _resolve_pypi_name(module_name: str) -> str | None:
    """Map a Python import name to its PyPI package name."""
    if module_name in _IMPORT_TO_PYPI:
        return _IMPORT_TO_PYPI[module_name]
    if module_name in _KNOWN_THIRD_PARTY:
        return module_name
    # For unknown modules that passed stdlib/project-internal filtering,
    # assume import name == package name.
    return module_name


def generate_test_requirements(state: "WorkflowState") -> Path | None:
    """Scan generated tests and source files for imports; create or append to requirements.txt.

    If a requirements file already exists in the repo, new dependencies are appended.
    Otherwise a new ``requirements.txt`` is created at the repo root.

    Returns the path to the requirements file, or ``None`` if no generated tests to scan.
    """
    files_to_scan: set[Path] = set()
    for result in state.generated_tests.values():
        if result.status not in {"generated", "appended"}:
            continue
        if result.test_file_path:
            tp = Path(result.test_file_path)
            if tp.exists():
                files_to_scan.add(tp)
        src = Path(state.repo_path) / result.source_file_path
        if src.exists():
            files_to_scan.add(src)

    if not files_to_scan:
        logger.info("generate_test_requirements: no files to scan")
        return None

    # Extract all imports from test + source files
    all_imports: set[str] = set()
    for fpath in files_to_scan:
        all_imports.update(_extract_top_level_imports(fpath))

    # Filter out stdlib and project-internal modules
    stdlib_names: set[str] = getattr(sys, "stdlib_module_names", set())
    third_party = {
        name for name in all_imports
        if name not in stdlib_names and not _is_project_internal(name, state.repo_path)
    }

    # Map to PyPI package names
    pypi_packages: set[str] = set()
    for mod in third_party:
        pkg = _resolve_pypi_name(mod)
        if pkg:
            pypi_packages.add(pkg)

    # Always include test runner dependencies
    pypi_packages.add("pytest")
    pypi_packages.add("pytest-cov")

    # Find existing or decide on new requirements path
    existing_req = _detect_dependency_file(state.repo_path)
    req_path = existing_req if existing_req is not None else state.repo_path / "requirements.txt"

    # Avoid duplicating already-listed packages
    existing_packages = _parse_existing_requirements(req_path)
    new_packages = sorted(pkg for pkg in pypi_packages if pkg.lower() not in existing_packages)

    if not new_packages:
        logger.info("All detected dependencies already present in requirements", path=str(req_path))
        state.generated_requirements_path = str(req_path)
        return req_path

    # Append new packages
    req_path.parent.mkdir(parents=True, exist_ok=True)
    with req_path.open("a", encoding="utf-8") as f:
        f.write("\n# Auto-detected dependencies for generated tests (codelyzer)\n")
        for pkg in new_packages:
            f.write(f"{pkg}\n")

    logger.info(
        "Appended test dependencies to requirements",
        path=str(req_path),
        new_packages=new_packages,
    )
    state.generated_requirements_path = str(req_path)
    return req_path


def _detect_pythonpath_override(repo_path: Path) -> str | None:
    backend_app = repo_path / "backend" / "app"
    if backend_app.exists() and backend_app.is_dir():
        return "backend"
    return None


def _venv_python_path(repo_path: Path) -> Path:
    if os.name == "nt":
        return repo_path / ".codelyzer_venv" / "Scripts" / "python.exe"
    return repo_path / ".codelyzer_venv" / "bin" / "python"


def _bootstrap_test_environment(
    repo_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "enabled": True,
        "venv_path": str(repo_path / ".codelyzer_venv"),
        "venv_python": "",
        "dependency_file": None,
        "pythonpath_override": _detect_pythonpath_override(repo_path),
        "created_venv": False,
        "install_command": [],
        "status": "ok",
        "error": "",
    }
    pybin = _venv_python_path(repo_path)
    info["venv_python"] = str(pybin)
    if not pybin.exists():
        create_cmd = ["python3", "-m", "venv", str(repo_path / ".codelyzer_venv")]
        created = subprocess.run(
            create_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if created.returncode != 0:
            info["status"] = "failed"
            info["error"] = f"Failed to create venv: {(created.stderr or created.stdout)[-300:]}"
            return info
        info["created_venv"] = True

    dep_file = _detect_dependency_file(repo_path)
    if dep_file:
        info["dependency_file"] = str(dep_file)
        install_cmd = [
            str(pybin),
            "-m",
            "pip",
            "install",
            "-r",
            str(dep_file),
            "pytest",
            "pytest-cov",
        ]
    else:
        install_cmd = [str(pybin), "-m", "pip", "install", "pytest", "pytest-cov"]
    info["install_command"] = install_cmd
    installed = subprocess.run(
        install_cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if installed.returncode != 0:
        info["status"] = "failed"
        info["error"] = f"Dependency install failed: {(installed.stderr or installed.stdout)[-300:]}"
    return info


def _candidate_source_files(state: WorkflowState) -> list[str]:
    if not state.structured_diff or not state.structured_diff.files:
        return []
    return [str(item.file_path) for item in state.structured_diff.files if str(item.file_path).endswith(".py")]


def run_test_and_coverage(
    state: WorkflowState,
    cov_target: str = "codelyzer",
    timeout_seconds: int = 600,
) -> WorkflowState:
    """Single pytest invocation used by both testRunnerAgent and coverageAgent."""

    bootstrap = _bootstrap_test_environment(state.repo_path, timeout_seconds=timeout_seconds)
    if bootstrap.get("status") != "ok":
        msg = bootstrap.get("error") or "QA bootstrap failed"
        state.add_error(msg)
        state.test_run_report = {
            "status": "failed",
            "command": [],
            "targets": _pytest_target_paths(state),
            "return_code": None,
            "collected": None,
            "counts": {},
            "failures": [],
            "stdout_tail": "",
            "stderr_tail": "",
            "bootstrap": bootstrap,
        }
        state.coverage_report = {"status": "unavailable", "reason": msg}
        state.final_validation_status = "failed"
        return state

    targets = _pytest_target_paths(state)
    pybin = bootstrap.get("venv_python") or "python3"
    cmd = [
        str(pybin),
        "-m",
        "pytest",
        *targets,
        f"--cov={cov_target}",
        "--cov-branch",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "-q",
    ]
    logger.info("Running tandem test+coverage", command=cmd)
    env = os.environ.copy()
    pythonpath_override = bootstrap.get("pythonpath_override")
    if isinstance(pythonpath_override, str) and pythonpath_override:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{pythonpath_override}{os.pathsep}{existing}" if existing else pythonpath_override
        )
    try:
        completed = subprocess.run(
            cmd,
            cwd=state.repo_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        message = f"QA test run timed out after {timeout_seconds}s"
        state.add_error(message)
        report = {
            "status": "timeout",
            "command": cmd,
            "targets": targets,
            "return_code": None,
            "stdout_tail": (exc.stdout or "")[-2000:],
            "stderr_tail": (exc.stderr or "")[-2000:],
        }
        state.test_run_report = report
        state.coverage_report = {
            "status": "unavailable",
            "reason": message,
        }
        state.final_validation_status = "timeout"
        return state

    artifacts = QARunArtifacts(
        command=cmd,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        return_code=completed.returncode,
    )

    state.test_run_report = parse_test_runner_output(artifacts)
    state.test_run_report["bootstrap"] = bootstrap
    state.coverage_report = parse_coverage_output(artifacts, state.repo_path)
    state.final_validation_status = state.test_run_report.get("status", "unknown")
    return state


def parse_test_runner_output(artifacts: QARunArtifacts) -> dict[str, Any]:
    stdout = artifacts.stdout
    stderr = artifacts.stderr
    counts: dict[str, int] = {}
    for match in _SUMMARY_LINE_RE.finditer(stdout):
        counts[match.group(2)] = counts.get(match.group(2), 0) + int(match.group(1))

    collected = None
    collected_match = _COLLECTED_RE.search(stdout)
    if collected_match:
        collected = int(collected_match.group(1))

    failed_sections: list[str] = []
    if "FAILURES" in stdout:
        failures_blob = stdout.split("FAILURES", maxsplit=1)[-1]
        for part in failures_blob.split("__ ")[:6]:
            trimmed = part.strip()
            if trimmed:
                failed_sections.append(trimmed[:1200])

    status = "passed" if artifacts.return_code == 0 else "failed"
    return {
        "status": status,
        "command": artifacts.command,
        "return_code": artifacts.return_code,
        "collected": collected,
        "counts": counts,
        "failures": failed_sections,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-2000:],
    }


def _parse_coverage_xml(repo_path: Path) -> dict[str, Any]:
    xml_path = repo_path / "coverage.xml"
    if not xml_path.exists():
        return {"files": []}
    try:
        xml_text = xml_path.read_text(encoding="utf-8")
    except Exception:
        return {"files": []}

    # Lightweight parse by regex to avoid XML parser overhead.
    file_hits = re.findall(
        r'<class[^>]*filename="([^"]+)"[^>]*line-rate="([^"]+)"[^>]*branch-rate="([^"]+)"',
        xml_text,
    )
    files = []
    for filename, line_rate, branch_rate in file_hits:
        try:
            files.append(
                {
                    "file": filename,
                    "line_rate": round(float(line_rate) * 100, 2),
                    "branch_rate": round(float(branch_rate) * 100, 2),
                }
            )
        except ValueError:
            continue
    return {"files": files}


def parse_coverage_output(artifacts: QARunArtifacts, repo_path: Path) -> dict[str, Any]:
    stdout = artifacts.stdout
    total_match = _COVERAGE_TOTAL_RE.search(stdout)
    total_percent = int(total_match.group(1)) if total_match else None

    per_file: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        columns = line.strip().split()
        if len(columns) < 4:
            continue
        if not columns[-1].endswith("%"):
            continue
        file_name = columns[0]
        if file_name == "TOTAL":
            continue
        try:
            cover = int(columns[-1].rstrip("%"))
            stmts = int(columns[1])
            miss = int(columns[2])
        except ValueError:
            continue
        per_file.append(
            {
                "file": file_name,
                "cover_percent": cover,
                "stmts": stmts,
                "miss": miss,
            }
        )

    xml_data = _parse_coverage_xml(repo_path)
    if xml_data.get("files"):
        xml_map = {item["file"]: item for item in xml_data["files"]}
        for item in per_file:
            merged = xml_map.get(item["file"])
            if merged:
                item["line_rate"] = merged["line_rate"]
                item["branch_rate"] = merged["branch_rate"]

    low_coverage = sorted(
        [item for item in per_file if item["cover_percent"] < 80],
        key=lambda row: row["cover_percent"],
    )
    status = "available" if total_percent is not None or per_file else "unavailable"
    return {
        "status": status,
        "total_percent": total_percent,
        "files": per_file,
        "low_coverage_files": low_coverage[:20],
    }


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _resolve_repo_path(repo: Path, candidate: str) -> Path:
    p = Path(candidate)
    if p.is_absolute():
        return p
    return repo / p


def _build_code_review_snippets(state: WorkflowState) -> str:
    blocks: list[str] = []
    for source_key, result in state.generated_tests.items():
        if result.status not in {"generated", "appended"}:
            continue
        unit_hint = ""
        if isinstance(result.trace_context, dict):
            unit_hint = str(result.trace_context.get("unit_under_test", ""))
        src_path = _resolve_repo_path(state.repo_path, source_key)
        source_snippet = ""
        if src_path.exists():
            try:
                source_snippet = _truncate_text(src_path.read_text(encoding="utf-8"), 4500)
            except OSError:
                source_snippet = ""
        test_snippet = ""
        if result.test_file_path:
            tp = _resolve_repo_path(state.repo_path, result.test_file_path)
            if tp.exists():
                try:
                    test_snippet = _truncate_text(tp.read_text(encoding="utf-8"), 4500)
                except OSError:
                    test_snippet = ""
        blocks.append(
            f"### Source file: {source_key}\n"
            f"Unit under test (hint): {unit_hint or 'n/a'}\n"
            f"```python\n{source_snippet}\n```\n"
            f"### Generated tests: {result.test_file_path}\n"
            f"```python\n{test_snippet}\n```\n"
        )
    return "\n".join(blocks) if blocks else "No generated test files to review."


def _get_llm_client_safe() -> BaseLLMClient | None:
    try:
        return get_llm_client()
    except Exception as exc:
        logger.warning("LLM unavailable for QA agents", error=str(exc))
        return None


def review_generated_tests_agent(state: WorkflowState) -> WorkflowState:
    """Review generated pytest code only. Intended to run after generation, before pytest."""

    findings: list[dict[str, Any]] = []

    for source_file, result in state.generated_tests.items():
        if result.status == "failed":
            findings.append(
                {
                    "severity": "high",
                    "source_file": source_file,
                    "message": f"Test generation failed: {result.reason or 'unknown error'}",
                }
            )
            continue
        if result.status in {"generated", "appended"} and not result.test_names:
            findings.append(
                {
                    "severity": "medium",
                    "source_file": source_file,
                    "message": "Generated test artifact has no test_* functions",
                }
            )
        if result.partial:
            findings.append(
                {
                    "severity": "medium",
                    "source_file": source_file,
                    "message": result.reason or "Generated tests are partial and may be invalid",
                }
            )
        if result.status in {"generated", "appended"} and result.test_code:
            src_path = _resolve_repo_path(state.repo_path, source_file)
            if src_path.exists():
                try:
                    src_text = src_path.read_text(encoding="utf-8")
                except OSError:
                    src_text = ""
                if (
                    src_text
                    and source_suggests_external_io(src_text)
                    and not mock_helpers_in_test_code(result.test_code)
                    and "pytest.mark.skip" not in result.test_code
                ):
                    findings.append(
                        {
                            "severity": "medium",
                            "source_file": source_file,
                            "message": (
                                "Source likely uses external I/O but tests show no patch/Mock/MonkeyPatch; "
                                "add mocks at the module's binding site matching API/DB call shapes."
                            ),
                        }
                    )

    has_generated_body = any(
        r.status in {"generated", "appended"} and (r.test_code or r.test_file_path)
        for r in state.generated_tests.values()
    )
    llm = _get_llm_client_safe()
    if llm is not None and has_generated_body:
        code_snippets = _build_code_review_snippets(state)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You review ONLY the generated pytest code in the snippets below. "
                    "Use the source snippets only as context for correct mocking and scope—not to request production refactors. "
                    "Return a compact bullet list only. "
                    "Flag: tests that hit real HTTP/DB/subprocess without mocks; wrong patch import path; "
                    "mocks that do not mirror what production code calls; weak or missing assertions; "
                    "tests that assert behavior outside the unit-under-test hint without justification.",
                ),
                (
                    "human",
                    "Related source and generated test snippets (truncated):\n{code_snippets}\n\n"
                    "Generated tests summary:\n{tests_summary}\n\n"
                    "Return up to 7 findings prefixed with severity [high|medium|low]. "
                    "Do not review or summarize the PR/diff; only the generated tests.",
                ),
            ]
        )
        try:
            response = llm.generate_text(
                prompt,
                {
                    "code_snippets": code_snippets,
                    "tests_summary": str(
                        {
                            k: {
                                "status": v.status,
                                "partial": v.partial,
                                "reason": v.reason,
                                "unit_under_test": (
                                    (v.trace_context or {}).get("unit_under_test")
                                    if isinstance(v.trace_context, dict)
                                    else None
                                ),
                            }
                            for k, v in state.generated_tests.items()
                        }
                    ),
                },
            )
            for line in response.splitlines():
                item = line.strip().lstrip("- ").strip()
                if not item:
                    continue
                sev = "low"
                lowered = item.lower()
                if "[high]" in lowered:
                    sev = "high"
                elif "[medium]" in lowered:
                    sev = "medium"
                findings.append({"severity": sev, "source_file": "", "message": item})
        except Exception as exc:
            logger.warning("LLM generated-test review failed", error=str(exc))

    state.review_findings = findings
    return state


def append_pytest_failure_findings(state: WorkflowState) -> WorkflowState:
    """After pytest, attach failure excerpts to review_findings (replaces prior pytest rows)."""

    prefix = "Pytest failure:"
    state.review_findings = [
        f for f in state.review_findings if not str(f.get("message", "")).startswith(prefix)
    ]
    test_run = state.test_run_report or {}
    if test_run.get("status") != "failed":
        return state
    for failure in test_run.get("failures", [])[:5]:
        state.review_findings.append(
            {
                "severity": "high",
                "source_file": "",
                "message": f"{prefix} {failure}",
            }
        )
    return state


def test_research_agent(state: WorkflowState) -> WorkflowState:
    coverage = state.coverage_report or {}
    low_files = coverage.get("low_coverage_files", [])[:10]
    suggestions: list[dict[str, Any]] = []
    for row in low_files:
        suggestions.append(
            {
                "priority": "high" if row.get("cover_percent", 100) < 60 else "medium",
                "target_file": row.get("file", ""),
                "suggested_test_name": f"test_{Path(row.get('file', 'module')).stem}_uncovered_branches",
                "rationale": (
                    f"Coverage is {row.get('cover_percent', 'unknown')}%; add branch and error-path assertions."
                ),
            }
        )

    llm = _get_llm_client_safe()
    if llm is not None and state.structured_diff and state.structured_diff.files:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a test research agent. Produce targeted unit-test ideas that maximize coverage quickly. "
                    "Focus on changed modules and uncovered branches.",
                ),
                (
                    "human",
                    "Changed files:\n{changed_files}\n\n"
                    "Low coverage files:\n{low_coverage}\n\n"
                    "Return 5 concise bullets in format: target | suggested_test_name | rationale.",
                ),
            ]
        )
        try:
            response = llm.generate_text(
                prompt,
                {
                    "changed_files": "\n".join(str(f.file_path) for f in state.structured_diff.files),
                    "low_coverage": str(low_files),
                },
            )
            for line in response.splitlines():
                item = line.strip().lstrip("- ").strip()
                if not item:
                    continue
                parts = [part.strip() for part in item.split("|")]
                if len(parts) == 3:
                    suggestions.append(
                        {
                            "priority": "medium",
                            "target_file": parts[0],
                            "suggested_test_name": parts[1],
                            "rationale": parts[2],
                        }
                    )
        except Exception as exc:
            logger.warning("LLM research agent failed", error=str(exc))

    state.research_suggestions = suggestions[:20]
    return state


def research_before_generation_agent(state: WorkflowState) -> WorkflowState:
    suggestions: list[dict[str, Any]] = []
    for source_file in _candidate_source_files(state):
        stem = Path(source_file).stem
        suggestions.append(
            {
                "priority": "high",
                "target_file": source_file,
                "suggested_test_name": f"test_{stem}_behavior_changes",
                "rationale": (
                    "Cover changed branches, return paths, and errors; mock HTTP/DB/subprocess at the module binding site."
                ),
            }
        )

    llm = _get_llm_client_safe()
    if llm is not None and state.structured_diff and state.structured_diff.files:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a pre-generation test research agent. Propose high-value unit tests for changed files. "
                    "For any external API, DB, or subprocess usage, say how to mock it (patch target + return_value/side_effect shape). "
                    "Keep scope to the functions changed in the diff.",
                ),
                (
                    "human",
                    "Changed files:\n{changed_files}\n\nReturn 6 bullets in format: target | suggested_test_name | rationale.",
                ),
            ]
        )
        try:
            response = llm.generate_text(
                prompt,
                {"changed_files": "\n".join(_candidate_source_files(state))},
            )
            for line in response.splitlines():
                item = line.strip().lstrip("- ").strip()
                if not item:
                    continue
                parts = [part.strip() for part in item.split("|")]
                if len(parts) == 3:
                    suggestions.append(
                        {
                            "priority": "medium",
                            "target_file": parts[0],
                            "suggested_test_name": parts[1],
                            "rationale": parts[2],
                        }
                    )
        except Exception as exc:
            logger.warning("LLM pre-generation research failed", error=str(exc))

    state.pre_generation_research = suggestions[:30]
    return state


def _remove_vacuous_tests_from_content(content: str) -> tuple[str, list[str]]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content, []

    lines = content.splitlines()
    to_remove: list[tuple[int, int, str]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        body = node.body
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            to_remove.append((node.lineno, node.end_lineno or node.lineno, node.name))
            continue
        has_assert = any(isinstance(stmt, ast.Assert) for stmt in ast.walk(node))
        if not has_assert:
            to_remove.append((node.lineno, node.end_lineno or node.lineno, node.name))

    if not to_remove:
        return content, []

    remove_names = [name for _, _, name in to_remove]
    removed_line_numbers = set()
    for start, end, _ in to_remove:
        removed_line_numbers.update(range(start, end + 1))

    kept = [line for idx, line in enumerate(lines, start=1) if idx not in removed_line_numbers]
    normalized = "\n".join(kept).strip() + "\n"
    return normalized, remove_names


def auto_fix_generated_tests_agent(state: WorkflowState) -> WorkflowState:
    fixes: list[dict[str, Any]] = []
    for source_file, generated in state.generated_tests.items():
        if generated.status not in {"generated", "appended"}:
            continue
        test_path = Path(generated.test_file_path)
        if not test_path.exists():
            continue
        try:
            original = test_path.read_text(encoding="utf-8")
        except Exception:
            continue
        fixed, removed_names = _remove_vacuous_tests_from_content(original)
        if removed_names and fixed != original:
            test_path.write_text(fixed, encoding="utf-8")
            fixes.append(
                {
                    "source_file": source_file,
                    "test_file": str(test_path),
                    "action": "remove_vacuous_tests",
                    "tests": removed_names,
                    "reason": "Auto-fix removed tests without assertions.",
                }
            )
            generated.test_code = fixed
            generated.test_names = [name for name in generated.test_names if name not in removed_names]

    state.refinement_fixes.extend(fixes)
    return state


def remove_inappropriate_tests_agent(state: WorkflowState) -> WorkflowState:
    removed: list[dict[str, Any]] = []
    for source_file, generated in state.generated_tests.items():
        if generated.status not in {"generated", "appended"}:
            continue
        # Delete only if still partial/weak after fix attempts.
        if not generated.partial and generated.test_names:
            continue
        if not generated.test_file_path:
            continue
        path = Path(generated.test_file_path)
        if not path.exists():
            continue
        try:
            path.unlink()
            removed.append(
                {
                    "source_file": source_file,
                    "test_file": str(path),
                    "tests": generated.test_names,
                    "reason": generated.reason or "Removed after failed auto-fix cycle",
                }
            )
            generated.status = "skipped"
            generated.reason = "Removed as inappropriate after refinement cycle"
            generated.test_names = []
            generated.test_code = ""
        except Exception as exc:
            state.add_error(f"Failed to remove inappropriate test file {path}: {exc}")
    state.removed_tests.extend(removed)
    return state


def write_qa_markdown_report(state: WorkflowState, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_run = state.test_run_report or {}
    coverage = state.coverage_report or {}

    lines: list[str] = [
        "# Codelyzer QA Report",
        "",
        "## Test Runner",
        f"- Status: {test_run.get('status', 'unknown')}",
        f"- Return code: {test_run.get('return_code', 'n/a')}",
        f"- Collected tests: {test_run.get('collected', 'n/a')}",
        f"- Counts: {test_run.get('counts', {})}",
        f"- Final validation status: {state.final_validation_status or 'unknown'}",
        "",
    ]
    bootstrap = test_run.get("bootstrap", {}) if isinstance(test_run, dict) else {}
    lines.extend(
        [
            "### Test Environment Bootstrap",
            f"- Status: {bootstrap.get('status', 'not-run')}",
            f"- Venv path: {bootstrap.get('venv_path', 'n/a')}",
            f"- Dependency file: {bootstrap.get('dependency_file', 'none')}",
            f"- PYTHONPATH override: {bootstrap.get('pythonpath_override', 'none')}",
        ]
    )
    if bootstrap.get("error"):
        lines.append(f"- Error: {bootstrap.get('error')}")
    lines.append("")

    failures = test_run.get("failures", [])
    lines.append("### Failures")
    if failures:
        for idx, failure in enumerate(failures, start=1):
            lines.append(f"{idx}. `{failure[:400]}`")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Coverage",
            f"- Status: {coverage.get('status', 'unknown')}",
            f"- Total coverage: {coverage.get('total_percent', 'n/a')}%",
            "",
            "### Low Coverage Files (<80%)",
        ]
    )
    low_files = coverage.get("low_coverage_files", [])
    if low_files:
        for row in low_files:
            lines.append(f"- `{row.get('file', 'unknown')}`: {row.get('cover_percent', 'n/a')}%")
    else:
        lines.append("- None")

    lines.extend(["", "## Generated test review"])
    if state.review_findings:
        for finding in state.review_findings:
            src = f" ({finding.get('source_file')})" if finding.get("source_file") else ""
            lines.append(f"- [{finding.get('severity', 'low')}] {finding.get('message', '')}{src}")
    else:
        lines.append("- No findings")

    lines.extend(["", "## Test Research Suggestions"])
    if state.research_suggestions:
        for suggestion in state.research_suggestions:
            lines.append(
                f"- [{suggestion.get('priority', 'medium')}] {suggestion.get('target_file', '')} | "
                f"{suggestion.get('suggested_test_name', '')} | {suggestion.get('rationale', '')}"
            )
    else:
        lines.append("- No suggestions")

    lines.extend(["", "## Pre-Generation Research Used"])
    if state.pre_generation_research:
        for item in state.pre_generation_research[:20]:
            lines.append(
                f"- [{item.get('priority', 'medium')}] {item.get('target_file', '')} | "
                f"{item.get('suggested_test_name', '')} | {item.get('rationale', '')}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Fixes Applied"])
    if state.refinement_fixes:
        for fix in state.refinement_fixes:
            lines.append(
                f"- {fix.get('action', 'fix')} on `{fix.get('test_file', '')}` "
                f"tests={fix.get('tests', [])} ({fix.get('reason', '')})"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Removed Tests"])
    if state.removed_tests:
        for item in state.removed_tests:
            lines.append(
                f"- `{item.get('test_file', '')}` tests={item.get('tests', [])} reason={item.get('reason', '')}"
            )
    else:
        lines.append("- None")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    state.qa_report_path = str(output_path)
    return output_path


def run_qa_agents(
    state: WorkflowState,
    report_path: Path,
    cov_target: str = "codelyzer",
    min_coverage: int | None = None,
) -> WorkflowState:
    """Run tests + coverage, record pytest failures, coverage research, and report (no LLM test-code review)."""

    state = run_test_and_coverage(state, cov_target=cov_target)
    state = append_pytest_failure_findings(state)
    state = test_research_agent(state)

    coverage_total = state.coverage_report.get("total_percent")
    if min_coverage is not None and isinstance(coverage_total, int):
        if coverage_total < min_coverage:
            state.add_error(
                f"Coverage threshold unmet: {coverage_total}% < required {min_coverage}%"
            )
    write_qa_markdown_report(state, report_path)
    return state


def run_integrated_generation_validation(
    state: WorkflowState,
    report_path: Path,
    cov_target: str = "codelyzer",
    min_coverage: int | None = None,
) -> WorkflowState:
    """After generation: review generated tests, auto-fix, pytest+coverage, prune if needed, research, report."""

    state = review_generated_tests_agent(state)
    state = auto_fix_generated_tests_agent(state)
    state = run_test_and_coverage(state, cov_target=cov_target)
    state = append_pytest_failure_findings(state)

    if state.test_run_report.get("status") != "passed":
        state = remove_inappropriate_tests_agent(state)
        state = run_test_and_coverage(state, cov_target=cov_target)
        state = append_pytest_failure_findings(state)

    state = test_research_agent(state)
    coverage_total = state.coverage_report.get("total_percent")
    if min_coverage is not None and isinstance(coverage_total, int) and coverage_total < min_coverage:
        state.add_error(
            f"Coverage threshold unmet: {coverage_total}% < required {min_coverage}%"
        )
    write_qa_markdown_report(state, report_path)
    return state
