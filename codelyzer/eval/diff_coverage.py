"""Approximate coverage on diff-touched new-file lines using coverage.xml."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from codelyzer.diff.parser import DiffHunk, FileDiff, StructuredDiff
from codelyzer.workflow.state import WorkflowState


def _added_new_file_line_numbers(hunk: DiffHunk) -> list[int]:
    """Line numbers in the *new* revision for added `+` lines (excluding `+++` noise)."""
    lines: list[int] = []
    new_line = hunk.start_line_new
    for raw in hunk.content.splitlines():
        if not raw:
            continue
        if raw.startswith("\\"):
            continue
        prefix = raw[0] if raw else " "
        if prefix == " ":
            new_line += 1
        elif prefix == "+":
            lines.append(new_line)
            new_line += 1
        elif prefix == "-":
            continue
    return lines


def collect_diff_touch_lines(structured_diff: StructuredDiff | None) -> dict[str, set[int]]:
    """Map repo-relative posix path string -> set of 1-based line numbers touched as additions."""
    out: dict[str, set[int]] = {}
    if not structured_diff:
        return out
    for fd in structured_diff.files:
        if fd.change_type == "deleted":
            continue
        if not str(fd.file_path).endswith(".py"):
            continue
        key = fd.file_path.as_posix()
        acc = out.setdefault(key, set())
        for hunk in fd.hunks:
            acc.update(_added_new_file_line_numbers(hunk))
    return out


def _parse_coverage_line_hits(coverage_xml: Path) -> dict[str, dict[int, int]]:
    """filename (as in XML) -> line_no -> hits."""
    if not coverage_xml.exists():
        return {}
    try:
        tree = ET.parse(coverage_xml)
    except ET.ParseError:
        return {}
    root = tree.getroot()
    result: dict[str, dict[int, int]] = {}
    for cls in root.iter("class"):
        fn = cls.get("filename")
        if not fn:
            continue
        hits_map: dict[int, int] = {}
        for line_el in cls.iter("line"):
            try:
                num = int(line_el.get("number", "0"))
            except ValueError:
                continue
            try:
                hits = int(line_el.get("hits", "0"))
            except ValueError:
                hits = 0
            hits_map[num] = hits
        if hits_map:
            result[fn] = hits_map
    return result


def _normalize_cov_path(repo_path: Path, xml_path: str) -> str | None:
    """Match coverage.xml path to diff path keys."""
    p = Path(xml_path)
    parts = p.parts
    for i, part in enumerate(parts):
        if part in ("site-packages",):
            return None
    try:
        rel = p.relative_to(repo_path.resolve())
        return rel.as_posix()
    except ValueError:
        return p.as_posix()


def compute_diff_touch_coverage(state: WorkflowState) -> dict[str, Any]:
    """
    Fraction of *added* diff lines (new revision) that have hit count > 0 in coverage.xml.

    Returns status plus per-file breakdown. Requires pytest --cov-report=xml at repo root.
    """
    if not state.structured_diff:
        return {"status": "no_diff", "percent": None, "files": []}

    touch = collect_diff_touch_lines(state.structured_diff)
    total_touch = sum(len(v) for v in touch.values())
    if total_touch == 0:
        return {"status": "no_py_touch_lines", "percent": None, "files": []}

    xml_path = state.repo_path / "coverage.xml"
    line_hits = _parse_coverage_line_hits(xml_path)
    if not line_hits:
        return {"status": "no_coverage_xml", "percent": None, "touch_lines": total_touch, "files": []}

    covered = 0
    per_file: list[dict[str, Any]] = []
    for rel, line_nums in sorted(touch.items()):
        cov_key = None
        for candidate in line_hits:
            norm = _normalize_cov_path(state.repo_path, candidate)
            if norm == rel or candidate.endswith(rel):
                cov_key = candidate
                break
        if not cov_key or cov_key not in line_hits:
            per_file.append({"file": rel, "touch": len(line_nums), "covered": 0, "matched": False})
            continue
        hm = line_hits[cov_key]
        c = sum(1 for ln in line_nums if hm.get(ln, 0) > 0)
        covered += c
        per_file.append({"file": rel, "touch": len(line_nums), "covered": c, "matched": True})

    pct = round(100.0 * covered / total_touch, 2) if total_touch else None
    return {
        "status": "ok",
        "percent": pct,
        "touch_lines": total_touch,
        "covered_lines": covered,
        "files": per_file,
    }
