"""Per-file PR-style summaries from structured git diffs (LLM + fallback)."""

from __future__ import annotations

import structlog
from langchain_core.prompts import ChatPromptTemplate

from codelyzer.config import settings
from codelyzer.llm.client import get_llm_client

logger = structlog.get_logger(__name__)


def _extract_changed_examples(file_diff, max_examples: int = 3) -> tuple[list[str], list[str]]:
    added: list[str] = []
    removed: list[str] = []
    for hunk in file_diff.hunks:
        for raw_line in hunk.content.splitlines():
            if raw_line.startswith("+") and len(added) < max_examples:
                text = raw_line[1:].strip()
                if text:
                    added.append(text)
            elif raw_line.startswith("-") and len(removed) < max_examples:
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


def summarize_file_with_llm(llm, file_diff, max_chars_per_file: int) -> tuple[str, str, str]:
    llm_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert software reviewer. Explain code changes precisely from diff evidence only. "
                "Avoid generic phrases like 'improves maintainability' unless you justify with concrete diff details.",
            ),
            (
                "human",
                "File path: {file_path}\n"
                "Change type: {change_type}\n"
                "Git diff:\n{diff_payload}\n\n"
                "Return exactly three lines, one sentence each:\n"
                "What changed: ...\n"
                "Why it was changed: ...\n"
                "What it does: ...",
            ),
        ]
    )

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
                extracted[key] = stripped[len(prefix) :].strip()

    if not extracted["what changed"]:
        raise RuntimeError("LLM response missing 'What changed' field")
    if not extracted["why it was changed"]:
        raise RuntimeError("LLM response missing 'Why it was changed' field")
    if not extracted["what it does"]:
        raise RuntimeError("LLM response missing 'What it does' field")

    return extracted["what changed"], extracted["why it was changed"], extracted["what it does"]


def fallback_file_summary(file_diff, error_message: str) -> tuple[str, str, str]:
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


def render_changed_files_summary(structured_diff, *, use_llm: bool = True) -> str:
    if not structured_diff or not structured_diff.files:
        return "No changed files were detected."
    max_files = max(1, settings.summary_max_files)
    max_chars_per_file = max(500, settings.summary_max_chars_per_file)
    llm = None
    if use_llm:
        try:
            llm = get_llm_client()
        except Exception as exc:
            logger.warning("LLM client unavailable, using per-file fallback summaries", error=str(exc))

    lines: list[str] = []
    for file_diff in structured_diff.files[:max_files]:
        if llm is not None:
            try:
                what_changed, why_changed, effect = summarize_file_with_llm(
                    llm,
                    file_diff,
                    max_chars_per_file,
                )
            except Exception as exc:
                what_changed, why_changed, effect = fallback_file_summary(file_diff, str(exc))
        else:
            what_changed, why_changed, effect = fallback_file_summary(file_diff, "LLM client not available")

        lines.append(f"- {file_diff.file_path}")
        lines.append(f"  What changed: {what_changed}")
        lines.append(f"  Why it was changed: {why_changed}")
        lines.append(f"  What it does: {effect}")

    if len(structured_diff.files) > max_files:
        omitted = len(structured_diff.files) - max_files
        lines.append(f"- [omitted {omitted} files due to summary_max_files limit]")

    return "\n".join(lines)
