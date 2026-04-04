from git import Repo
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import structlog

logger = structlog.get_logger(__name__)


class DiffHunk(BaseModel):
    start_line_old: int
    start_line_new: int
    lines_old: int
    lines_new: int
    content: str
    changes: List[Dict[str, str]] = Field(default_factory=list)


class FileDiff(BaseModel):
    file_path: Path
    old_path: Optional[Path] = None
    new_path: Optional[Path] = None
    change_type: str  # added, modified, deleted, renamed
    hunks: List[DiffHunk] = Field(default_factory=list)
    language: Optional[str] = None
    module_path: Optional[str] = None


class StructuredDiff(BaseModel):
    base_commit: str
    target_commit: str
    total_files_changed: int
    total_insertions: int
    total_deletions: int
    files: List[FileDiff] = Field(default_factory=list)


class GitDiffParser:
    """Read-only git repository diff parser that works with any local git repo."""

    def __init__(self, repo_path: Path):
        self.repo = Repo(repo_path)
        # what if it is already in a git repository? should we check for that and raise error if not?
        if not self.repo.bare:
            self.repo_path = repo_path.resolve()
        else:
            raise ValueError("Provided path is not a valid git repository")
        logger.info("Initialized git repository", path=self.repo_path)

    def parse_diff(self, base_ref: str = "HEAD", target_ref: Optional[str] = None) -> StructuredDiff:
        """Parse diff between two git references."""

        if target_ref:
            diff = self.repo.git.diff(base_ref, target_ref, find_renames=True)
        else:
            diff = self.repo.git.diff(base_ref, cached=False)

        return self._parse_diff_content(diff, base_ref, target_ref or "working")

    def _parse_diff_content(self, diff_content: str, base_commit: str, target_commit: str) -> StructuredDiff:
        structured_diff = StructuredDiff(
            base_commit=base_commit,
            target_commit=target_commit,
            total_files_changed=0,
            total_insertions=0,
            total_deletions=0,
            files=[]
        )

        diff_lines = diff_content.splitlines()
        current_file: Optional[FileDiff] = None
        current_hunk: Optional[DiffHunk] = None
        hunk_content = []

        for line in diff_lines:
            if line.startswith('diff --git'):
                if current_file and current_hunk:
                    current_hunk.content = '\n'.join(hunk_content)
                    current_file.hunks.append(current_hunk)
                    hunk_content = []

                if current_file:
                    structured_diff.files.append(current_file)
                    structured_diff.total_files_changed += 1

                # Reset hunk state when moving to a new file diff.
                current_hunk = None
                hunk_content = []

                paths = line.split()
                old_path = paths[2][2:] if len(paths) > 2 else None
                new_path = paths[3][2:] if len(paths) > 3 else None

                current_file = FileDiff(
                    file_path=Path(new_path or old_path),
                    old_path=Path(old_path) if old_path else None,
                    new_path=Path(new_path) if new_path else None,
                    change_type='modified',
                    hunks=[]
                )

            elif line.startswith('new file mode'):
                if current_file:
                    current_file.change_type = 'added'

            elif line.startswith('deleted file mode'):
                if current_file:
                    current_file.change_type = 'deleted'

            elif line.startswith('rename from'):
                if current_file:
                    current_file.change_type = 'renamed'

            elif line.startswith('@@ '):
                if current_file and current_hunk:
                    current_hunk.content = '\n'.join(hunk_content)
                    current_file.hunks.append(current_hunk)
                    hunk_content = []

                hunk_info = line.split('@@')[1].strip()
                old_part, new_part = hunk_info.split()
                old_start = int(old_part.split(',')[0][1:])
                old_count = int(old_part.split(',')[1]) if ',' in old_part else 1
                new_start = int(new_part.split(',')[0][1:])
                new_count = int(new_part.split(',')[1]) if ',' in new_part else 1

                current_hunk = DiffHunk(
                    start_line_old=old_start,
                    start_line_new=new_start,
                    lines_old=old_count,
                    lines_new=new_count,
                    content=''
                )

            elif current_hunk is not None:
                hunk_content.append(line)

                if line.startswith('+'):
                    structured_diff.total_insertions += 1
                elif line.startswith('-'):
                    structured_diff.total_deletions += 1

        if current_file and current_hunk:
            current_hunk.content = '\n'.join(hunk_content)
            current_file.hunks.append(current_hunk)

        if current_file:
            structured_diff.files.append(current_file)
            structured_diff.total_files_changed += 1

        logger.info("Parsed git diff",
                    files=structured_diff.total_files_changed,
                    insertions=structured_diff.total_insertions,
                    deletions=structured_diff.total_deletions)

        return structured_diff