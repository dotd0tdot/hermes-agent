"""
Scan Python project files and collect code metrics.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Directories to skip
SKIP_DIRS = {
    "__pycache__", ".git", "venv", ".venv", "node_modules",
    "optional-skills", "tests", "test", "e2e", "integration",
    "stress", "website", "docker", ".tox", ".mypy_cache",
}


@dataclass
class FileMetrics:
    """Metrics for a single Python file."""
    path: str
    rel_path: str
    lines: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    code_lines: int = 0
    functions: int = 0
    classes: int = 0
    imports: int = 0
    try_count: int = 0
    except_count: int = 0
    todo_count: int = 0
    fixme_count: int = 0
    type_hinted_functions: int = 0
    total_functions: int = 0
    has_docstring: bool = False
    last_modified: datetime = field(default_factory=datetime.now)
    change_frequency: int = 0  # commits in last 30 days


def scan_project(
    project_path: Path,
    scope: str = "full",
    target: Optional[str] = None,
) -> list[FileMetrics]:
    """Scan a Python project and return file metrics."""
    if target:
        # Scan specific file
        target_path = project_path / target
        if target_path.exists() and target_path.suffix == ".py":
            return [_scan_file(target_path, project_path)]
        return []

    if scope == "local":
        return _scan_local_changes(project_path)

    # Full scan
    metrics = []
    for root, dirs, files in os.walk(project_path):
        # Filter out skip dirs
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            if fname.endswith(".py"):
                fpath = Path(root) / fname
                try:
                    fm = _scan_file(fpath, project_path)
                    metrics.append(fm)
                except Exception as e:
                    logger.debug("Failed to scan %s: %s", fpath, e)

    return metrics


def _scan_local_changes(project_path: Path) -> list[FileMetrics]:
    """Scan only files with uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True,
            cwd=str(project_path),
        )
        changed_files = [
            f.strip() for f in result.stdout.strip().split("\n") if f.strip()
        ]
    except Exception:
        changed_files = []

    # Also include staged
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True,
            cwd=str(project_path),
        )
        staged = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        changed_files = list(set(changed_files + staged))
    except Exception:
        pass

    metrics = []
    for fpath_str in changed_files:
        fpath = project_path / fpath_str
        if fpath.exists() and fpath.suffix == ".py":
            try:
                fm = _scan_file(fpath, project_path)
                metrics.append(fm)
            except Exception as e:
                logger.debug("Failed to scan %s: %s", fpath, e)

    return metrics


def _scan_file(fpath: Path, project_path: Path) -> FileMetrics:
    """Scan a single Python file and collect metrics."""
    rel_path = str(fpath.relative_to(project_path))

    try:
        content = fpath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return FileMetrics(path=str(fpath), rel_path=rel_path)

    lines = content.split("\n")
    total_lines = len(lines)

    # Count blank/comment/code lines
    blank = 0
    comments = 0
    code = 0
    todos = 0
    fixmes = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank += 1
        elif stripped.startswith("#"):
            comments += 1
            if "TODO" in stripped.upper():
                todos += 1
            if "FIXME" in stripped.upper() or "HACK" in stripped.upper():
                fixmes += 1
        else:
            code += 1

    # AST analysis
    functions = 0
    classes = 0
    imports = 0
    try_count = 0
    except_count = 0
    type_hinted = 0
    has_docstring = False

    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                functions += 1
                if node.returns:
                    type_hinted += 1
                for arg in node.args.args + node.args.kwonlyargs:
                    if arg.annotation:
                        type_hinted += 1
                        break
            elif isinstance(node, ast.ClassDef):
                classes += 1
                if ast.get_docstring(node):
                    has_docstring = True
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports += 1
            elif isinstance(node, ast.Try):
                try_count += 1
                except_count += len(node.handlers)
    except SyntaxError:
        pass

    # Git change frequency (last 30 days)
    change_freq = 0
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--since=30 days ago", "--", str(fpath)],
            capture_output=True, text=True,
            cwd=str(project_path),
        )
        change_freq = len([l for l in result.stdout.strip().split("\n") if l.strip()])
    except Exception:
        pass

    # Last modified
    try:
        stat = fpath.stat()
        last_mod = datetime.fromtimestamp(stat.st_mtime)
    except Exception:
        last_mod = datetime.now()

    return FileMetrics(
        path=str(fpath),
        rel_path=rel_path,
        lines=total_lines,
        blank_lines=blank,
        comment_lines=comments,
        code_lines=code,
        functions=functions,
        classes=classes,
        imports=imports,
        try_count=try_count,
        except_count=except_count,
        todo_count=todos,
        fixme_count=fixmes,
        type_hinted_functions=type_hinted,
        total_functions=functions,
        has_docstring=has_docstring,
        last_modified=last_mod,
        change_frequency=change_freq,
    )
