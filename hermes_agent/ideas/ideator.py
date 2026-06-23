"""
Generate improvement ideas from code analysis findings.

This module creates structured ideas from analyzer findings.
LLM-based generation is done by the agent itself (the calling Hermes instance).
This module handles the structured, deterministic part.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from hermes_agent.ideas.analyzer import Finding
from hermes_agent.ideas.scanner import FileMetrics

logger = logging.getLogger(__name__)


@dataclass
class Idea:
    """A single improvement idea."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    project: str = "hermes"
    title: str = ""
    description: str = ""
    category: str = ""       # "refactor", "feature", "fix", "optimize", "test", "type_safety"
    priority: int = 3        # 1-5 (5 = highest)
    complexity: str = "medium"  # "low", "medium", "high"
    target_files: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    status: str = "new"      # "new", "approved", "in_progress", "done", "rejected"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    applied_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Idea:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def generate_ideas(
    findings: list[Finding],
    metrics: list[FileMetrics],
    project: str = "hermes",
) -> list[Idea]:
    """
    Generate improvement ideas from findings.

    This creates structured ideas from deterministic analysis.
    For LLM-powered creative ideas, the calling agent extends these.
    """
    ideas = []
    seen_categories = set()

    # Group findings by category
    by_category: dict[str, list[Finding]] = {}
    for f in findings:
        by_category.setdefault(f.category, []).append(f)

    # Generate ideas per category
    for category, cat_findings in by_category.items():
        if category == "god_file":
            idea = _idea_from_god_files(cat_findings, project)
        elif category == "error_handling":
            idea = _idea_from_error_handling(cat_findings, project)
        elif category == "no_type_hints":
            idea = _idea_from_type_hints(cat_findings, project)
        elif category == "todo":
            idea = _idea_from_todos(cat_findings, project)
        elif category == "hot_spot":
            idea = _idea_from_hot_spots(cat_findings, project)
        else:
            continue

        if idea:
            ideas.append(idea)
            seen_categories.add(category)

    # Add cross-cutting ideas based on overall metrics
    cross_cutting = _cross_cutting_ideas(metrics, project)
    ideas.extend(cross_cutting)

    # Sort by priority (highest first)
    ideas.sort(key=lambda i: -i.priority)

    return ideas


def _idea_from_god_files(findings: list[Finding], project: str) -> Idea | None:
    """Create idea for god file refactoring."""
    if not findings:
        return None

    high = [f for f in findings if f.severity == "high"]
    if not high:
        return None

    target_files = [f.rel_path for f in high]
    total_lines = sum(int(f.metric_value or 0) for f in high)

    return Idea(
        project=project,
        title=f"Decompose {len(high)} god file(s) ({total_lines:.0f} total lines)",
        description=(
            f"Found {len(high)} files exceeding 5000 lines. "
            f"God files are hard to navigate, test, and maintain. "
            f"Split into focused modules by domain."
        ),
        category="refactor",
        priority=5,
        complexity="high",
        target_files=target_files,
        findings=[f.description for f in high],
        plan=[
            "Identify logical boundaries in each god file",
            "Create new module files (e.g., cli/core.py, cli/statusbar.py)",
            "Move functions/classes to appropriate modules",
            "Update imports across the project",
            "Run tests to verify no regressions",
        ],
    )


def _idea_from_error_handling(findings: list[Finding], project: str) -> Idea | None:
    """Create idea for error handling unification."""
    if not findings:
        return None

    total_trys = sum(int(f.metric_value or 0) for f in findings)
    target_files = list(set(f.rel_path for f in findings))

    return Idea(
        project=project,
        title=f"Unify error handling ({total_trys} try/except blocks)",
        description=(
            f"Found {total_trys} try/except blocks across {len(target_files)} files. "
            f"Most use identical patterns. Create a @resilient decorator or "
            f"context manager for consistency."
        ),
        category="refactor",
        priority=3,
        complexity="medium",
        target_files=target_files[:5],
        findings=[f.description for f in findings],
        plan=[
            "Design @resilient decorator with configurable retry/log/skip",
            "Replace bare try/except in process_loop and helpers",
            "Add structured error types for common failures",
            "Update tests to verify error paths",
        ],
    )


def _idea_from_type_hints(findings: list[Finding], project: str) -> Idea | None:
    """Create idea for adding type hints."""
    if not findings:
        return None

    target_files = list(set(f.rel_path for f in findings))
    total_untyped = sum(int(f.metric_value or 0) for f in findings if f.metric_value == 0)

    return Idea(
        project=project,
        title=f"Add type hints to {len(target_files)} file(s)",
        description=(
            f"Found {len(target_files)} files with low type hint coverage. "
            f"Type hints improve IDE support, catch bugs early, and serve as documentation."
        ),
        category="type_safety",
        priority=2,
        complexity="low",
        target_files=target_files[:5],
        findings=[f.description for f in findings],
        plan=[
            "Add type hints to public function signatures",
            "Add return types to functions",
            "Create dataclass for StatusBarSnapshot (replace dict)",
            "Run mypy to verify",
        ],
    )


def _idea_from_todos(findings: list[Finding], project: str) -> Idea | None:
    """Create idea for addressing TODO backlog."""
    if not findings:
        return None

    total_todos = sum(int(f.metric_value or 0) for f in findings)
    target_files = list(set(f.rel_path for f in findings))

    return Idea(
        project=project,
        title=f"Address {total_todos} TODO/FIXME items",
        description=(
            f"Found {total_todos} TODO/FIXME/HACK items across {len(target_files)} files. "
            f"Track in issue tracker or resolve."
        ),
        category="fix",
        priority=2,
        complexity="low",
        target_files=target_files,
        findings=[f.description for f in findings],
        plan=[
            "Review each TODO/FIXME",
            "Create issues for trackable items",
            "Fix simple items immediately",
            "Remove stale TODOs",
        ],
    )


def _idea_from_hot_spots(findings: list[Finding], project: str) -> Idea | None:
    """Create idea for stabilizing hot spots."""
    if not findings:
        return None

    target_files = [f.rel_path for f in findings]

    return Idea(
        project=project,
        title=f"Stabilize {len(target_files)} hot spot file(s)",
        description=(
            f"Found {len(target_files)} files with >20 commits in 30 days. "
            f"Frequent changes suggest instability or unclear design."
        ),
        category="optimize",
        priority=3,
        complexity="medium",
        target_files=target_files,
        findings=[f.description for f in findings],
        plan=[
            "Review recent changes for patterns",
            "Identify root cause of frequent modifications",
            "Refactor for stability",
            "Add tests to prevent regressions",
        ],
    )


def _cross_cutting_ideas(metrics: list[FileMetrics], project: str) -> list[Idea]:
    """Generate ideas that span multiple files."""
    ideas = []

    # Check for missing test coverage
    test_files = [m for m in metrics if "test" in m.rel_path.lower()]
    source_files = [m for m in metrics if "test" not in m.rel_path.lower() and m.lines > 100]

    if source_files and len(test_files) < len(source_files) * 0.1:
        ideas.append(Idea(
            project=project,
            title=f"Add tests for {len(source_files)} source files",
            description=(
                f"Only {len(test_files)} test files for {len(source_files)} source files. "
                f"Low test coverage increases regression risk."
            ),
            category="test",
            priority=3,
            complexity="medium",
            target_files=[],
            plan=[
                "Identify critical untested modules",
                "Write unit tests for core logic",
                "Add integration tests for key workflows",
                "Set up CI to enforce coverage",
            ],
        ))

    return ideas
