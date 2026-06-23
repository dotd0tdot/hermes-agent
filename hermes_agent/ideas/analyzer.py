"""
Analyze file metrics to find code quality issues and improvement opportunities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from hermes_agent.ideas.scanner import FileMetrics

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    """A single code quality issue found during analysis."""
    file: str
    rel_path: str
    category: str       # "god_file", "high_complexity", "no_type_hints", "no_tests", "todo", "duplication"
    severity: str       # "high", "medium", "low"
    title: str
    description: str
    metric_value: Optional[float] = None
    suggestion: str = ""


# Thresholds
GOD_FILE_LINES = 5000
GOD_FILE_FUNCTIONS = 50
HIGH_TRY_COUNT = 50
LOW_TYPE_HINT_RATIO = 0.3
HIGH_CHANGE_FREQUENCY = 20


def analyze_metrics(metrics: list[FileMetrics]) -> list[Finding]:
    """Analyze collected metrics and return findings."""
    findings = []

    for fm in metrics:
        # God file detection
        if fm.lines > GOD_FILE_LINES:
            findings.append(Finding(
                file=fm.path,
                rel_path=fm.rel_path,
                category="god_file",
                severity="high",
                title=f"God file: {fm.rel_path}",
                description=f"{fm.lines} lines, {fm.functions} functions, {fm.classes} classes",
                metric_value=float(fm.lines),
                suggestion=f"Split into {max(2, fm.lines // 2000)} modules",
            ))
        elif fm.lines > 2000 and fm.functions > GOD_FILE_FUNCTIONS:
            findings.append(Finding(
                file=fm.path,
                rel_path=fm.rel_path,
                category="god_file",
                severity="medium",
                title=f"Large file: {fm.rel_path}",
                description=f"{fm.lines} lines, {fm.functions} functions",
                metric_value=float(fm.lines),
                suggestion="Consider extracting modules",
            ))

        # High try/except count
        if fm.try_count > HIGH_TRY_COUNT:
            findings.append(Finding(
                file=fm.path,
                rel_path=fm.rel_path,
                category="error_handling",
                severity="medium",
                title=f"Excessive try/except: {fm.rel_path}",
                description=f"{fm.try_count} try blocks, {fm.except_count} except handlers",
                metric_value=float(fm.try_count),
                suggestion="Unify error handling with decorator/context manager",
            ))

        # Missing type hints
        if fm.total_functions > 5 and fm.type_hinted_functions == 0:
            findings.append(Finding(
                file=fm.path,
                rel_path=fm.rel_path,
                category="no_type_hints",
                severity="low",
                title=f"No type hints: {fm.rel_path}",
                description=f"{fm.total_functions} functions without type annotations",
                metric_value=0.0,
                suggestion="Add type hints for public functions",
            ))
        elif fm.total_functions > 5:
            ratio = fm.type_hinted_functions / max(fm.total_functions, 1)
            if ratio < LOW_TYPE_HINT_RATIO:
                findings.append(Finding(
                    file=fm.path,
                    rel_path=fm.rel_path,
                    category="no_type_hints",
                    severity="low",
                    title=f"Low type hint coverage: {fm.rel_path}",
                    description=f"{fm.type_hinted_functions}/{fm.total_functions} functions typed ({ratio:.0%})",
                    metric_value=ratio,
                    suggestion="Add type hints to public API functions",
                ))

        # TODOs and FIXMEs
        if fm.todo_count > 3:
            findings.append(Finding(
                file=fm.path,
                rel_path=fm.rel_path,
                category="todo",
                severity="low",
                title=f"TODO backlog: {fm.rel_path}",
                description=f"{fm.todo_count} TODO items",
                metric_value=float(fm.todo_count),
                suggestion="Address or track in issue tracker",
            ))

        if fm.fixme_count > 0:
            findings.append(Finding(
                file=fm.path,
                rel_path=fm.rel_path,
                category="todo",
                severity="medium",
                title=f"FIXME items: {fm.rel_path}",
                description=f"{fm.fixme_count} FIXME/HACK items",
                metric_value=float(fm.fixme_count),
                suggestion="Fix or create issues for tracking",
            ))

        # High change frequency (hot spot)
        if fm.change_frequency > HIGH_CHANGE_FREQUENCY:
            findings.append(Finding(
                file=fm.path,
                rel_path=fm.rel_path,
                category="hot_spot",
                severity="medium",
                title=f"Hot spot: {fm.rel_path}",
                description=f"{fm.change_frequency} commits in last 30 days",
                metric_value=float(fm.change_frequency),
                suggestion="Consider stabilizing — frequent changes indicate instability",
            ))

    # Sort by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (severity_order.get(f.severity, 3), f.category))

    return findings
