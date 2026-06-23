"""
Idea generator for Hermes Agent codebase.

Scans the project, analyzes code metrics, generates improvement ideas,
and can apply them via patches.

Usage:
    /ideas                     — scan current project
    /ideas hermes              — scan Hermes Agent
    /ideas /path/to/project    — scan any project
    /ideas hermes --local      — only local (uncommitted) changes
    /ideas hermes --file cli.py — specific file
    /ideas apply 3             — start implementing idea #3
    /ideas history             — show past ideas
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Project aliases — short names to absolute paths
PROJECT_ALIASES = {
    "hermes": str(Path.home() / ".hermes" / "hermes-agent"),
}


def resolve_project(name_or_path: str) -> Path:
    """Resolve a project alias or path to an absolute Path."""
    if name_or_path in PROJECT_ALIASES:
        path = Path(PROJECT_ALIASES[name_or_path])
    else:
        path = Path(name_or_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Project not found: {path}")
    return path


def run_ideas_command(
    project: str,
    scope: str = "full",
    target: str | None = None,
    apply_id: int | None = None,
    show_history: bool = False,
) -> str:
    """Main entry point — called from HermesCLI.process_command()."""
    from hermes_agent.ideas.scanner import scan_project
    from hermes_agent.ideas.analyzer import analyze_metrics
    from hermes_agent.ideas.ideator import generate_ideas
    from hermes_agent.ideas.formatter import format_ideas, format_history
    from hermes_agent.ideas.storage import IdeaStorage

    if show_history:
        storage = IdeaStorage()
        ideas = storage.list_ideas(project=project)
        return format_history(ideas)

    project_path = resolve_project(project)

    # Phase 1: Scan
    print(f"  Scanning {project_path}...")
    t0 = time.monotonic()
    metrics = scan_project(project_path, scope=scope, target=target)
    scan_time = time.monotonic() - t0
    print(f"  Found {len(metrics)} files ({scan_time:.1f}s)")

    if not metrics:
        return "  No Python files found in the specified scope."

    # Phase 2: Analyze
    print("  Analyzing...")
    t0 = time.monotonic()
    findings = analyze_metrics(metrics)
    analyze_time = time.monotonic() - t0
    print(f"  Found {len(findings)} issues ({analyze_time:.1f}s)")

    if not findings:
        return "  Code looks clean! No improvement ideas at this time."

    # Phase 3: Generate ideas
    print("  Generating ideas...")
    t0 = time.monotonic()
    ideas = generate_ideas(findings, metrics, project)
    ideate_time = time.monotonic() - t0
    print(f"  Generated {len(ideas)} ideas ({ideate_time:.1f}s)")

    # Phase 4: Store
    storage = IdeaStorage()
    for idea in ideas:
        storage.save_idea(idea)

    # Phase 5: Format output
    output = format_ideas(ideas, project)
    return output


def apply_idea(idea_id: int, project: str = "hermes") -> str:
    """Start implementing an idea by creating a plan."""
    from hermes_agent.ideas.storage import IdeaStorage

    storage = IdeaStorage()
    idea = storage.get_idea(idea_id, project=project)
    if not idea:
        return f"  Idea #{idea_id} not found in project '{project}'. Use /ideas history to see available ideas."

    idea["status"] = "in_progress"
    storage.update_idea(idea_id, project=project, status="in_progress")

    # Generate implementation plan
    plan = _create_implementation_plan(idea)

    # Save plan
    plans_dir = Path.home() / ".hermes" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plans_dir / f"idea-{idea_id:03d}.md"
    plan_path.write_text(plan, encoding="utf-8")

    output = [
        f"  Idea #{idea_id}: {idea['title']}",
        f"  Status: in_progress",
        f"  Plan saved to: {plan_path}",
        "",
        "  Next steps:",
        "  1. Review the plan",
        "  2. Start implementing with /plan or direct commands",
        "",
        "  Plan preview:",
        plan[:500],
    ]
    return "\n".join(output)


def _create_implementation_plan(idea: dict) -> str:
    """Create a markdown implementation plan from an idea."""
    lines = [
        f"# Plan: {idea['title']}",
        "",
        f"**Category:** {idea.get('category', 'N/A')}",
        f"**Priority:** {idea.get('priority', 'N/A')}",
        f"**Complexity:** {idea.get('complexity', 'N/A')}",
        "",
        "## Description",
        idea.get("description", ""),
        "",
        "## Target Files",
    ]
    for f in idea.get("target_files", []):
        lines.append(f"- `{f}`")

    lines.extend(["", "## Steps"])
    for i, step in enumerate(idea.get("plan", []), 1):
        lines.append(f"{i}. {step}")

    lines.extend([
        "",
        "## Acceptance Criteria",
        "- [ ] All existing tests pass",
        "- [ ] New tests added (if applicable)",
        "- [ ] No regressions",
        "",
        "## Notes",
        f"Generated: {datetime.now().isoformat()}",
    ])
    return "\n".join(lines)
