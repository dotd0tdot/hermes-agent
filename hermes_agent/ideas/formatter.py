"""
Format ideas and history for terminal display.
"""

from __future__ import annotations

from hermes_agent.ideas.ideator import Idea


SEVERITY_COLORS = {
    "high": "\033[91m",    # red
    "medium": "\033[93m",  # yellow
    "low": "\033[92m",     # green
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def format_ideas(ideas: list[Idea], project: str) -> str:
    """Format a list of ideas for terminal display."""
    if not ideas:
        return "  No ideas generated."

    lines = [
        "",
        f"  {BOLD}{len(ideas)} improvement idea(s) for {project}{RESET}",
        "",
    ]

    for i, idea in enumerate(ideas, 1):
        priority_marker = "🔴" if idea.priority >= 4 else "🟡" if idea.priority >= 2 else "🟢"
        complexity_badge = f"[{idea.complexity}]"

        lines.append(f"  {BOLD}#{i}{RESET} {priority_marker} {idea.title}  {DIM}{complexity_badge}{RESET}")
        lines.append(f"     Category: {idea.category}")
        lines.append(f"     {DIM}{idea.description}{RESET}")

        if idea.target_files:
            lines.append(f"     Files: {', '.join(idea.target_files[:3])}")
            if len(idea.target_files) > 3:
                lines.append(f"     {DIM}...and {len(idea.target_files) - 3} more{RESET}")

        if idea.plan:
            lines.append(f"     {DIM}Plan: {len(idea.plan)} steps{RESET}")

        lines.append(f"     → /ideas apply {i}")
        lines.append("")

    lines.append(f"  {DIM}Use /ideas apply N to start implementing idea #{RESET}")
    return "\n".join(lines)


def format_history(ideas: list[dict]) -> str:
    """Format idea history for display."""
    if not ideas:
        return "  No ideas in history."

    lines = [
        "",
        f"  {BOLD}Idea History ({len(ideas)} items){RESET}",
        "",
    ]

    for idea in ideas[:20]:
        status = idea.get("status", "new")
        status_marker = {
            "new": "📝",
            "approved": "✅",
            "in_progress": "🔨",
            "done": "🎉",
            "rejected": "❌",
        }.get(status, "❓")

        title = idea.get("title", "Untitled")[:60]
        created = idea.get("created_at", "")[:10]

        lines.append(f"  {status_marker} [{status:>12}] {title}")
        lines.append(f"     {DIM}created: {created}{RESET}")

    return "\n".join(lines)
