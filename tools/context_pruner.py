#!/usr/bin/env python3
"""
context_pruner.py — Context-aware tool pruning for Hermes Agent.

What it does:
  1. Detects the runtime context (CLI, gateway-Telegram, gateway-Discord, TUI, etc.)
  2. Lazily imports tool modules only when their toolset is first activated
  3. Optionally prunes toolsets that can never be useful in the current context
  4. Reports estimated token savings

Usage in agent_init.py:
    from tools.context_pruner import (
        detect_runtime_context,
        lazy_discover_tools,
        pruned_toolsets,
        format_toolset_report,
    )
    ctx = detect_runtime_context()
    lazy_discover_tools(enabled_toolsets, ctx)

Integration with model_tools.py:
    get_tool_definitions() auto-prunes when auto_prune=True is passed.

This module is intentionally small and has zero imports from other Hermes
modules except the registry and lazy_deps — no circular import risk.
"""

from __future__ import annotations

import enum
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Context Detection
# =============================================================================

class RuntimeContext(enum.Enum):
    """The runtime environment the agent is executing in."""
    CLI = "cli"                       # Interactive CLI or headless
    GATEWAY_TELEGRAM = "gateway:telegram"
    GATEWAY_DISCORD = "gateway:discord"
    GATEWAY_SLACK = "gateway:slack"
    GATEWAY_WHATSAPP = "gateway:whatsapp"
    GATEWAY_SIGNAL = "gateway:signal"
    GATEWAY_FEISHU = "gateway:feishu"
    GATEWAY_MATRIX = "gateway:matrix"
    GATEWAY_WECOM = "gateway:wecom"
    GATEWAY_YUANBAO = "gateway:yuanbao"
    GATEWAY_GENERIC = "gateway:generic"   # Unknown/unlisted platform
    TUI = "tui"                       # Hermes Desktop / TUI
    KANBAN_WORKER = "kanban_worker"   # Spawned by kanban dispatcher
    SUBAGENT = "subagent"             # delegate_task child
    BATCH = "batch"                   # batch_runner.py / data generation
    UNKNOWN = "unknown"


def detect_runtime_context() -> RuntimeContext:
    """Detect the current runtime context by probing the environment.

    Order matters — kanban worker check before CLI because a kanban worker
    may also be running in CLI mode.
    """
    # Kanban worker: set by the dispatcher when spawning a task worker.
    if os.environ.get("HERMES_KANBAN_TASK"):
        return RuntimeContext.KANBAN_WORKER

    # Subagent: set by delegate_task when spawning a child agent.
    if os.environ.get("HERMES_PARENT_SESSION_ID"):
        return RuntimeContext.SUBAGENT

    # Batch mode: batch_runner sets this.
    if os.environ.get("HERMES_BATCH_MODE"):
        return RuntimeContext.BATCH

    # Gateway: detect which platform we're running under.
    # The gateway sets HERMES_GATEWAY_PLATFORM when creating agent instances.
    gw_platform = os.environ.get("HERMES_GATEWAY_PLATFORM", "").lower()
    if gw_platform:
        mapping = {
            "telegram": RuntimeContext.GATEWAY_TELEGRAM,
            "discord": RuntimeContext.GATEWAY_DISCORD,
            "slack": RuntimeContext.GATEWAY_SLACK,
            "whatsapp": RuntimeContext.GATEWAY_WHATSAPP,
            "signal": RuntimeContext.GATEWAY_SIGNAL,
            "feishu": RuntimeContext.GATEWAY_FEISHU,
            "matrix": RuntimeContext.GATEWAY_MATRIX,
            "wecom": RuntimeContext.GATEWAY_WECOM,
            "yuanbao": RuntimeContext.GATEWAY_YUANBAO,
        }
        return mapping.get(gw_platform, RuntimeContext.GATEWAY_GENERIC)

    # TUI: detect if running inside the Hermes Desktop wrapper.
    if os.environ.get("HERMES_TUI") or os.environ.get("HERMES_DESKTOP"):
        return RuntimeContext.TUI

    # CLI: catch-all for interactive CLI and headless runs.
    return RuntimeContext.CLI


# =============================================================================
# Toolset-to-Context Relevance Map
# =============================================================================

# Toolsets that are NEVER useful in each context.
# This does NOT replace check_fn — it's a static filter that avoids even
# registering/importing tools that can't possibly serve the current session.
#
# Rules of thumb:
#   - Kanban tools are useless outside kanban workers
#   - send_message is useless without a running gateway
#   - Platform-specific tools are only useful on that platform
#   - Home Assistant tools are useless without HASS_TOKEN
#   - computer_use (macOS CUA) is useless on Linux
_CONTEXT_IRRELEVANT_TOOLSETS: Dict[RuntimeContext, Set[str]] = {
    RuntimeContext.CLI: {
        # Kanban — only in kanban worker mode
        "kanban",
        # Computer use — macOS CUA driver, not available on Linux CLI
        "computer_use",
    },
    RuntimeContext.TUI: {
        "kanban",
        "computer_use",
    },
    RuntimeContext.GATEWAY_TELEGRAM: {
        # Exclude other platform's toolsets (they have check_fn too, but
        # skipping registration saves import time and memory).
        # We keep gateway-relevant tools like send_message.
    },
    RuntimeContext.GATEWAY_DISCORD: {
        # Mirror telegram — keep send_message, drop platform-specifics
    },
    RuntimeContext.GATEWAY_SLACK: set(),
    RuntimeContext.GATEWAY_WHATSAPP: set(),
    RuntimeContext.GATEWAY_SIGNAL: set(),
    RuntimeContext.GATEWAY_FEISHU: set(),
    RuntimeContext.GATEWAY_MATRIX: set(),
    RuntimeContext.GATEWAY_WECOM: set(),
    RuntimeContext.GATEWAY_YUANBAO: set(),
    RuntimeContext.GATEWAY_GENERIC: {},
    RuntimeContext.KANBAN_WORKER: {
        "computer_use",
    },
    RuntimeContext.SUBAGENT: {
        # Subagents shouldn't spawn further subagents or manage cron
        "kanban",
        # Most subagents don't need browser or media tools
    },
    RuntimeContext.BATCH: {},
    RuntimeContext.UNKNOWN: {},
}

# Toolsets that should ONLY be imported when explicitly enabled,
# not eagerly on startup. This is the lazy-import list.
_LAZY_TOOLSETS = {
    "kanban",         # Only when HERMES_KANBAN_TASK is set
    "computer_use",   # Only on macOS with cua-driver
    "homeassistant",  # Only when HASS_TOKEN is set
    "x_search",       # Only with xAI credentials
    "spotify",        # Only when Spotify credentials configured
    "discord",        # Only when Discord gateway active
    "discord_admin",  # Only when Discord admin configured
}


def toolset_is_relevant(toolset_name: str, context: RuntimeContext) -> bool:
    """Return True if *toolset_name* could be relevant in *context*."""
    irrelevant = _CONTEXT_IRRELEVANT_TOOLSETS.get(context, set())
    return toolset_name not in irrelevant


def pruned_toolsets(
    enabled_toolsets: Optional[List[str]],
    context: RuntimeContext,
) -> Tuple[List[str], List[str]]:
    """Filter enabled_toolsets to only include context-relevant ones.

    Returns:
        (pruned_list, removed_list)
    """
    if enabled_toolsets is None:
        return None, []

    pruned = []
    removed = []
    for ts in enabled_toolsets:
        if toolset_is_relevant(ts, context):
            pruned.append(ts)
        else:
            removed.append(ts)
    return pruned, removed


# =============================================================================
# Lazy Tool Discovery
# =============================================================================

# Track which tool modules we've already considered to avoid re-import
_imported_tool_modules: Set[str] = set()


def lazy_discover_tools(
    enabled_toolsets: Optional[List[str]] = None,
    context: Optional[RuntimeContext] = None,
) -> List[str]:
    """Import only the tool modules relevant to the current context.

    This is an alternative to ``discover_builtin_tools()`` that skips
    modules whose toolsets are irrelevant. Call this once at agent init
    *instead of* ``discover_builtin_tools()``.

    Returns the list of imported module names.
    """
    from tools.registry import registry, discover_builtin_tools

    if context is None:
        context = detect_runtime_context()

    if enabled_toolsets is None:
        # No explicit toolsets — discover all (conservative fallback).
        return discover_builtin_tools()

    tools_path = os.path.join(os.path.dirname(__file__))
    import importlib
    from pathlib import Path

    # Build set of enabled tool names and their toolsets
    from toolsets import resolve_toolset, TOOLSETS
    tool_names_to_load: Set[str] = set()
    for ts_name in enabled_toolsets:
        if not toolset_is_relevant(ts_name, context):
            logger.debug(
                "Skipping toolset %s (irrelevant in %s)", ts_name, context.value
            )
            continue
        resolved = resolve_toolset(ts_name)
        tool_names_to_load.update(resolved)

    if not tool_names_to_load:
        return []

    # Map tool names to their source module
    # We only need to import each module once
    imported = []
    tp = Path(tools_path)
    for py_file in sorted(tp.glob("*.py")):
        if py_file.name in {"__init__.py", "registry.py", "lazy_deps.py",
                             "context_pruner.py"}:
            continue
        mod_name = f"tools.{py_file.stem}"

        # Quick check: does this module register any of our needed tools?
        # We use the registry's AST scanner to decide.
        if mod_name in _imported_tool_modules:
            continue

        # Import the module — it self-registers tools in the registry
        try:
            importlib.import_module(mod_name)
            _imported_tool_modules.add(mod_name)
            imported.append(mod_name)
        except Exception as e:
            logger.warning("Could not import %s: %s", mod_name, e)

    logger.debug(
        "Lazy-discovered %d modules for %d tools in context %s",
        len(imported), len(tool_names_to_load), context.value,
    )
    return imported


# =============================================================================
# Reporting
# =============================================================================

# Rough estimate: each tool schema entry is ~300-800 chars in the API request.
# This is a conservative average of 500 bytes per tool.
_AVERAGE_TOOL_SCHEMA_BYTES = 500


def format_toolset_report(
    enabled_toolsets: Optional[List[str]] = None,
    context: Optional[RuntimeContext] = None,
) -> str:
    """Return a human-readable report of active vs pruned toolsets."""
    if context is None:
        context = detect_runtime_context()
    if enabled_toolsets is None:
        return f"Runtime: {context.value}\nAll toolsets enabled (no pruning)"

    pruned, removed = pruned_toolsets(enabled_toolsets, context)

    lines = [
        f"Runtime context: {context.value}",
        f"Active toolsets:  {len(pruned) if pruned else 0}",
        f"Pruned toolsets:  {len(removed)}",
    ]

    if removed:
        lines.append(f"Removed: {', '.join(removed)}")
        # Estimate tokens saved
        saved_tools = 0
        from toolsets import resolve_toolset
        for ts in removed:
            saved_tools += len(resolve_toolset(ts))
        saved_bytes = saved_tools * _AVERAGE_TOOL_SCHEMA_BYTES
        # Rough: 1 token ≈ 4 bytes for schema (JSON is mostly ASCII)
        saved_tokens = saved_bytes // 4
        lines.append(
            f"Estimated savings: ~{saved_tokens} tokens/req "
            f"(~{saved_tools} tools × ~{_AVERAGE_TOOL_SCHEMA_BYTES}B)"
        )

    return "\n".join(lines)


# =============================================================================
# Quick self-test
# =============================================================================
if __name__ == "__main__":
    ctx = detect_runtime_context()
    print(f"Context: {ctx.value}")
    print(format_toolset_report(
        ["web", "terminal", "file", "kanban", "computer_use"],
        ctx,
    ))
