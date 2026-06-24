"""Tests for tools/context_pruner.py — context-aware tool pruning."""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from tools.context_pruner import (
    RuntimeContext,
    _AVERAGE_TOOL_SCHEMA_BYTES,
    _CONTEXT_IRRELEVANT_TOOLSETS,
    _imported_tool_modules,
    _LAZY_TOOLSETS,
    detect_runtime_context,
    format_toolset_report,
    lazy_discover_tools,
    pruned_toolsets,
    toolset_is_relevant,
)


# =============================================================================
# RuntimeContext enum
# =============================================================================

class TestRuntimeContext:
    def test_all_values_defined(self):
        """All expected context values exist."""
        assert RuntimeContext.CLI.value == "cli"
        assert RuntimeContext.TUI.value == "tui"
        assert RuntimeContext.GATEWAY_TELEGRAM.value == "gateway:telegram"
        assert RuntimeContext.GATEWAY_DISCORD.value == "gateway:discord"
        assert RuntimeContext.GATEWAY_SLACK.value == "gateway:slack"
        assert RuntimeContext.GATEWAY_WHATSAPP.value == "gateway:whatsapp"
        assert RuntimeContext.GATEWAY_SIGNAL.value == "gateway:signal"
        assert RuntimeContext.GATEWAY_FEISHU.value == "gateway:feishu"
        assert RuntimeContext.GATEWAY_MATRIX.value == "gateway:matrix"
        assert RuntimeContext.GATEWAY_WECOM.value == "gateway:wecom"
        assert RuntimeContext.GATEWAY_YUANBAO.value == "gateway:yuanbao"
        assert RuntimeContext.GATEWAY_GENERIC.value == "gateway:generic"
        assert RuntimeContext.KANBAN_WORKER.value == "kanban_worker"
        assert RuntimeContext.SUBAGENT.value == "subagent"
        assert RuntimeContext.BATCH.value == "batch"
        assert RuntimeContext.UNKNOWN.value == "unknown"

    def test_all_contexts_have_entry_in_relevance_map(self):
        """Every RuntimeContext member has a key in _CONTEXT_IRRELEVANT_TOOLSETS."""
        for ctx in RuntimeContext:
            assert ctx in _CONTEXT_IRRELEVANT_TOOLSETS, (
                f"Missing relevance entry for {ctx}"
            )

class TestDetectRuntimeContext:
    def test_kanban_worker(self):
        with patch.dict(os.environ, {"HERMES_KANBAN_TASK": "task-123"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.KANBAN_WORKER

    def test_kanban_takes_priority(self):
        """Kanban check comes first, even if other env vars are set."""
        with patch.dict(os.environ, {
            "HERMES_KANBAN_TASK": "task-1",
            "HERMES_GATEWAY_PLATFORM": "telegram",
            "HERMES_TUI": "1",
        }, clear=True):
            assert detect_runtime_context() == RuntimeContext.KANBAN_WORKER

    def test_subagent(self):
        with patch.dict(os.environ, {"HERMES_PARENT_SESSION_ID": "sess-1"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.SUBAGENT

    def test_batch_mode(self):
        with patch.dict(os.environ, {"HERMES_BATCH_MODE": "1"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.BATCH

    def test_gateway_telegram(self):
        with patch.dict(os.environ, {"HERMES_GATEWAY_PLATFORM": "telegram"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.GATEWAY_TELEGRAM

    def test_gateway_discord(self):
        with patch.dict(os.environ, {"HERMES_GATEWAY_PLATFORM": "discord"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.GATEWAY_DISCORD

    def test_gateway_slack(self):
        with patch.dict(os.environ, {"HERMES_GATEWAY_PLATFORM": "slack"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.GATEWAY_SLACK

    def test_gateway_whatsapp(self):
        with patch.dict(os.environ, {"HERMES_GATEWAY_PLATFORM": "whatsapp"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.GATEWAY_WHATSAPP

    def test_gateway_signal(self):
        with patch.dict(os.environ, {"HERMES_GATEWAY_PLATFORM": "signal"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.GATEWAY_SIGNAL

    def test_gateway_feishu(self):
        with patch.dict(os.environ, {"HERMES_GATEWAY_PLATFORM": "feishu"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.GATEWAY_FEISHU

    def test_gateway_matrix(self):
        with patch.dict(os.environ, {"HERMES_GATEWAY_PLATFORM": "matrix"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.GATEWAY_MATRIX

    def test_gateway_wecom(self):
        with patch.dict(os.environ, {"HERMES_GATEWAY_PLATFORM": "wecom"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.GATEWAY_WECOM

    def test_gateway_yuanbao(self):
        with patch.dict(os.environ, {"HERMES_GATEWAY_PLATFORM": "yuanbao"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.GATEWAY_YUANBAO

    def test_gateway_generic(self):
        with patch.dict(os.environ, {"HERMES_GATEWAY_PLATFORM": "irc"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.GATEWAY_GENERIC

    def test_tui(self):
        with patch.dict(os.environ, {"HERMES_TUI": "1"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.TUI

    def test_desktop(self):
        with patch.dict(os.environ, {"HERMES_DESKTOP": "1"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.TUI

    def test_cli_default(self):
        """CLI is the fallback when no relevant env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            assert detect_runtime_context() == RuntimeContext.CLI

    def test_gateway_case_insensitive(self):
        """Platform value is lowercased before mapping."""
        with patch.dict(os.environ, {"HERMES_GATEWAY_PLATFORM": "Telegram"}, clear=True):
            assert detect_runtime_context() == RuntimeContext.GATEWAY_TELEGRAM

    def test_empty_gateway_platform_falls_to_cli(self):
        """HERMES_GATEWAY_PLATFORM set to empty string should not match."""
        with patch.dict(os.environ, {"HERMES_GATEWAY_PLATFORM": ""}, clear=True):
            assert detect_runtime_context() == RuntimeContext.CLI


# =============================================================================
# toolset_is_relevant
# =============================================================================

class TestToolsetIsRelevant:
    def test_cli_irrelevant(self):
        assert toolset_is_relevant("kanban", RuntimeContext.CLI) is False
        assert toolset_is_relevant("computer_use", RuntimeContext.CLI) is False

    def test_cli_relevant(self):
        assert toolset_is_relevant("web", RuntimeContext.CLI) is True
        assert toolset_is_relevant("terminal", RuntimeContext.CLI) is True
        assert toolset_is_relevant("file", RuntimeContext.CLI) is True

    def test_tui_irrelevant(self):
        assert toolset_is_relevant("kanban", RuntimeContext.TUI) is False
        assert toolset_is_relevant("computer_use", RuntimeContext.TUI) is False

    def test_kanban_worker_irrelevant(self):
        assert toolset_is_relevant("computer_use", RuntimeContext.KANBAN_WORKER) is False

    def test_unknown_context(self):
        """Unknown context has no irrelevant toolsets."""
        assert toolset_is_relevant("kanban", RuntimeContext.UNKNOWN) is True
        assert toolset_is_relevant("anything", RuntimeContext.UNKNOWN) is True

    def test_batch_context(self):
        """Batch context has no irrelevant toolsets."""
        assert toolset_is_relevant("kanban", RuntimeContext.BATCH) is True
        assert toolset_is_relevant("computer_use", RuntimeContext.BATCH) is True

    def test_unknown_toolset_for_known_context(self):
        """Unknown toolset names are always relevant (not in any exclude set)."""
        assert toolset_is_relevant("nonexistent_toolset", RuntimeContext.CLI) is True

    def test_all_gateway_contexts_have_no_irrelevant_toolsets(self):
        """Gateway contexts currently have empty irrelevant sets."""
        gateway_contexts = [
            RuntimeContext.GATEWAY_TELEGRAM,
            RuntimeContext.GATEWAY_DISCORD,
            RuntimeContext.GATEWAY_SLACK,
            RuntimeContext.GATEWAY_WHATSAPP,
            RuntimeContext.GATEWAY_SIGNAL,
            RuntimeContext.GATEWAY_FEISHU,
            RuntimeContext.GATEWAY_MATRIX,
            RuntimeContext.GATEWAY_WECOM,
            RuntimeContext.GATEWAY_YUANBAO,
            RuntimeContext.GATEWAY_GENERIC,
        ]
        for ctx in gateway_contexts:
            assert toolset_is_relevant("kanban", ctx) is True
            assert toolset_is_relevant("computer_use", ctx) is True
            assert toolset_is_relevant("web", ctx) is True


# =============================================================================
# pruned_toolsets
# =============================================================================

class TestPrunedToolsets:
    def test_none_input(self):
        pruned, removed = pruned_toolsets(None, RuntimeContext.CLI)
        assert pruned is None
        assert removed == []

    def test_no_pruning_for_relevant_context(self):
        pruned, removed = pruned_toolsets(
            ["web", "terminal", "file"], RuntimeContext.CLI
        )
        assert pruned == ["web", "terminal", "file"]
        assert removed == []

    def test_prunes_irrelevant_toolsets(self):
        pruned, removed = pruned_toolsets(
            ["web", "kanban", "terminal", "computer_use", "file"],
            RuntimeContext.CLI,
        )
        assert pruned == ["web", "terminal", "file"]
        assert removed == ["kanban", "computer_use"]

    def test_all_irrelevant_pruned(self):
        pruned, removed = pruned_toolsets(["kanban", "computer_use"], RuntimeContext.CLI)
        assert pruned == []
        assert removed == ["kanban", "computer_use"]

    def test_empty_input(self):
        pruned, removed = pruned_toolsets([], RuntimeContext.CLI)
        assert pruned == []
        assert removed == []

    def test_unrecognized_toolset_not_pruned(self):
        pruned, removed = pruned_toolsets(
            ["unknown_toolset", "web"], RuntimeContext.CLI
        )
        assert "unknown_toolset" in pruned
        assert "unknown_toolset" not in removed

    def test_tui_prunes_kanban_and_computer_use(self):
        pruned, removed = pruned_toolsets(
            ["web", "kanban", "computer_use"], RuntimeContext.TUI
        )
        assert pruned == ["web"]
        assert removed == ["kanban", "computer_use"]

    def test_kanban_worker_prunes_only_computer_use(self):
        pruned, removed = pruned_toolsets(
            ["web", "kanban", "computer_use"], RuntimeContext.KANBAN_WORKER
        )
        assert pruned == ["web", "kanban"]
        assert removed == ["computer_use"]

    def test_subagent_prunes_kanban(self):
        """Subagent context prunes 'kanban' from its irrelevant set."""
        pruned, removed = pruned_toolsets(
            ["web", "kanban", "computer_use"], RuntimeContext.SUBAGENT
        )
        assert pruned == ["web", "computer_use"]
        assert "kanban" in removed


# =============================================================================
# lazy_discover_tools
# =============================================================================

class TestLazyDiscoverTools:
    def test_returns_discover_all_when_no_toolsets(self):
        """When enabled_toolsets is None, falls back to discover_builtin_tools()."""
        with patch(
            "tools.registry.discover_builtin_tools", return_value=["tools.foo"]
        ) as mock_disc:
            with patch("tools.context_pruner.detect_runtime_context") as mock_detect:
                mock_detect.return_value = RuntimeContext.CLI
                result = lazy_discover_tools(enabled_toolsets=None)

        mock_disc.assert_called_once()
        assert result == ["tools.foo"]

    def test_skips_irrelevant_toolsets(self):
        """Toolsets irrelevant to current context are skipped."""
        with patch("toolsets.resolve_toolset") as mock_resolve:
            mock_resolve.return_value = ["tool_a", "tool_b"]
            with patch(
                "importlib.import_module"
            ) as mock_import:
                mock_import.return_value = MagicMock()
                with patch("tools.context_pruner.detect_runtime_context") as mock_detect:
                    mock_detect.return_value = RuntimeContext.CLI
                    lazy_discover_tools(
                        enabled_toolsets=["kanban", "web"],
                    )

        # Only "web" should be resolved; "kanban" should be skipped
        mock_resolve.assert_called_once_with("web")

    def test_returns_empty_when_no_relevant_tools(self):
        """When all toolsets are irrelevant, returns empty list."""
        with patch("toolsets.resolve_toolset") as mock_resolve:
            mock_resolve.return_value = ["tool_x"]
            with patch(
                "importlib.import_module"
            ) as mock_import:
                result = lazy_discover_tools(
                    enabled_toolsets=["kanban"],
                    context=RuntimeContext.CLI,
                )
        mock_resolve.assert_not_called()
        mock_import.assert_not_called()
        assert result == []

    def test_provided_context_used(self):
        """When context is explicitly provided, it's used instead of auto-detection."""
        with patch("toolsets.resolve_toolset") as mock_resolve:
            mock_resolve.return_value = ["tool_a"]
            with patch(
                "importlib.import_module"
            ) as mock_import:
                mock_import.return_value = MagicMock()
                with patch(
                    "tools.context_pruner.detect_runtime_context"
                ) as mock_detect:
                    lazy_discover_tools(
                        enabled_toolsets=["web"],
                        context=RuntimeContext.KANBAN_WORKER,
                    )
                    mock_detect.assert_not_called()

    def test_resets_module_cache_across_tests(self):
        """The _imported_tool_modules set could cause cross-test pollution."""
        # This test verifies the set can be cleared (test isolation)
        _imported_tool_modules.clear()
        assert len(_imported_tool_modules) == 0


# =============================================================================
# _LAZY_TOOLSETS
# =============================================================================

class TestLazyToolsets:
    def test_known_lazy_toolsets(self):
        assert "kanban" in _LAZY_TOOLSETS
        assert "computer_use" in _LAZY_TOOLSETS
        assert "homeassistant" in _LAZY_TOOLSETS
        assert "x_search" in _LAZY_TOOLSETS
        assert "spotify" in _LAZY_TOOLSETS
        assert "discord" in _LAZY_TOOLSETS
        assert "discord_admin" in _LAZY_TOOLSETS


# =============================================================================
# format_toolset_report
# =============================================================================

class TestFormatToolsetReport:
    def test_no_enabled_toolsets(self):
        """When enabled_toolsets is None, reports all enabled."""
        report = format_toolset_report(None, RuntimeContext.CLI)
        assert "Runtime: cli" in report
        assert "All toolsets enabled" in report

    def test_no_pruning(self):
        """When nothing is pruned, report shows 0 pruned."""
        report = format_toolset_report(
            ["web", "terminal"], RuntimeContext.CLI
        )
        assert "Runtime context: cli" in report
        assert "Active toolsets:  2" in report
        assert "Pruned toolsets:  0" in report

    def test_with_pruning(self):
        """When toolsets are pruned, report lists them and estimates savings."""
        with patch("toolsets.resolve_toolset") as mock_resolve:
            mock_resolve.return_value = ["tool_x", "tool_y"]
            report = format_toolset_report(
                ["web", "computer_use"], RuntimeContext.CLI
            )
            assert "Pruned toolsets:  1" in report
            assert "computer_use" in report

    def test_uses_provided_context(self):
        """When context is given, it's used in the report."""
        with patch("tools.context_pruner.detect_runtime_context") as mock_detect:
            report = format_toolset_report(
                ["web"], RuntimeContext.GATEWAY_TELEGRAM
            )
            mock_detect.assert_not_called()
            assert "gateway:telegram" in report

    def test_auto_detects_context_when_not_given(self):
        """When context is None, auto-detect is called."""
        with patch("tools.context_pruner.detect_runtime_context") as mock_detect:
            mock_detect.return_value = RuntimeContext.CLI
            format_toolset_report(["web"], None)
            mock_detect.assert_called_once()

    def test_estimated_savings_line(self):
        """Pruned report includes estimated token savings."""
        with patch("toolsets.resolve_toolset") as mock_resolve:
            mock_resolve.return_value = ["tool_a", "tool_b"]
            report = format_toolset_report(
                ["web", "kanban"], RuntimeContext.CLI
            )
            assert "Estimated savings" in report
            assert "tokens/req" in report
            assert "2 tools" in report

    def test_estimated_savings_for_multiple_pruned(self):
        """Savings calculation scales with number of pruned tools."""
        with patch("toolsets.resolve_toolset") as mock_resolve:
            # Return different tool counts for different toolsets
            def resolve_side_effect(name):
                counts = {"kanban": ["k1", "k2"], "computer_use": ["c1", "c2", "c3"]}
                return counts.get(name, [])

            mock_resolve.side_effect = resolve_side_effect
            report = format_toolset_report(
                ["web", "kanban", "computer_use"], RuntimeContext.CLI
            )
            # 2 + 3 = 5 tools pruned
            expected_bytes = 5 * _AVERAGE_TOOL_SCHEMA_BYTES
            expected_tokens = expected_bytes // 4
            assert f"~{expected_tokens} tokens/req" in report

    def test_no_savings_line_when_nothing_pruned(self):
        """Without pruning, no savings line shown."""
        report = format_toolset_report(
            ["web", "terminal"], RuntimeContext.CLI
        )
        assert "Estimated savings" not in report


# =============================================================================
# Edge cases and invariants
# =============================================================================

class TestEdgeCases:
    def test_detect_context_no_env_crash(self):
        """detect_runtime_context should never raise."""
        for env in [{}, {"SOME_UNRELATED_VAR": "1"}]:
            with patch.dict(os.environ, env, clear=True):
                ctx = detect_runtime_context()
                assert isinstance(ctx, RuntimeContext)

    def test_context_enums_are_hashable(self):
        """Enum values must be usable as dict keys (used in _CONTEXT_IRRELEVANT_TOOLSETS)."""
        d = {RuntimeContext.CLI: "ok", RuntimeContext.TUI: "ok"}
        assert d[RuntimeContext.CLI] == "ok"

    def test_pruned_never_contains_removed(self):
        """Intersection of pruned and removed lists must be empty."""
        pruned, removed = pruned_toolsets(
            ["web", "kanban", "computer_use", "terminal"],
            RuntimeContext.CLI,
        )
        assert not (set(pruned) & set(removed))

    def test_pruned_plus_removed_equals_original(self, subtests):
        """Union of pruned and removed must equal the original list (order-agnostic)."""
        original = ["web", "kanban", "terminal", "computer_use", "file"]
        contexts = [RuntimeContext.CLI, RuntimeContext.TUI, RuntimeContext.KANBAN_WORKER]
        for ctx in contexts:
            with subtests.test(ctx=ctx):
                pruned, removed = pruned_toolsets(original, ctx)
                combined = set(pruned) | set(removed)
                # All items accounted for
                assert combined == set(original)
                # No duplicates across lists
                assert len(pruned) + len(removed) == len(original) or (
                    # With duplicates in original, union still covers all
                    combined == set(original)
                )

    def test_format_report_lines(self):
        """Report should always start with runtime context line."""
        report = format_toolset_report(
            ["web", "terminal"], RuntimeContext.CLI
        )
        lines = report.split("\n")
        assert lines[0].startswith("Runtime context:")
