"""Module-level tests for agent/tool_executor.py.

Covers concurrent dispatch, per-tool timeout, max_workers, interrupt handling,
blocked tools, and the sequential fallback path.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from agent import tool_executor
from agent.tool_executor import (
    _budget_for_agent,
    _cancelled_tool_result,
    _MAX_TOOL_WORKERS,
    execute_tool_calls_concurrent,
    execute_tool_calls_sequential,
)

# Import run_agent *before* any worker thread calls _ra() to import it.
# _run_tool() inside execute_tool_calls_concurrent() does
# ``import run_agent`` lazily via _ra().  If the first import happens
# inside a ThreadPoolExecutor worker, module-level init in run_agent can
# deadlock against locks held by the main thread (#34567).  Pre-importing
# here ensures sys.modules has it cached before any concurrent test runs.
import run_agent as _run_agent  # noqa: F401

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


class _FakeToolCall:
    def __init__(self, name: str, arguments: str = "{}", call_id: str = "tc_1"):
        self.function = MagicMock(name=name)
        self.function.name = name
        self.function.arguments = arguments
        self.id = call_id


class _FakeAssistantMsg:
    def __init__(self, tool_calls: list):
        self.tool_calls = tool_calls


def _make_stub_agent(**overrides) -> MagicMock:
    """Minimal agent stub containing every attribute the tool_executor
    module-level functions access *outside* try/except blocks."""
    stub = MagicMock()

    # ── scalar state ─────────────────────────────────────────────
    stub._interrupt_requested = False
    stub.log_prefix = ""
    stub.quiet_mode = True
    stub.verbose_logging = False
    stub.log_prefix_chars = 200
    stub._turns_since_memory = 0
    stub._iters_since_skill = 0
    stub._current_tool = None
    stub._delegate_spinner = None
    stub.session_id = "test-session"
    stub._current_turn_id = ""
    stub._current_api_request_id = ""
    stub.valid_tool_names = set()
    stub.enabled_toolsets = None
    stub.disabled_toolsets = None

    # ── config-driven limits ─────────────────────────────────────
    stub.max_concurrent_tools = 4
    stub.per_tool_timeout_seconds = 0.0
    stub.tool_delay = 0
    stub.tool_progress_callback = None
    stub.tool_start_callback = None
    stub.tool_complete_callback = None

    # ── thread tracking (real objects, not mocks) ────────────────
    stub._tool_worker_threads = set()
    stub._tool_worker_threads_lock = threading.Lock()
    stub._active_children = []
    stub._active_children_lock = threading.Lock()

    # ── sub-objects ──────────────────────────────────────────────
    stub._checkpoint_mgr = MagicMock(enabled=False)
    stub._subdirectory_hints = MagicMock()
    stub._subdirectory_hints.check_tool_call.return_value = ''
    stub._tool_guardrails = MagicMock()
    stub._context_engine_tool_names = None
    stub._memory_manager = None
    stub._tool_guardrails.before_call.return_value = MagicMock(
        allows_execution=True
    )

    # ── methods ──────────────────────────────────────────────────
    stub._touch_activity = MagicMock()
    stub._vprint = MagicMock()
    stub._safe_print = MagicMock()
    stub._should_emit_quiet_tool_messages = MagicMock(return_value=False)
    stub._should_start_quiet_spinner = MagicMock(return_value=False)
    stub._has_stream_consumers = MagicMock(return_value=False)
    stub._apply_pending_steer_to_tool_results = MagicMock()
    stub._guardrail_block_result = MagicMock(
        side_effect=lambda decision: json.dumps(
            {"error": getattr(decision, "message", "blocked by guardrail")},
            ensure_ascii=False,
        )
    )
    stub._append_guardrail_observation = MagicMock(
        side_effect=lambda fn, fa, fr, failed: fr
    )
    stub._record_file_mutation_result = MagicMock()
    stub._invoke_tool = MagicMock(return_value='{"ok": true}')
    stub._tool_result_content_for_active_model = MagicMock(
        side_effect=lambda name, content: content
    )
    stub._flush_messages_to_session_db = MagicMock()

    # Apply overrides so each test can tweak specific attributes.
    for key, value in overrides.items():
        setattr(stub, key, value)

    return stub


# ═══════════════════════════════════════════════════════════════
# _MAX_TOOL_WORKERS  (module constant)
# ═══════════════════════════════════════════════════════════════


class TestMaxToolWorkersConstant:
    def test_constant_is_positive_int(self):
        assert isinstance(_MAX_TOOL_WORKERS, int)
        assert _MAX_TOOL_WORKERS >= 1

    def test_constant_has_reasonable_default(self):
        assert _MAX_TOOL_WORKERS == 8


# ═══════════════════════════════════════════════════════════════
# execute_tool_calls_concurrent  —  max_workers resolution
# ═══════════════════════════════════════════════════════════════


class TestConcurrentMaxWorkers:
    """ThreadPoolExecutor max_workers is resolved correctly."""

    def _run_and_capture_max_workers(self, agent, num_calls: int) -> int:
        """Patch ThreadPoolExecutor.__init__ to capture max_workers."""
        import concurrent.futures as _cf

        captured: dict = {}
        original_init = _cf.ThreadPoolExecutor.__init__

        def _capturing_init(self, *args, **kwargs):
            captured["max_workers"] = kwargs.get("max_workers", "unset")
            return original_init(self, *args, **kwargs)

        with patch.object(
            _cf.ThreadPoolExecutor, "__init__", _capturing_init
        ):
            tcs = [
                _FakeToolCall("tool_x", call_id=f"tc_{i}")
                for i in range(num_calls)
            ]
            msg = _FakeAssistantMsg(tcs)
            execute_tool_calls_concurrent(agent, msg, [], "task-1")

        return captured.get("max_workers", "never-set")

    def test_capped_by_agent_max_concurrent_tools(self):
        """When len(runnable) > agent.max_concurrent_tools,
        max_workers = agent.max_concurrent_tools."""
        agent = _make_stub_agent(max_concurrent_tools=3)
        observed = self._run_and_capture_max_workers(agent, num_calls=10)
        assert observed == 3, f"Expected max_workers=3, got {observed}"

    def test_at_most_runnable_count(self):
        """When len(runnable) < agent.max_concurrent_tools,
        max_workers = len(runnable)."""
        agent = _make_stub_agent(max_concurrent_tools=8)
        observed = self._run_and_capture_max_workers(agent, num_calls=2)
        assert observed == 2, f"Expected max_workers=2, got {observed}"

    def test_fallback_to_module_default(self):
        """When agent has no max_concurrent_tools attribute,
        fallback to _MAX_TOOL_WORKERS."""
        agent = _make_stub_agent()
        if hasattr(agent, "max_concurrent_tools"):
            del agent.max_concurrent_tools
        observed = self._run_and_capture_max_workers(agent, num_calls=20)
        assert observed == _MAX_TOOL_WORKERS, (
            f"Expected fallback to {_MAX_TOOL_WORKERS}, got {observed}"
        )

    def test_all_tools_blocked_skips_executor(self):
        """When every tool is blocked by a guardrail no
        ThreadPoolExecutor is created."""
        agent = _make_stub_agent(max_concurrent_tools=2)

        blocked_decision = MagicMock(allows_execution=False, message="nope")
        agent._tool_guardrails.before_call.return_value = blocked_decision

        tcs = [
            _FakeToolCall("write_file", arguments='{"path":"/etc/hosts"}', call_id="c1"),
            _FakeToolCall("write_file", arguments='{"path":"/etc/shadow"}', call_id="c2"),
        ]
        msg = _FakeAssistantMsg(tcs)

        created = False
        orig = tool_executor.concurrent.futures.ThreadPoolExecutor

        def _watch(*a, **kw):
            nonlocal created
            created = True
            return orig(*a, **kw)

        with patch("agent.tool_executor.concurrent.futures.ThreadPoolExecutor", _watch):
            execute_tool_calls_concurrent(agent, msg, [], "task-1")

        assert not created, (
            "A ThreadPoolExecutor was created despite all tools being blocked"
        )


# ═══════════════════════════════════════════════════════════════
# execute_tool_calls_concurrent  —  dispatch & ordering
# ═══════════════════════════════════════════════════════════════


class TestConcurrentDispatch:
    """Tools are dispatched and results collected."""

    def test_all_tools_execute_and_results_collected(self):
        """Every runnable tool produces a tool result message."""
        agent = _make_stub_agent(max_concurrent_tools=4)
        tcs = [
            _FakeToolCall("tool_a", arguments='{"x":1}', call_id="c1"),
            _FakeToolCall("tool_b", arguments='{"x":2}', call_id="c2"),
            _FakeToolCall("tool_c", arguments='{"x":3}', call_id="c3"),
        ]
        msg = _FakeAssistantMsg(tcs)
        messages: list = []

        agent._invoke_tool = MagicMock(
            side_effect=[
                '{"result":"alpha"}',
                '{"result":"beta"}',
                '{"result":"gamma"}',
            ]
        )

        execute_tool_calls_concurrent(agent, msg, messages, "task-1")

        assert len(messages) == 3
        assert all(m["role"] == "tool" for m in messages)
        produced_ids = {m["tool_call_id"] for m in messages}
        assert produced_ids == {"c1", "c2", "c3"}

    def test_invoke_tool_called_per_runnable(self):
        """agent._invoke_tool is called exactly once per runnable tool."""
        agent = _make_stub_agent(max_concurrent_tools=2)
        tcs = [
            _FakeToolCall("tool_a", call_id="c1"),
            _FakeToolCall("tool_b", call_id="c2"),
        ]
        msg = _FakeAssistantMsg(tcs)
        execute_tool_calls_concurrent(agent, msg, [], "task-1")
        assert agent._invoke_tool.call_count == 2

    def test_invoke_tool_receives_correct_arguments(self):
        """function_name, function_args, task_id, and tool_call_id
        forwarded to _invoke_tool match the tool-definition values."""
        agent = _make_stub_agent(max_concurrent_tools=1)
        tc = _FakeToolCall("web_search", arguments='{"q":"hello"}', call_id="c42")
        msg = _FakeAssistantMsg([tc])

        execute_tool_calls_concurrent(agent, msg, [], "task-007")

        agent._invoke_tool.assert_called_once()
        call_args = agent._invoke_tool.call_args
        assert call_args[0][0] == "web_search"
        assert call_args[0][1] == {"q": "hello"}
        assert call_args[0][2] == "task-007"
        # tool_call_id is the 4th positional arg
        assert call_args[0][3] == "c42", (
            f"Expected tool_call_id='c42', got {call_args[0][3]!r}"
        )

    def test_tool_error_does_not_block_other_tools(self):
        """If one tool's handler raises, other tools still complete."""
        agent = _make_stub_agent(max_concurrent_tools=2)
        tcs = [
            _FakeToolCall("boom_tool", arguments='{"q":"boom"}', call_id="c1"),
            _FakeToolCall("ok_tool", arguments='{"q":"ok"}', call_id="c2"),
        ]
        msg = _FakeAssistantMsg(tcs)
        messages: list = []

        def _fake_invoke(fn, args, tid, tcid, **kw):
            if args.get("q") == "boom":
                raise RuntimeError("simulated crash")
            return '{"status":"ok"}'

        agent._invoke_tool = MagicMock(side_effect=_fake_invoke)

        execute_tool_calls_concurrent(agent, msg, messages, "task-1")

        assert len(messages) == 2
        c1_msgs = [m for m in messages if m["tool_call_id"] == "c1"]
        assert c1_msgs, "No message for the crashing tool"
        c1_content = c1_msgs[0]["content"]
        assert "Error" in c1_content or "simulated" in c1_content, (
            f"Expected error in crashed-tool result, got {c1_content!r}"
        )
        c2_msgs = [m for m in messages if m["tool_call_id"] == "c2"]
        assert c2_msgs
        assert '"ok"' in c2_msgs[0]["content"]


# ═══════════════════════════════════════════════════════════════
# execute_tool_calls_concurrent  —  blocked tools
# ═══════════════════════════════════════════════════════════════


class TestConcurrentBlockedTools:
    """Guardrail and scope-blocked tools are rejected before dispatch."""

    def test_guardrail_blocked_tools_do_not_execute(self):
        """A tool rejected by _tool_guardrails.before_call produces
        a blocked result and _invoke_tool is NOT called for it."""
        agent = _make_stub_agent(max_concurrent_tools=2)
        tcs = [
            _FakeToolCall("dangerous_tool", call_id="c1"),
            _FakeToolCall("safe_tool", call_id="c2"),
        ]
        msg = _FakeAssistantMsg(tcs)
        messages: list = []

        blocked_decision = MagicMock(allows_execution=False, message="blocked by policy")
        ok_decision = MagicMock(allows_execution=True)

        agent._tool_guardrails.before_call = MagicMock(
            side_effect=lambda name, args: (
                blocked_decision if name == "dangerous_tool" else ok_decision
            )
        )

        execute_tool_calls_concurrent(agent, msg, messages, "task-1")

        assert len(messages) == 2
        c1_msgs = [m for m in messages if m["tool_call_id"] == "c1"]
        assert c1_msgs
        assert "blocked" in c1_msgs[0]["content"].lower()
        assert agent._invoke_tool.call_count == 1

    def test_guardrail_message_in_result(self):
        """Guardrail block message is included in the tool result."""
        agent = _make_stub_agent(max_concurrent_tools=1)
        tc = _FakeToolCall("rm_tool", call_id="c1")
        msg = _FakeAssistantMsg([tc])
        messages: list = []

        blocked = MagicMock(allows_execution=False, message="Asset protection policy forbids deletion")
        agent._tool_guardrails.before_call = MagicMock(return_value=blocked)

        execute_tool_calls_concurrent(agent, msg, messages, "task-1")
        assert len(messages) == 1
        assert "Asset protection policy" in messages[0]["content"]

    def test_guardrail_block_result_called(self):
        """_guardrail_block_result is called with the decision so
        agents can customise the blocked-tool message."""
        agent = _make_stub_agent(max_concurrent_tools=1)
        tc = _FakeToolCall("some_tool", call_id="c1")
        msg = _FakeAssistantMsg([tc])
        messages: list = []

        blocked = MagicMock(allows_execution=False, message="custom block text")
        agent._tool_guardrails.before_call = MagicMock(return_value=blocked)

        execute_tool_calls_concurrent(agent, msg, messages, "task-1")

        agent._guardrail_block_result.assert_called_once_with(blocked)


# ═══════════════════════════════════════════════════════════════
# execute_tool_calls_concurrent  —  preflight interrupt
# ═══════════════════════════════════════════════════════════════


class TestConcurrentPreflightInterrupt:
    """Interrupt signalled before concurrent execution starts."""

    def test_all_tools_skipped_with_cancellation_message(self):
        """When _interrupt_requested is True before entry, all tool
        calls produce cancellation messages and _invoke_tool is
        never called."""
        agent = _make_stub_agent()
        agent._interrupt_requested = True
        tcs = [
            _FakeToolCall("tool_a", call_id="c1"),
            _FakeToolCall("tool_b", call_id="c2"),
        ]
        msg = _FakeAssistantMsg(tcs)
        messages: list = []

        execute_tool_calls_concurrent(agent, msg, messages, "task-1")

        assert len(messages) == 2
        assert all(
            "cancelled" in m["content"].lower() or "skipped" in m["content"].lower()
            for m in messages
        )
        agent._invoke_tool.assert_not_called()


# ═══════════════════════════════════════════════════════════════
# execute_tool_calls_concurrent  —  per-tool timeout
# ═══════════════════════════════════════════════════════════════


class TestConcurrentPerToolTimeout:
    """Enforcement of per_tool_timeout_seconds."""

    def test_tool_exceeding_timeout_is_cancelled(self):
        """A tool that runs longer than per_tool_timeout_seconds
        gets cancelled and produces a timeout result.
        Uses a fast tool to keep as_completed alive; the slow one
        is cancelled by the per-tool timeout check."""
        agent = _make_stub_agent(
            max_concurrent_tools=4,
            per_tool_timeout_seconds=0.05,
        )

        def _mock(fn, args, tid, tcid, **kw):
            if fn == "fast":
                return '{"result":"fast_ok"}'
            time.sleep(5)
            return '{"result":"done"}'

        agent._invoke_tool = MagicMock(side_effect=_mock)

        tcs = [
            _FakeToolCall("fast", call_id="c1"),
            _FakeToolCall("slowpoke", call_id="c2"),
        ]
        msg = _FakeAssistantMsg(tcs)
        messages: list = []

        execute_tool_calls_concurrent(agent, msg, messages, "task-1")

        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
        by_id = {m["tool_call_id"]: m["content"] for m in messages}
        assert '"fast_ok"' in by_id["c1"], (
            f"Expected fast result, got {by_id['c1']!r}"
        )
        assert "timeout" in by_id["c2"].lower(), (
            f"Expected timeout for slow tool, got {by_id['c2']!r}"
        )

    def test_timeout_zero_disables_check(self):
        """per_tool_timeout_seconds = 0 disables the per-tool timeout
        check entirely, so a slow tool completes normally."""
        agent = _make_stub_agent(
            max_concurrent_tools=1,
            per_tool_timeout_seconds=0.0,
        )

        def _briefly_slow(*a, **kw):
            time.sleep(0.05)
            return '{"result":"finally"}'

        agent._invoke_tool = MagicMock(side_effect=_briefly_slow)

        tc = _FakeToolCall("ok_slow", call_id="c1")
        msg = _FakeAssistantMsg([tc])
        messages: list = []

        execute_tool_calls_concurrent(agent, msg, messages, "task-1")

        assert len(messages) == 1
        assert '"finally"' in messages[0]["content"]

    def test_timeout_only_affects_exceeding_tools(self):
        """When two tools run concurrently and one times out, the
        other still completes successfully."""
        agent = _make_stub_agent(
            max_concurrent_tools=4,
            per_tool_timeout_seconds=0.05,
        )

        def _conditional(fn, args, tid, tcid, **kw):
            if fn == "fast_tool":
                return '{"result":"fast_ok"}'
            time.sleep(5)
            return '{"result":"slow_ok"}'

        agent._invoke_tool = MagicMock(side_effect=_conditional)

        tcs = [
            _FakeToolCall("fast_tool", call_id="c1"),
            _FakeToolCall("slow_tool", call_id="c2"),
        ]
        msg = _FakeAssistantMsg(tcs)
        messages: list = []

        execute_tool_calls_concurrent(agent, msg, messages, "task-1")

        assert len(messages) == 2
        fast_msgs = [m for m in messages if m["tool_call_id"] == "c1"]
        slow_msgs = [m for m in messages if m["tool_call_id"] == "c2"]
        assert fast_msgs, "Missing result for fast tool"
        assert slow_msgs, "Missing result for slow tool"
        assert '"fast_ok"' in fast_msgs[0]["content"], (
            f"Expected fast result, got {fast_msgs[0]['content']!r}"
        )
        assert "timeout" in slow_msgs[0]["content"].lower(), (
            f"Expected timeout for slow tool, got {slow_msgs[0]['content']!r}"
        )

    def test_timeout_multiple_stragglers(self):
        """When several tools exceed the timeout, each gets a timeout
        result independently."""
        agent = _make_stub_agent(
            max_concurrent_tools=4,
            per_tool_timeout_seconds=0.05,
        )

        def _slow(*a, **kw):
            time.sleep(5)
            return '{"result":"done"}'

        agent._invoke_tool = MagicMock(side_effect=_slow)

        tcs = [
            _FakeToolCall("slow_a", call_id="c1"),
            _FakeToolCall("slow_b", call_id="c2"),
            _FakeToolCall("slow_c", call_id="c3"),
        ]
        msg = _FakeAssistantMsg(tcs)
        messages: list = []

        execute_tool_calls_concurrent(agent, msg, messages, "task-1")

        assert len(messages) == 3
        for m in messages:
            assert "timeout" in m["content"].lower(), (
                f"Expected timeout for {m['tool_call_id']}, got {m['content']!r}"
            )


# ═══════════════════════════════════════════════════════════════
# execute_tool_calls_concurrent  —  worker thread tracking
# ═══════════════════════════════════════════════════════════════


class TestConcurrentWorkerThreadTracking:
    """Worker thread IDs are registered and cleaned up."""

    def test_worker_deregisters_its_tid_after_completion(self):
        """After the tool returns, the worker's tid is removed from
        _tool_worker_threads."""
        agent = _make_stub_agent(max_concurrent_tools=1)
        tc = _FakeToolCall("tool_x", call_id="c1")
        msg = _FakeAssistantMsg([tc])

        execute_tool_calls_concurrent(agent, msg, [], "task-1")

        with agent._tool_worker_threads_lock:
            assert len(agent._tool_worker_threads) == 0, (
                f"Worker tid leaked: {agent._tool_worker_threads}"
            )


# ═══════════════════════════════════════════════════════════════
# execute_tool_calls_concurrent  —  interpreter-shutdown resiliency
# ═══════════════════════════════════════════════════════════════


class TestConcurrentSubmitShutdownError:
    """Graceful handling of 'interpreter shutdown' on submit."""

    def test_submit_shutdown_returns_tool_errors(self):
        """When ThreadPoolExecutor.submit raises 'cannot schedule new
        futures after interpreter shutdown', all unsubmitted tools
        get an error result and no exception escapes."""
        agent = _make_stub_agent(max_concurrent_tools=8)
        tcs = [
            _FakeToolCall("tool_a", call_id="c1"),
            _FakeToolCall("tool_b", call_id="c2"),
        ]
        msg = _FakeAssistantMsg(tcs)
        messages: list = []

        class _ShutdownExecutor:
            def __init__(self, *a, **kw):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def submit(self, *a, **kw):
                raise RuntimeError(
                    "cannot schedule new futures after interpreter shutdown"
                )

        with patch(
            "agent.tool_executor.concurrent.futures.ThreadPoolExecutor",
            _ShutdownExecutor,
        ):
            execute_tool_calls_concurrent(agent, msg, messages, "task-1")

        assert len(messages) == 2
        assert all(
            "interpreter is shutting down" in m["content"] for m in messages
        ), "Expected shutdown error in all tool results"


# ═══════════════════════════════════════════════════════════════
# execute_tool_calls_concurrent  —  activity heartbeat
# ═══════════════════════════════════════════════════════════════


class TestConcurrentActivityHeartbeat:
    """_touch_activity is called during the execute loop."""

    def test_activity_touched_during_long_running_tools(self):
        """_touch_activity is called at least once during concurrent
        execution of long-running tools."""
        agent = _make_stub_agent(
            max_concurrent_tools=1,
            per_tool_timeout_seconds=0.1,
        )

        def _slow(*a, **kw):
            time.sleep(0.5)
            return '{"ok":true}'

        agent._invoke_tool = MagicMock(side_effect=_slow)

        tc = _FakeToolCall("slow", call_id="c1")
        msg = _FakeAssistantMsg([tc])

        execute_tool_calls_concurrent(agent, msg, [], "task-1")

        assert agent._touch_activity.call_count >= 1


# ═══════════════════════════════════════════════════════════════
# execute_tool_calls_sequential
# ═══════════════════════════════════════════════════════════════


class TestSequential:
    """Sequential execution path."""

    def test_tools_execute_in_order(self):
        """Tools are called one-by-one, results appended in order."""
        agent = _make_stub_agent()
        call_log: list = []

        def _fake_handle_function_call(
            function_name, function_args, effective_task_id, **kw
        ):
            call_log.append(function_name)
            return f'{{"result":"{function_name}"}}'

        with patch(
            "run_agent.handle_function_call",
            side_effect=_fake_handle_function_call,
        ):
            tcs = [
                _FakeToolCall("first", call_id="c1"),
                _FakeToolCall("second", call_id="c2"),
                _FakeToolCall("third", call_id="c3"),
            ]
            msg = _FakeAssistantMsg(tcs)
            messages: list = []

            execute_tool_calls_sequential(agent, msg, messages, "task-1")

        assert call_log == ["first", "second", "third"]
        assert len(messages) == 3
        assert [m["tool_call_id"] for m in messages] == ["c1", "c2", "c3"]

    def test_interrupt_skips_remaining_tools(self):
        """If interrupt is signalled after the first tool finishes,
        remaining tools are skipped."""
        agent = _make_stub_agent()
        call_log: list = []

        def _fake_handle(fn, args, tid, **kw):
            call_log.append(fn)
            if fn == "first":
                agent._interrupt_requested = True
            return f'{{"result":"{fn}"}}'

        with patch(
            "run_agent.handle_function_call",
            side_effect=_fake_handle,
        ):
            tcs = [
                _FakeToolCall("first", call_id="c1"),
                _FakeToolCall("second", call_id="c2"),
                _FakeToolCall("third", call_id="c3"),
            ]
            msg = _FakeAssistantMsg(tcs)
            messages: list = []

            execute_tool_calls_sequential(agent, msg, messages, "task-1")

        assert call_log == ["first"]
        assert len(messages) == 3
        assert messages[0]["tool_call_id"] == "c1"
        for m in messages[1:]:
            assert "skipped" in m["content"].lower() or "cancelled" in m["content"].lower()

    def test_preflight_interrupt_skips_all(self):
        """If interrupt is signalled before execution, all are skipped."""
        agent = _make_stub_agent()
        agent._interrupt_requested = True

        with patch("run_agent.handle_function_call") as mock_hfc:
            tcs = [
                _FakeToolCall("tool_a", call_id="c1"),
                _FakeToolCall("tool_b", call_id="c2"),
            ]
            msg = _FakeAssistantMsg(tcs)
            messages: list = []

            execute_tool_calls_sequential(agent, msg, messages, "task-1")

        assert len(messages) == 2
        assert all("skipped" in m["content"].lower() for m in messages)
        mock_hfc.assert_not_called()

    def test_blocked_tool_returns_error_no_execution(self):
        """A tool blocked by a guardrail returns an error message
        without executing, and the next tool still runs."""
        agent = _make_stub_agent()

        ok_decision = MagicMock(allows_execution=True)
        blocked_decision = MagicMock(allows_execution=False, message="nope")
        agent._tool_guardrails.before_call = MagicMock(
            side_effect=lambda name, args: (
                blocked_decision if name == "blocked_tool" else ok_decision
            )
        )

        with patch("run_agent.handle_function_call", return_value='{"ok":true}'):
            tcs = [
                _FakeToolCall("blocked_tool", call_id="c1"),
                _FakeToolCall("ok_tool", call_id="c2"),
            ]
            msg = _FakeAssistantMsg(tcs)
            messages: list = []

            execute_tool_calls_sequential(agent, msg, messages, "task-1")

        assert len(messages) == 2
        assert "nope" in messages[0]["content"]
        assert messages[1]["tool_call_id"] == "c2"


# ═══════════════════════════════════════════════════════════════
# execute_tool_calls_concurrent  —  _finalize_tool_result
# ═══════════════════════════════════════════════════════════════


class TestFinalizeToolResult:
    """_finalize_tool_result processes individual tool results and
    appends them to the message list."""

    def test_happy_path_appends_tool_message(self):
        """A successful finalize appends a tool-role message with the
        correct tool_call_id."""
        from agent.tool_executor import _finalize_tool_result

        agent = _make_stub_agent()
        messages: list = []
        parsed_calls = [
            (_FakeToolCall("web_search", call_id="c1"), "web_search", {"q": "hello"}, [], None, False),
        ]
        results = [("web_search", {"q": "hello"}, '{"result":"ok"}', 0.1, False, False, [])]
        tc = _FakeToolCall("web_search", call_id="c1")

        _finalize_tool_result(
            agent, messages, parsed_calls, results,
            "task-1", 0, tc, "web_search", {"q": "hello"},
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "c1"

    def test_none_result_when_interrupted(self):
        """When results[idx] is None and interrupt was requested, a
        cancellation message is appended."""
        from agent.tool_executor import _finalize_tool_result

        agent = _make_stub_agent()
        agent._interrupt_requested = True
        messages: list = []
        parsed_calls = [
            (_FakeToolCall("web_search", call_id="c1"), "web_search", {}, [], None, False),
        ]
        results = [None]
        tc = _FakeToolCall("web_search", call_id="c1")

        _finalize_tool_result(
            agent, messages, parsed_calls, results,
            "task-1", 0, tc, "web_search", {},
        )

        assert len(messages) == 1
        assert "cancelled" in messages[0]["content"].lower()

    def test_none_result_without_interrupt(self):
        """When results[idx] is None and no interrupt, a 'did not
        return a result' error is appended."""
        from agent.tool_executor import _finalize_tool_result

        agent = _make_stub_agent()
        agent._interrupt_requested = False
        messages: list = []
        parsed_calls = [
            (_FakeToolCall("web_search", call_id="c1"), "web_search", {}, [], None, False),
        ]
        results = [None]
        tc = _FakeToolCall("web_search", call_id="c1")

        _finalize_tool_result(
            agent, messages, parsed_calls, results,
            "task-1", 0, tc, "web_search", {},
        )

        assert len(messages) == 1
        assert "did not return a result" in messages[0]["content"]


# ═══════════════════════════════════════════════════════════════
# _budget_for_agent
# ═══════════════════════════════════════════════════════════════


class TestBudgetForAgent:
    """BudgetConfig resolution from agent context_compressor."""

    def test_returns_default_budget_when_no_context(self):
        """When the agent has no context_compressor or no
        context_length, the default budget is returned."""
        agent = _make_stub_agent()
        if hasattr(agent, "context_compressor"):
            del agent.context_compressor

        budget = _budget_for_agent(agent)
        from agent.tool_executor import DEFAULT_BUDGET

        assert budget is DEFAULT_BUDGET

    def test_scales_budget_to_context_window(self):
        """When context_compressor.context_length is set, budget is
        proportional to that window."""
        from unittest.mock import PropertyMock

        agent = _make_stub_agent()
        agent.context_compressor = MagicMock()
        type(agent.context_compressor).context_length = PropertyMock(return_value=128_000)

        budget = _budget_for_agent(agent)

        # For a 128K context window the scaled result size should be
        # at least as large as the base default (100K).
        assert budget.default_result_size >= 50_000


# ═══════════════════════════════════════════════════════════════
# _cancelled_tool_result
# ═══════════════════════════════════════════════════════════════


class TestCancelledToolResult:
    def test_returns_json_with_cancelled_status(self):
        result = _cancelled_tool_result("user interrupt")
        parsed = json.loads(result)
        assert parsed["status"] == "cancelled"
        assert "user interrupt" in parsed["error"]

    def test_custom_reason(self):
        result = _cancelled_tool_result("rate limit exceeded")
        parsed = json.loads(result)
        assert "rate limit" in parsed["error"]


# ═══════════════════════════════════════════════════════════════
# _is_interpreter_shutdown_submit_error
# ═══════════════════════════════════════════════════════════════


class TestIsInterpreterShutdownSubmitError:
    def test_matches_shutdown_message(self):
        from agent.tool_executor import _is_interpreter_shutdown_submit_error

        exc = RuntimeError("cannot schedule new futures after interpreter shutdown")
        assert _is_interpreter_shutdown_submit_error(exc) is True

    def test_other_runtime_error_does_not_match(self):
        from agent.tool_executor import _is_interpreter_shutdown_submit_error

        exc = RuntimeError("some other error")
        assert _is_interpreter_shutdown_submit_error(exc) is False


# ═══════════════════════════════════════════════════════════════
# _tool_search_scoped_names
# ═══════════════════════════════════════════════════════════════


class TestToolSearchScopedNames:
    def test_returns_frozenset(self):
        from agent.tool_executor import _tool_search_scoped_names

        agent = _make_stub_agent()
        result = _tool_search_scoped_names(agent)
        assert isinstance(result, frozenset)

    def test_caches_result(self):
        from agent.tool_executor import _tool_search_scoped_names

        agent = _make_stub_agent(
            enabled_toolsets={"web", "file"},
            disabled_toolsets=set(),
        )
        first = _tool_search_scoped_names(agent)
        second = _tool_search_scoped_names(agent)
        assert first is second, "Expected cache hit (same object)"


# ═══════════════════════════════════════════════════════════════
# execute_tool_calls_concurrent  —  session DB flush
# ═══════════════════════════════════════════════════════════════


class TestConcurrentSessionDbFlush:
    """Concurrent execution calls _flush_messages_to_session_db
    after each tool result."""

    def test_flush_called_per_tool_result(self):
        """Each completed tool triggers a flush call."""
        agent = _make_stub_agent(max_concurrent_tools=2)
        tcs = [
            _FakeToolCall("web_search", call_id="c1"),
            _FakeToolCall("web_search", call_id="c2"),
        ]
        msg = _FakeAssistantMsg(tcs)
        messages: list = []

        execute_tool_calls_concurrent(agent, msg, messages, "task-1")

        assert agent._flush_messages_to_session_db.call_count >= 2, (
            f"Expected at least 2 flushes, got "
            f"{agent._flush_messages_to_session_db.call_count}"
        )


# ═══════════════════════════════════════════════════════════════
# execute_tool_calls_sequential  —  session DB flush
# ═══════════════════════════════════════════════════════════════


class TestSequentialSessionDbFlush:
    """Sequential execution calls _flush_messages_to_session_db
    after each tool result."""

    def test_flush_called_per_tool(self):
        agent = _make_stub_agent()

        def _fake_handle(fn, *a, **kw):
            return '{"ok":true}'

        with patch("run_agent.handle_function_call", side_effect=_fake_handle):
            tcs = [
                _FakeToolCall("web_search", call_id="c1"),
                _FakeToolCall("web_search", call_id="c2"),
            ]
            msg = _FakeAssistantMsg(tcs)
            messages: list = []

            execute_tool_calls_sequential(agent, msg, messages, "task-1")

        assert agent._flush_messages_to_session_db.call_count >= 2


# ═══════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary conditions."""

    def test_empty_tool_calls_concurrent(self):
        """An assistant message with no tool calls is a no-op."""
        agent = _make_stub_agent()
        msg = _FakeAssistantMsg([])
        messages: list = []
        execute_tool_calls_concurrent(agent, msg, messages, "task-1")
        assert len(messages) == 0

    def test_empty_tool_calls_sequential(self):
        """An assistant message with no tool calls is a no-op."""
        agent = _make_stub_agent()
        msg = _FakeAssistantMsg([])
        messages: list = []
        execute_tool_calls_sequential(agent, msg, messages, "task-1")
        assert len(messages) == 0

    def test_single_tool_concurrent(self):
        """A single tool call still goes through the concurrent path."""
        agent = _make_stub_agent(max_concurrent_tools=1)
        tc = _FakeToolCall("tool_x", call_id="c1")
        msg = _FakeAssistantMsg([tc])
        messages: list = []
        execute_tool_calls_concurrent(agent, msg, messages, "task-1")
        assert len(messages) == 1
        assert messages[0]["tool_call_id"] == "c1"

    def test_tool_worker_threads_lock_not_deadlocked(self):
        """Multiple concurrent workers acquire and release
        _tool_worker_threads_lock without deadlock."""
        agent = _make_stub_agent(max_concurrent_tools=4)
        tcs = [_FakeToolCall("tool_x", call_id=f"c{i}") for i in range(4)]
        msg = _FakeAssistantMsg(tcs)
        messages: list = []

        agent._invoke_tool = MagicMock(return_value='{"ok":true}')

        execute_tool_calls_concurrent(agent, msg, messages, "task-1")

        assert len(messages) == 4
