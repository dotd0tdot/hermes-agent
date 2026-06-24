"""Tests for agent/turn_pipeline.py — pipeline abstraction for conversation loop."""

from __future__ import annotations

import json
import logging
import time
from unittest.mock import MagicMock, call, patch

import pytest
from agent.turn_pipeline import (
    LogTurnEndStage,
    LogTurnStartStage,
    MetricsStage,
    PipelineContext,
    ToolUseReportStage,
    TurnPipeline,
    TurnStage,
    _assemble_result,
    default_stages,
    pipeline_run,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fake_agent():
    """Return a minimal agent-like object with the attributes the pipeline expects."""
    return MagicMock(
        provider="test-provider",
        model="test-model",
        tools=["tool_a", "tool_b"],
    )


@pytest.fixture
def ctx(fake_agent):
    """Return a bare PipelineContext with sensible defaults."""
    return PipelineContext(
        agent=fake_agent,
        user_message="hello world",
        original_user_message="hello world",
        messages=[{"role": "user", "content": "hello world"}],
        active_system_prompt="You are a test bot.",
        effective_task_id="task-001",
        turn_id="turn-1",
        current_turn_user_idx=0,
        conversation_history=None,
    )


@pytest.fixture
def empty_pipeline():
    return TurnPipeline(stages=[])


class _APICallBoundary(TurnStage):
    """Helper stage that acts as the pre/post boundary marker."""
    name = "api_call"
    def run(self, ctx):
        return ctx


# =============================================================================
# PipelineContext
# =============================================================================

class TestPipelineContext:
    def test_default_field_values(self, fake_agent):
        """Runtime counters and metadata have sensible defaults."""
        c = PipelineContext(
            agent=fake_agent,
            user_message="hi",
            original_user_message="hi",
            messages=[],
            active_system_prompt=None,
            effective_task_id="t1",
            turn_id="t1",
            current_turn_user_idx=0,
            conversation_history=None,
        )
        assert c.api_call_count == 0
        assert c.final_response is None
        assert c.interrupted is False
        assert c.failed is False
        assert c.compression_attempts == 0
        assert c.exit_reason == "unknown"
        assert c.stage_timing == {}
        assert c.extras == {}

    def test_extras_stores_arbitrary_data(self, ctx):
        ctx.extras["stream_callback"] = lambda x: x
        ctx.extras["score"] = 42
        assert ctx.extras["stream_callback"] is not None
        assert ctx.extras["score"] == 42


# =============================================================================
# TurnStage base class
# =============================================================================

class TestTurnStage:
    def test_base_run_raises_not_implemented(self):
        stage = TurnStage()
        stage.name = "test"
        with pytest.raises(NotImplementedError):
            stage.run(MagicMock())

    def test_repr(self):
        stage = TurnStage()
        stage.name = "my_stage"
        assert repr(stage) == "<TurnStage 'my_stage'>"

    def test_default_name(self):
        stage = TurnStage()
        assert stage.name == "unnamed"


# =============================================================================
# TurnPipeline orchestration
# =============================================================================

class TestTurnPipeline:
    def test_empty_pipeline_stages(self, empty_pipeline):
        assert empty_pipeline.stages() == []

    def test_add_and_stages(self, empty_pipeline):
        s1 = MagicMock(spec=TurnStage, name="stage_one")
        s1.name = "stage_one"
        s2 = MagicMock(spec=TurnStage, name="stage_two")
        s2.name = "stage_two"

        empty_pipeline.add(s1)
        empty_pipeline.add(s2)
        assert empty_pipeline.stages() == [s1, s2]
        assert len(empty_pipeline.stages()) == 2

    def test_add_returns_stage(self, empty_pipeline):
        s = MagicMock(spec=TurnStage, name="s")
        s.name = "s"
        returned = empty_pipeline.add(s)
        assert returned is s

    def test_insert_before(self, empty_pipeline):
        a = MagicMock(spec=TurnStage, name="a"); a.name = "a"
        b = MagicMock(spec=TurnStage, name="b"); b.name = "b"
        c = MagicMock(spec=TurnStage, name="c"); c.name = "c"
        empty_pipeline.add(a); empty_pipeline.add(b)
        assert empty_pipeline.insert_before("b", c) is True
        assert empty_pipeline.stages() == [a, c, b]

    def test_insert_before_not_found(self, empty_pipeline, caplog):
        caplog.set_level(logging.WARNING)
        s = MagicMock(spec=TurnStage, name="s"); s.name = "s"
        assert empty_pipeline.insert_before("nonexistent", s) is False
        assert "insert_before: stage 'nonexistent' not found" in caplog.text

    def test_insert_after(self, empty_pipeline):
        a = MagicMock(spec=TurnStage, name="a"); a.name = "a"
        b = MagicMock(spec=TurnStage, name="b"); b.name = "b"
        c = MagicMock(spec=TurnStage, name="c"); c.name = "c"
        empty_pipeline.add(a); empty_pipeline.add(b)
        assert empty_pipeline.insert_after("a", c) is True
        assert empty_pipeline.stages() == [a, c, b]

    def test_insert_after_not_found(self, empty_pipeline, caplog):
        caplog.set_level(logging.WARNING)
        s = MagicMock(spec=TurnStage, name="s"); s.name = "s"
        assert empty_pipeline.insert_after("nonexistent", s) is False

    def test_replace(self, empty_pipeline):
        a = MagicMock(spec=TurnStage, name="a"); a.name = "a"
        b = MagicMock(spec=TurnStage, name="b"); b.name = "b"
        empty_pipeline.add(a)
        assert empty_pipeline.replace("a", b) is True
        assert empty_pipeline.stages() == [b]

    def test_replace_not_found(self, empty_pipeline, caplog):
        caplog.set_level(logging.WARNING)
        s = MagicMock(spec=TurnStage, name="s"); s.name = "s"
        assert empty_pipeline.replace("nonexistent", s) is False

    def test_remove(self, empty_pipeline):
        a = MagicMock(spec=TurnStage, name="a"); a.name = "a"
        b = MagicMock(spec=TurnStage, name="b"); b.name = "b"
        empty_pipeline.add(a); empty_pipeline.add(b)
        assert empty_pipeline.remove("a") is True
        assert empty_pipeline.stages() == [b]

    def test_remove_not_found(self, empty_pipeline, caplog):
        caplog.set_level(logging.WARNING)
        assert empty_pipeline.remove("nonexistent") is False

    def test_stages_returns_copy(self, empty_pipeline):
        s = MagicMock(spec=TurnStage, name="s"); s.name = "s"
        empty_pipeline.add(s)
        stages_copy = empty_pipeline.stages()
        stages_copy.clear()
        # Original should be unaffected
        assert len(empty_pipeline.stages()) == 1


# =============================================================================
# TurnPipeline.run — pre-stages
# =============================================================================

class TestTurnPipelineRun:
    def test_pre_stages_run_in_order(self, fake_agent):
        """Pre-stages are executed in insertion order before run_conversation."""
        order = []

        class TrackingStage(TurnStage):
            def __init__(self, name):
                self.name = name
            def run(self, ctx):
                order.append(self.name)
                return ctx

        pipe = TurnPipeline(stages=[
            TrackingStage("first"),
            TrackingStage("second"),
            _APICallBoundary(),  # boundary between pre and post stages
        ])

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "ok", "messages": [], "exit_reason": "completed"}
            pipe.run(fake_agent, "hi")

        # Verify pre-stage order (first two entries, before post-stages re-run)
        assert order[:2] == ["first", "second"]
        assert len(order) == 4  # Run twice: pre-stages + post-stages

    def test_pre_stage_sets_failed_on_exception(self, fake_agent):
        class FailingStage(TurnStage):
            name = "pre_api"
            def run(self, ctx):
                raise ValueError("boom")

        pipe = TurnPipeline(stages=[FailingStage()])
        result = pipe.run(fake_agent, "hi")

        assert result["failed"] is True
        assert "stage_error:pre_api" in result["exit_reason"]

    def test_interrupt_during_pre_stage_aborts(self, fake_agent):
        class InterruptStage(TurnStage):
            name = "interrupter"
            def run(self, ctx):
                ctx.interrupted = True
                return ctx

        pipe = TurnPipeline(stages=[InterruptStage()])

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            result = pipe.run(fake_agent, "hi")
            mock_loop.assert_not_called()

        assert result["interrupted"] is True

    def test_only_pre_stages_run_before_api_call(self, fake_agent):
        """Stages with name 'api_call' or 'tool_dispatch' are post-stages."""
        order = []

        class T(TurnStage):
            def __init__(self, name):
                self.name = name
            def run(self, ctx):
                order.append(self.name)
                return ctx

        pipe = TurnPipeline(stages=[
            T("log_turn_start"),
            T("api_call"),       # boundary: pre-stage loop stops here
            T("tool_dispatch"),
            T("log_turn_end"),
        ])

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "ok", "messages": [], "exit_reason": "completed"}
            pipe.run(fake_agent, "hi")

        # api_call and tool_dispatch should NOT appear in pre-stages
        assert "log_turn_start" in order
        assert "api_call" not in order
        assert "tool_dispatch" not in order

    def test_stage_timing_recorded(self, fake_agent):
        """Each stage's wall-clock time is recorded in ctx.stage_timing."""
        pipe = TurnPipeline(stages=default_stages())

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "ok", "messages": [], "exit_reason": "completed"}
            result = pipe.run(fake_agent, "hi")

        timing = result.get("stage_timing", {})
        # Pre-stages appear
        assert "log_turn_start" in timing
        # Main turn appears
        assert "_run_conversation" in timing
        # Post-stages appear
        assert "log_turn_end" in timing
        assert "metrics" in timing
        assert "tool_use_report" in timing
        for v in timing.values():
            assert v >= 0


# =============================================================================
# TurnPipeline.run — main turn delegation
# =============================================================================

class TestTurnPipelineRunMain:
    def test_delegates_to_run_conversation(self, fake_agent):
        pipe = TurnPipeline(stages=default_stages())

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "hello back", "messages": [], "exit_reason": "completed"}
            result = pipe.run(fake_agent, "hi")

        mock_loop.assert_called_once()
        call_args = mock_loop.call_args[0]
        assert call_args[0] is fake_agent

    def test_passes_kwargs_to_run_conversation(self, fake_agent):
        pipe = TurnPipeline(stages=default_stages())

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "ok", "messages": [], "exit_reason": "completed"}
            pipe.run(
                fake_agent,
                user_message="test-msg",
                system_message="test-system",
                conversation_history=[{"role": "user", "content": "prev"}],
                task_id="task-99",
                stream_callback=lambda m: None,
                persist_user_message="persisted",
                persist_user_timestamp=12345.0,
            )

        mock_loop.assert_called_once_with(
            fake_agent,
            user_message="test-msg",
            system_message="test-system",
            conversation_history=[{"role": "user", "content": "prev"}],
            task_id="task-99",
            stream_callback=mock_loop.call_args[1]["stream_callback"],
            persist_user_message="persisted",
            persist_user_timestamp=12345.0,
        )

    def test_merges_loop_result_into_ctx(self, fake_agent):
        pipe = TurnPipeline(stages=default_stages())

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {
                "response": "final answer",
                "messages": [{"role": "assistant", "content": "final answer"}],
                "interrupted": True,
                "failed": False,
                "exit_reason": "max_iterations",
                "turn_id": "turn-42",
            }
            result = pipe.run(fake_agent, "hi")

        assert result["response"] == "final answer"
        assert result["interrupted"] is True
        assert result["failed"] is False
        assert result["exit_reason"] == "max_iterations"
        assert result["turn_id"] == "turn-42"

    def test_run_conversation_exception_sets_failed(self, fake_agent):
        pipe = TurnPipeline(stages=[])

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.side_effect = RuntimeError("loop crash")
            result = pipe.run(fake_agent, "hi")

        assert result["failed"] is True
        assert "loop_error" in result["exit_reason"]

    def test_post_stages_run_after_loop(self, fake_agent):
        """Post-stages execute after run_conversation returns successfully."""
        post_order = []

        class PostStage(TurnStage):
            def __init__(self, name):
                self.name = name
            def run(self, ctx):
                post_order.append(self.name)
                return ctx

        pipe = TurnPipeline(stages=[
            _APICallBoundary(),
            PostStage("post_one"),
            PostStage("post_two"),
        ])

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "ok", "messages": [], "exit_reason": "completed"}
            pipe.run(fake_agent, "hi")

        assert post_order == ["post_one", "post_two"]

    def test_post_stage_exception_does_not_mark_failed(self, fake_agent):
        class FailingPostStage(TurnStage):
            name = "post_broken"
            def run(self, ctx):
                raise ValueError("post error")

        pipe = TurnPipeline(stages=[_APICallBoundary(), FailingPostStage()])

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "ok", "messages": [], "exit_reason": "completed"}
            result = pipe.run(fake_agent, "hi")

        # Post-stage failures are non-critical — result should still show success
        assert result["failed"] is False
        assert result["response"] == "ok"

    def test_setup_and_pre_api_skipped_in_post(self, fake_agent):
        """Stages named setup/pre_api are NOT run as post-stages."""
        all_run_order = []

        class T(TurnStage):
            def __init__(self, name):
                self.name = name
            def run(self, ctx):
                all_run_order.append(self.name)
                return ctx

        pipe = TurnPipeline(stages=[
            T("setup"),
            T("pre_api"),
            T("api_call"),
            T("tool_dispatch"),
            T("log_turn_end"),
        ])

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "ok", "messages": [], "exit_reason": "completed"}
            pipe.run(fake_agent, "hi")

        assert "setup" not in all_run_order or all_run_order.count("setup") == 1
        assert "pre_api" not in all_run_order or all_run_order.count("pre_api") == 1
        assert "api_call" not in all_run_order
        assert "tool_dispatch" not in all_run_order
        assert "log_turn_end" in all_run_order


# =============================================================================
# _assemble_result
# =============================================================================

class TestAssembleResult:
    def test_assembles_dict_from_ctx(self, ctx):
        ctx.final_response = "done"
        ctx.interrupted = True
        ctx.failed = False
        ctx.exit_reason = "completed"
        ctx.turn_id = "turn-5"
        ctx.stage_timing["pre"] = 0.1

        result = _assemble_result(ctx)
        assert result["response"] == "done"
        assert result["interrupted"] is True
        assert result["failed"] is False
        assert result["turn_id"] == "turn-5"
        assert result["exit_reason"] == "completed"
        assert result["stage_timing"] == {"pre": 0.1}

    def test_stage_timing_is_copy(self, ctx):
        ctx.stage_timing["a"] = 1.0
        result = _assemble_result(ctx)
        result["stage_timing"]["a"] = 99
        assert ctx.stage_timing["a"] == 1.0


# =============================================================================
# LogTurnStartStage
# =============================================================================

class TestLogTurnStartStage:
    def test_logs_message_preview(self, fake_agent, ctx):
        log_records = []
        stage = LogTurnStartStage(log_fn=lambda *a, **k: log_records.append((a, k)))
        stage.run(ctx)
        assert len(log_records) >= 1
        args, _ = log_records[0]
        # Format: "Turn start | msg='...' | model=.../... | tools=..."
        assert "Turn start" in args[0] if isinstance(args[0], str) else True
        # Verify the logger was called
        assert any("Turn start" in str(r) for r in args)

    def test_truncates_long_messages(self, fake_agent):
        long_msg = "x" * 200
        ctx = PipelineContext(
            agent=fake_agent,
            user_message=long_msg,
            original_user_message=long_msg,
            messages=[],
            active_system_prompt=None,
            effective_task_id="t1",
            turn_id="t1",
            current_turn_user_idx=0,
            conversation_history=None,
        )
        log_records = []
        stage = LogTurnStartStage(log_fn=lambda *a, **k: log_records.append((a, k)))
        stage.run(ctx)
        assert len(log_records) >= 1

    def test_handles_missing_agent_attrs(self, ctx):
        ctx.agent = MagicMock(spec=[])  # No provider/model/tools attributes
        log_records = []
        stage = LogTurnStartStage(log_fn=lambda *a, **k: log_records.append((a, k)))
        # Should not raise
        result = stage.run(ctx)
        assert result is ctx


# =============================================================================
# LogTurnEndStage
# =============================================================================

class TestLogTurnEndStage:
    def test_logs_timing_and_exit_reason(self, ctx):
        ctx.exit_reason = "completed"
        ctx.api_call_count = 3
        ctx.stage_timing["log_turn_start"] = 0.01
        ctx.stage_timing["_run_conversation"] = 0.5

        log_records = []
        stage = LogTurnEndStage(log_fn=lambda *a, **k: log_records.append((a, k)))
        stage.run(ctx)

        assert len(log_records) >= 1

    def test_handles_empty_timing(self, ctx):
        ctx.stage_timing = {}
        log_records = []
        stage = LogTurnEndStage(log_fn=lambda *a, **k: log_records.append((a, k)))
        result = stage.run(ctx)
        assert result is ctx


# =============================================================================
# MetricsStage
# =============================================================================

class TestMetricsStage:
    def test_collects_metrics(self, ctx):
        ctx.exit_reason = "completed"
        ctx.api_call_count = 5
        ctx.interrupted = False
        ctx.failed = False
        ctx.compression_attempts = 1
        ctx.stage_timing["a"] = 0.1
        ctx.stage_timing["b"] = 0.2

        with patch("time.time", return_value=1000.0):
            stage = MetricsStage()
            stage.run(ctx)

        metrics = ctx.extras["metrics"]
        assert metrics["exit_reason"] == "completed"
        assert metrics["api_call_count"] == 5
        assert metrics["interrupted"] is False
        assert metrics["failed"] is False
        assert metrics["compression_attempts"] == 1
        assert metrics["total_duration_ms"] == pytest.approx(300.0)
        assert metrics["stage_timing_ms"] == {"a": 100.0, "b": 200.0}
        assert metrics["timestamp"] == 1000.0

    def test_empty_stage_timing(self, ctx):
        ctx.stage_timing = {}
        stage = MetricsStage()
        stage.run(ctx)
        assert ctx.extras["metrics"]["total_duration_ms"] == 0.0


# =============================================================================
# ToolUseReportStage
# =============================================================================

class TestToolUseReportStage:
    def test_counts_tool_calls(self, ctx):
        ctx.messages = [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "web_search"}},
                {"function": {"name": "read_file"}},
            ]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "web_search"}},
            ]},
        ]
        stage = ToolUseReportStage()
        stage.run(ctx)
        assert ctx.extras["tool_uses"] == {"web_search": 2, "read_file": 1}

    def test_no_tool_calls(self, ctx):
        ctx.messages = [
            {"role": "assistant", "content": "just text"},
            {"role": "user", "content": "question"},
        ]
        stage = ToolUseReportStage()
        stage.run(ctx)
        assert ctx.extras["tool_uses"] == {}

    def test_empty_messages(self, ctx):
        ctx.messages = []
        stage = ToolUseReportStage()
        stage.run(ctx)
        assert ctx.extras["tool_uses"] == {}

    def test_handles_missing_function_name(self, ctx):
        ctx.messages = [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "valid"}},
                {"function": {}},  # missing name
                {},  # missing function entirely
            ]},
        ]
        stage = ToolUseReportStage()
        stage.run(ctx)
        assert ctx.extras["tool_uses"] == {"valid": 1, "?": 2}


# =============================================================================
# default_stages
# =============================================================================

class TestDefaultStages:
    def test_returns_four_stages(self):
        stages = default_stages()
        assert len(stages) == 4

    def test_correct_stage_types(self):
        stages = default_stages()
        assert isinstance(stages[0], LogTurnStartStage)
        assert isinstance(stages[1], LogTurnEndStage)
        assert isinstance(stages[2], MetricsStage)
        assert isinstance(stages[3], ToolUseReportStage)

    def test_all_have_names(self):
        stages = default_stages()
        for s in stages:
            assert s.name, f"Stage {type(s).__name__} has no name"

    def test_correct_stage_names(self):
        stages = default_stages()
        names = [s.name for s in stages]
        assert names == ["log_turn_start", "log_turn_end", "metrics", "tool_use_report"]

    def test_stages_are_independent_instances(self):
        stages1 = default_stages()
        stages2 = default_stages()
        for s1, s2 in zip(stages1, stages2):
            assert s1 is not s2


# =============================================================================
# pipeline_run
# =============================================================================

class TestPipelineRun:
    def test_creates_default_pipeline_when_none_given(self, fake_agent):
        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "ok", "messages": [], "exit_reason": "completed"}
            result = pipeline_run(fake_agent, "hi")

        assert result["response"] == "ok"

    def test_uses_provided_pipeline(self, fake_agent):
        custom_order = []

        class T(TurnStage):
            def __init__(self, name):
                self.name = name
            def run(self, ctx):
                custom_order.append(self.name)
                return ctx

        custom_pipe = TurnPipeline(stages=[_APICallBoundary(), T("custom")])

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "ok", "messages": [], "exit_reason": "completed"}
            pipeline_run(fake_agent, "hi", pipeline=custom_pipe)

        assert custom_order == ["custom"]

    def test_forwards_all_kwargs(self, fake_agent):
        """pipeline_run should pass through all keyword arguments to pipeline.run."""
        cb = lambda m: None

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "ok", "messages": [], "exit_reason": "completed"}
            with patch.object(TurnPipeline, "run") as mock_pipe_run:
                mock_pipe_run.return_value = {"response": "ok"}
                pipeline_run(
                    fake_agent,
                    "hello",
                    system_message="sys",
                    conversation_history=[{"role": "user", "content": "prev"}],
                    task_id="t99",
                    stream_callback=cb,
                    persist_user_message="persisted",
                    persist_user_timestamp=54321.0,
                )

        mock_pipe_run.assert_called_once()
        _, kwargs = mock_pipe_run.call_args
        assert kwargs["user_message"] == "hello"
        assert kwargs["system_message"] == "sys"
        assert kwargs["task_id"] == "t99"
        assert kwargs["stream_callback"] is cb


# =============================================================================
# End-to-end: default pipeline stages
# =============================================================================

class TestDefaultPipelineIntegration:
    def test_full_default_pipeline_runs_successfully(self, fake_agent):
        """Run a TurnPipeline with default_stages and verify the output shape."""
        pipe = TurnPipeline(stages=default_stages())

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {
                "response": "success",
                "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "success"}],
                "interrupted": False,
                "failed": False,
                "exit_reason": "completed",
                "turn_id": "turn-99",
            }
            result = pipe.run(fake_agent, "hello")

        assert result["response"] == "success"
        assert result["exit_reason"] == "completed"
        assert result["failed"] is False
        assert result["turn_id"] == "turn-99"
        # Stage timing should cover all stages
        assert "log_turn_start" in result["stage_timing"]
        assert "log_turn_end" in result["stage_timing"]
        assert "metrics" in result["stage_timing"]
        assert "tool_use_report" in result["stage_timing"]
        assert "_run_conversation" in result["stage_timing"]

    def test_default_pipeline_with_loop_error(self, fake_agent):
        """When run_conversation raises, the pipeline still returns a well-formed result."""
        pipe = TurnPipeline(stages=default_stages())

        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.side_effect = ValueError("something broke")
            result = pipe.run(fake_agent, "hello")

        assert result["failed"] is True
        assert "loop_error" in result["exit_reason"]


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    def test_empty_message(self, fake_agent):
        """Empty user_message should not crash."""
        pipe = TurnPipeline(stages=[])
        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "", "messages": [], "exit_reason": "completed"}
            result = pipe.run(fake_agent, "")
        assert result["response"] == ""

    def test_very_long_message(self, fake_agent):
        """Very long messages should not crash (truncation is in LogTurnStartStage)."""
        pipe = TurnPipeline(stages=default_stages())
        long_msg = "hello " * 5000
        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "ok", "messages": [], "exit_reason": "completed"}
            result = pipe.run(fake_agent, long_msg)
        assert result["response"] == "ok"

    def test_multiple_pre_stage_failures(self, fake_agent):
        """First pre-stage failure should prevent later stages."""

        class FailStage(TurnStage):
            name = "fail"
            def run(self, ctx):
                raise RuntimeError("fail")

        class NeverRunStage(TurnStage):
            name = "never"
            def run(self, ctx):
                pytest.fail("Should not run because earlier stage failed")

        pipe = TurnPipeline(stages=[FailStage(), NeverRunStage()])
        result = pipe.run(fake_agent, "hi")

        assert result["failed"] is True

    def test_empty_pipeline_returns_stage_timing_with_run_conversation(self, fake_agent):
        """Even with no stages, _run_conversation timing is present."""
        pipe = TurnPipeline(stages=[])
        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "ok", "messages": [], "exit_reason": "completed"}
            result = pipe.run(fake_agent, "hi")
        assert "_run_conversation" in result["stage_timing"]
        assert result["response"] == "ok"

    def test_pipeline_run_monkey_patching(self, fake_agent):
        with patch("agent.conversation_loop.run_conversation") as mock_loop:
            mock_loop.return_value = {"response": "ok", "messages": [], "exit_reason": "completed"}
            result = pipeline_run(fake_agent, "test", task_id="integration")
        assert isinstance(result, dict)
        assert result["response"] == "ok"
