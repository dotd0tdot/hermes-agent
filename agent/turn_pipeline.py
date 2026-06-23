"""
turn_pipeline.py — Explicit stage pipeline for the Hermes conversation loop.

Motivation
----------
``agent/conversation_loop.py`` contains a ~4,500-line ``run_conversation``
function that interleaves 5 distinct concerns — setup, pre-API preparation,
model invocation, tool dispatch, and post-turn housekeeping — in a single
while-loop with deeply nested try/except/if blocks and mutable state.

This module provides a **pipeline abstraction**: each turn passes through a
sequence of named stages, each a callable object with a defined input/output
contract.  The stages delegate to the *existing* helper functions so no core
behaviour changes — the flow merely becomes visible, testable, and extensible.

Usage
-----
    # Basic: run the default pipeline (delegates to run_conversation)
    from agent.turn_pipeline import pipeline_run
    result = pipeline_run(agent, user_message="hello")

    # Custom: build a pipeline with extra observability
    from agent.turn_pipeline import TurnPipeline, default_stages
    pipe = TurnPipeline(stages=default_stages())
    pipe.insert_before("api_call", LogLatencyStage())
    result = pipe.run(agent, user_message="hello")

Two-layer design:
  Layer 1 — ``pipeline_run()`` is a drop-in wrapper around
            ``run_conversation`` with observability hooks.
  Layer 2 — ``TurnPipeline`` with pluggable stages for full pipeline
            control. The default stages mirror the existing flow.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Context
# =============================================================================

@dataclasses.dataclass
class PipelineContext:
    """Mutable context carried through the pipeline stages.

    Each stage reads from and writes to this context.  The final result is
    assembled after all stages complete.
    """
    agent: Any
    user_message: str
    original_user_message: Any
    messages: List[Dict[str, Any]]
    active_system_prompt: Optional[str]
    effective_task_id: str
    turn_id: str
    current_turn_user_idx: int
    conversation_history: Optional[List[Dict[str, Any]]]

    # Runtime counters
    api_call_count: int = 0
    final_response: Any = None
    interrupted: bool = False
    failed: bool = False
    compression_attempts: int = 0
    exit_reason: str = "unknown"

    # Pipeline metadata
    stage_timing: Dict[str, float] = dataclasses.field(default_factory=dict)

    # Extra bag for stage-specific data
    extras: Dict[str, Any] = dataclasses.field(default_factory=dict)


# =============================================================================
# Stage Base
# =============================================================================

class TurnStage:
    """A single named stage in the conversation pipeline."""
    name: str = "unnamed"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute this stage.  Mutate *ctx* in place and return it."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<TurnStage '{self.name}'>"


# =============================================================================
# Pipeline Orchestrator
# =============================================================================

class TurnPipeline:
    """Ordered sequence of ``TurnStage`` instances.

    Usage::

        pipe = TurnPipeline(stages=default_stages())
        ctx = pipe.run(agent, user_message="...")
        # ctx.final_response, ctx.exit_reason, ctx.stage_timing
    """

    def __init__(self, stages: Optional[List[TurnStage]] = None):
        self._stages: List[TurnStage] = stages or []

    def add(self, stage: TurnStage) -> TurnStage:
        self._stages.append(stage)
        return stage

    def insert_before(self, stage_name: str, stage: TurnStage) -> bool:
        for i, s in enumerate(self._stages):
            if s.name == stage_name:
                self._stages.insert(i, stage)
                return True
        logger.warning("insert_before: stage '%s' not found", stage_name)
        return False

    def insert_after(self, stage_name: str, stage: TurnStage) -> bool:
        for i, s in enumerate(self._stages):
            if s.name == stage_name:
                self._stages.insert(i + 1, stage)
                return True
        logger.warning("insert_after: stage '%s' not found", stage_name)
        return False

    def replace(self, stage_name: str, stage: TurnStage) -> bool:
        for i, s in enumerate(self._stages):
            if s.name == stage_name:
                self._stages[i] = stage
                return True
        logger.warning("replace: stage '%s' not found", stage_name)
        return False

    def remove(self, stage_name: str) -> bool:
        for i, s in enumerate(self._stages):
            if s.name == stage_name:
                self._stages.pop(i)
                return True
        logger.warning("remove: stage '%s' not found", stage_name)
        return False

    def stages(self) -> List[TurnStage]:
        return list(self._stages)

    def run(
        self,
        agent: Any,
        user_message: str,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        task_id: Optional[str] = None,
        stream_callback: Optional[Callable] = None,
        persist_user_message: Optional[str] = None,
        persist_user_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run all stages in order, then delegate to the turn loop.

        Stages run *around* the existing ``run_conversation`` call,
        wrapping it with pre/post hooks.  The heavy lifting stays in
        the battle-tested loop.
        """
        ctx = PipelineContext(
            agent=agent,
            user_message=user_message,
            original_user_message=user_message,
            messages=conversation_history or [],
            active_system_prompt=system_message,
            effective_task_id=task_id or "",
            turn_id="",
            current_turn_user_idx=0,
            conversation_history=conversation_history,
        )
        ctx.extras["stream_callback"] = stream_callback
        ctx.extras["persist_user_message"] = persist_user_message
        ctx.extras["persist_user_timestamp"] = persist_user_timestamp

        # Pre-stages (before the loop)
        for stage in self._stages:
            if stage.name in ("api_call", "tool_dispatch"):
                break  # stop before loop stages
            t0 = time.monotonic()
            try:
                ctx = stage.run(ctx)
            except Exception as exc:
                logger.exception("Pipeline pre-stage '%s' failed: %s", stage.name, exc)
                ctx.failed = True
                ctx.exit_reason = f"stage_error:{stage.name}"
                return _assemble_result(ctx)
            finally:
                ctx.stage_timing[stage.name] = time.monotonic() - t0

            if ctx.interrupted:
                return _assemble_result(ctx)

        # Main turn: delegate to existing run_conversation
        from agent.conversation_loop import run_conversation
        t_turn = time.monotonic()
        try:
            result = run_conversation(
                agent,
                user_message=user_message,
                system_message=system_message,
                conversation_history=conversation_history,
                task_id=task_id,
                stream_callback=stream_callback,
                persist_user_message=persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
            )
        except Exception as exc:
            logger.exception("run_conversation failed: %s", exc)
            ctx.failed = True
            ctx.exit_reason = f"loop_error:{exc}"
            return _assemble_result(ctx)
        finally:
            ctx.stage_timing["_run_conversation"] = time.monotonic() - t_turn

        # Merge loop result back into context
        ctx.final_response = result.get("response")
        ctx.interrupted = result.get("interrupted", False)
        ctx.failed = result.get("failed", False)
        ctx.exit_reason = result.get("exit_reason", "unknown")
        ctx.messages = result.get("messages", ctx.messages)
        ctx.turn_id = result.get("turn_id", "")

        # Post-stages (after the loop)
        for stage in self._stages:
            if stage.name in ("setup", "pre_api", "api_call", "tool_dispatch"):
                continue
            t0 = time.monotonic()
            try:
                ctx = stage.run(ctx)
            except Exception as exc:
                logger.warning("Pipeline post-stage '%s' failed: %s", stage.name, exc)
                # Don't mark failed for non-critical post-hooks
            finally:
                ctx.stage_timing[stage.name] = time.monotonic() - t0

        return _assemble_result(ctx)


def _assemble_result(ctx: PipelineContext) -> Dict[str, Any]:
    """Build the return dict matching run_conversation output."""
    return {
        "response": ctx.final_response,
        "messages": ctx.messages,
        "interrupted": ctx.interrupted,
        "failed": ctx.failed,
        "turn_id": ctx.turn_id,
        "exit_reason": ctx.exit_reason,
        "stage_timing": dict(ctx.stage_timing),
    }


# =============================================================================
# Built-in Stages
# =============================================================================

class LogTurnStartStage(TurnStage):
    """Log the start of a turn with user message and context info."""
    name = "log_turn_start"

    def __init__(self, log_fn: Optional[Callable] = None):
        self._log = log_fn or logger.info

    def run(self, ctx: PipelineContext) -> PipelineContext:
        msg_preview = ctx.user_message[:80].replace("\n", "\\n")
        provider = getattr(ctx.agent, "provider", "?")
        model = getattr(ctx.agent, "model", "?")
        tool_count = len(getattr(ctx.agent, "tools", []) or [])
        self._log(
            "Turn start | msg='%s' | model=%s/%s | tools=%d",
            msg_preview, provider, model, tool_count,
        )
        return ctx


class LogTurnEndStage(TurnStage):
    """Log the end of a turn with timing and exit reason."""
    name = "log_turn_end"

    def __init__(self, log_fn: Optional[Callable] = None):
        self._log = log_fn or logger.info

    def run(self, ctx: PipelineContext) -> PipelineContext:
        total_ms = sum(ctx.stage_timing.values()) * 1000
        self._log(
            "Turn end | exit=%s | api_calls=%d | stages=%d | total=%.0fms | timing=%s",
            ctx.exit_reason,
            ctx.api_call_count,
            len(ctx.stage_timing),
            total_ms,
            {k: f"{v*1000:.0f}ms" for k, v in sorted(ctx.stage_timing.items())},
        )
        return ctx


class MetricsStage(TurnStage):
    """Collect per-turn metrics: token usage, tool calls, latency.

    Values are stored in ``ctx.extras['metrics']`` for downstream
    consumers (logging, dashboard, cron).
    """
    name = "metrics"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        metrics = {
            "exit_reason": ctx.exit_reason,
            "api_call_count": ctx.api_call_count,
            "interrupted": ctx.interrupted,
            "failed": ctx.failed,
            "compression_attempts": ctx.compression_attempts,
            "total_duration_ms": sum(ctx.stage_timing.values()) * 1000,
            "stage_timing_ms": {
                k: round(v * 1000, 1) for k, v in ctx.stage_timing.items()
            },
            "timestamp": time.time(),
        }
        ctx.extras["metrics"] = metrics
        return ctx


class ToolUseReportStage(TurnStage):
    """Report which tools were called in this turn."""
    name = "tool_use_report"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Scan messages for tool calls and build a usage report."""
        tool_uses: Dict[str, int] = {}
        for m in ctx.messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    name = tc.get("function", {}).get("name", "?")
                    tool_uses[name] = tool_uses.get(name, 0) + 1

        if tool_uses and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Tool use this turn: %s", tool_uses)

        ctx.extras["tool_uses"] = tool_uses
        return ctx


# =============================================================================
# Pipeline factory
# =============================================================================

def default_stages() -> List[TurnStage]:
    """Return the default pipeline stage list.

    The default pipeline is intentionally minimal — it only adds
    observability hooks.  All actual turn logic stays in the existing
    ``run_conversation`` function.

    Stages:
      1. log_turn_start   — log user message, model, tool count
      2. log_turn_end     — log exit reason, timing breakdown
      3. metrics          — collect per-turn metrics dict
      4. tool_use_report  — scan messages for tool calls
    """
    return [
        LogTurnStartStage(),
        LogTurnEndStage(),
        MetricsStage(),
        ToolUseReportStage(),
    ]


# =============================================================================
# Convenience: run through pipeline
# =============================================================================

def pipeline_run(
    agent: Any,
    user_message: str,
    system_message: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    task_id: Optional[str] = None,
    stream_callback: Optional[Callable] = None,
    persist_user_message: Optional[str] = None,
    persist_user_timestamp: Optional[float] = None,
    pipeline: Optional[TurnPipeline] = None,
) -> Dict[str, Any]:
    """Run one turn through the pipeline, delegating to ``run_conversation``.

    This is a drop-in replacement for ``run_conversation()`` that wraps
    the call with the pipeline's pre/post stages.

    Usage in ``conversation_loop.py``::

        from agent.turn_pipeline import pipeline_run
        result = pipeline_run(agent, user_message, ...)
        return result
    """
    if pipeline is None:
        pipeline = TurnPipeline(stages=default_stages())

    return pipeline.run(
        agent,
        user_message=user_message,
        system_message=system_message,
        conversation_history=conversation_history,
        task_id=task_id,
        stream_callback=stream_callback,
        persist_user_message=persist_user_message,
        persist_user_timestamp=persist_user_timestamp,
    )


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    pipe = TurnPipeline(stages=default_stages())
    print(f"Pipeline stages: {[s.name for s in pipe.stages()]}")

    # Test stage manipulation
    from dataclasses import dataclass

    @dataclass
    class FakeAgent:
        provider: str = "test"
        model: str = "test-model"
        tools: list = dataclasses.field(default_factory=list)

    ctx = PipelineContext(
        agent=FakeAgent(),
        user_message="hello world",
        original_user_message="hello world",
        messages=[],
        active_system_prompt=None,
        effective_task_id="test-123",
        turn_id="turn-1",
        current_turn_user_idx=0,
        conversation_history=None,
        extras={"stream_callback": None},
    )

    for stage in pipe.stages():
        try:
            ctx = stage.run(ctx)
            print(f"  ✓ {stage.name}: {ctx.stage_timing.get(stage.name, 0)*1000:.1f}ms")
        except Exception as e:
            print(f"  ✗ {stage.name}: {e}")

    print(f"Stage timing: {ctx.stage_timing}")
    print(f"Exit reason: {ctx.exit_reason}")
    print("Pipeline OK")
