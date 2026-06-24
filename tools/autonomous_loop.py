"""
Autonomous loop engine for Hermes Agent.

The ``AutonomousLoop`` class implements an in-process event loop that:
1. Scans the system for problems (logs, processes, git).
2. Reads an optional backlog file for user-defined tasks.
3. Picks the highest-priority task (NBA — Next Best Action).
4. Executes it via a caller-provided callback.
5. Persists state so it survives restarts.
6. Learns from outcomes (success/failure) via memory.

Features:
- **Result validation**: re-scans after execution to verify the problem is solved.
- **Task decomposition**: tasks may contain ``subtasks`` lists executed sequentially.
- **Parallel execution**: independent tasks run concurrently via ThreadPoolExecutor.
- **Feedback loop**: execution history is passed to the task generator.
- **Self-directed planning**: agent creates multi-step plans with goals, executes
  them step by step, and replans based on outcomes.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (overridable via config.yaml → autonomous.*)
# ---------------------------------------------------------------------------
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_SLEEP_ON_SUCCESS_S = 30
DEFAULT_SLEEP_ON_FAILURE_S = 180
DEFAULT_SCAN_INTERVAL_S = 60
DEFAULT_MAX_WORKERS = 4  # parallel execution pool size


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Task dataclass-like dicts (plain dicts for JSON serializability)
# ---------------------------------------------------------------------------

def _make_task(
    id: str,
    title: str,
    priority: int,
    source: str,
    attempts: int = 0,
    status: str = "pending",
    error: str = "",
    subtasks: Optional[List[dict]] = None,
) -> dict:
    task = {
        "id": id,
        "title": title,
        "priority": priority,
        "source": source,
        "attempts": attempts,
        "status": status,
        "error": error,
    }
    if subtasks:
        task["subtasks"] = subtasks
    return task


# ---------------------------------------------------------------------------
# System scanner — produces tasks from live system state
# ---------------------------------------------------------------------------

class SystemScanner:
    """Detects problems by scanning logs, processes, and git state."""

    def __init__(self, hermes_home: Path):
        self.hermes_home = hermes_home
        self.logs_dir = hermes_home / "logs"
        self.known_roots = [
            Path.home() / ".hermes" / "hermes-agent",
            Path.home() / "minecraft" / "rustmc",
            Path.home() / "plasma-workspace",
        ]

    def scan(self) -> List[dict]:
        """Return a list of detected problem tasks."""
        problems: List[dict] = []
        problems.extend(self._scan_hermes_logs())
        problems.extend(self._scan_git_repos())
        problems.extend(self._scan_systemd_services())
        return problems

    def scan_ids(self) -> set:
        """Return a set of current problem IDs (for validation comparison)."""
        return {p["id"] for p in self.scan()}

    # --- Log scanning -------------------------------------------------------

    def _scan_hermes_logs(self) -> List[dict]:
        """Look for ERROR/WARNING lines in recent hermes logs."""
        problems: List[dict] = []
        if not self.logs_dir.exists():
            return problems

        log_files = sorted(self.logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        # Only check the most recent log file
        for log_file in log_files[:1]:
            try:
                # Read last 200 lines only
                result = subprocess.run(
                    ["tail", "-n", "200", str(log_file)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    continue
                lines = result.stdout.splitlines()
                errors = [
                    line for line in lines
                    if re.search(r"\b(ERROR|CRITICAL|FATAL)\b", line)
                    and not re.search(r"(DEBUG|TRACE)", line)
                ]
                if len(errors) >= 3:
                    # Deduplicate by pattern
                    seen_patterns: set[str] = set()
                    unique_errors = []
                    for line in errors[-10:]:
                        # Normalize: strip timestamps and PIDs
                        pattern = re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", line)
                        pattern = re.sub(r"\b\d{3,}\b", "N", pattern)
                        if pattern not in seen_patterns:
                            seen_patterns.add(pattern)
                            unique_errors.append(line)

                    if unique_errors:
                        problems.append(_make_task(
                            id=f"log-errors-{log_file.stem}",
                            title=f"Log errors in {log_file.name} ({len(unique_errors)} unique)",
                            priority=10,
                            source="auto_detect",
                        ))
                        # Store first error as context
                        problems[-1]["context"] = unique_errors[0][:200]
            except (subprocess.TimeoutExpired, OSError):
                pass
        return problems

    # --- Git scanning -------------------------------------------------------

    def _scan_git_repos(self) -> List[dict]:
        """Check known git repos for uncommitted changes, unpushed branches, etc."""
        problems: List[dict] = []
        for root in self.known_roots:
            if not (root / ".git").exists():
                continue
            try:
                status_result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True, text=True,
                    cwd=str(root),
                )
                if status_result.returncode != 0:
                    continue
                dirty = [l for l in status_result.stdout.splitlines() if l.strip()]
                if len(dirty) > 10:
                    problems.append(_make_task(
                        id=f"git-dirty-{root.name}",
                        title=f"{root.name}: {len(dirty)} uncommitted changes",
                        priority=5,
                        source="auto_detect",
                    ))
                    problems[-1]["context"] = "\n".join(dirty[:5])

                # Check for unpushed commits
                push_result = subprocess.run(
                    ["git", "log", "--oneline", "@{upstream}..HEAD"],
                    capture_output=True, text=True,
                    cwd=str(root),
                )
                if push_result.returncode == 0 and push_result.stdout.strip():
                    unpushed = push_result.stdout.strip().splitlines()
                    problems.append(_make_task(
                        id=f"git-unpushed-{root.name}",
                        title=f"{root.name}: {len(unpushed)} unpushed commits",
                        priority=5,
                        source="auto_detect",
                    ))
                    problems[-1]["context"] = unpushed[0]
            except (subprocess.TimeoutExpired, OSError):
                pass
        return problems

    # --- Systemd service scanning -------------------------------------------

    def _scan_systemd_services(self) -> List[dict]:
        """Check for failed systemd user services."""
        problems: List[dict] = []
        try:
            result = subprocess.run(
                ["systemctl", "--user", "list-units", "--state=failed", "--no-legend"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                return problems
            failed = [
                line.split()[0] for line in result.stdout.splitlines()
                if line.strip() and ".service" in line
            ]
            for svc in failed:
                problems.append(_make_task(
                    id=f"systemd-failed-{svc}",
                    title=f"Failed service: {svc}",
                    priority=10,
                    source="auto_detect",
                ))
                problems[-1]["context"] = svc
        except (subprocess.TimeoutExpired, OSError):
            pass
        return problems


# ---------------------------------------------------------------------------
# Backlog parser
# ---------------------------------------------------------------------------

class BacklogReader:
    """Reads tasks from a markdown backlog file."""

    def __init__(self, backlog_path: Path):
        self.backlog_path = backlog_path

    def read(self) -> List[dict]:
        """Parse the backlog and return pending tasks."""
        if not self.backlog_path.exists():
            return []
        try:
            content = self.backlog_path.read_text(encoding="utf-8")
        except OSError:
            return []

        tasks: List[dict] = []
        task_id = 0
        for line in content.splitlines():
            # Match active (unchecked) items: "- [ ] task text"
            m = re.match(r"^[-*]\s+\[ \]\s+(.+)$", line.strip())
            if m:
                task_id += 1
                tasks.append(_make_task(
                    id=f"backlog-{task_id}",
                    title=m.group(1).strip(),
                    priority=5,
                    source="backlog",
                ))
        return tasks


# ---------------------------------------------------------------------------
# Self-directed planner — creates multi-step plans with goals
# ---------------------------------------------------------------------------

def _make_plan(
    goal: str,
    steps: List[dict],
    context: str = "",
    plan_id: Optional[str] = None,
) -> dict:
    """Create a plan dict.

    Each step: {"title": str, "description": str, "status": "pending"}
    """
    return {
        "id": plan_id or f"plan-{uuid.uuid4().hex[:12]}",
        "goal": goal,
        "created_at": _now_iso(),
        "status": "active",
        "current_step": 0,
        "steps": steps,
        "context": context,
        "replan_count": 0,
    }


class Planner:
    """Manages self-directed plans: create, advance, replan, persist.

    Plans are stored in ``~/.hermes/autonomous-plans.json``.
    Only one plan can be active at a time.
    """

    def __init__(self, hermes_home: Path):
        self.plans_path = hermes_home / "autonomous-plans.json"
        self.plans: List[dict] = self._load()

    # --- Persistence --------------------------------------------------------

    def _load(self) -> List[dict]:
        data = _load_json(self.plans_path, {"plans": []})
        return data.get("plans", [])

    def _save(self) -> None:
        _save_json(self.plans_path, {"plans": self.plans})

    # --- Active plan --------------------------------------------------------

    def active_plan(self) -> Optional[dict]:
        """Return the current active plan, or None."""
        for p in self.plans:
            if p["status"] == "active":
                return p
        return None

    def has_active_plan(self) -> bool:
        return self.active_plan() is not None

    # --- Create -------------------------------------------------------------

    def create_plan(self, goal: str, steps: List[dict], context: str = "") -> dict:
        """Create and activate a new plan. Deactivates any existing active plan."""
        # Deactivate old plan
        for p in self.plans:
            if p["status"] == "active":
                p["status"] = "superseded"

        plan = _make_plan(goal=goal, steps=steps, context=context)
        self.plans.append(plan)
        self._save()
        return plan

    # --- Advance ------------------------------------------------------------

    def current_step(self) -> Optional[dict]:
        """Return the current step of the active plan, or None."""
        plan = self.active_plan()
        if not plan:
            return None
        idx = plan["current_step"]
        if idx < len(plan["steps"]):
            return plan["steps"][idx]
        return None

    def advance(self, result: dict) -> Optional[dict]:
        """Mark current step done, advance to next.

        Args:
            result: {"success": bool, "message": str}

        Returns:
            The next step dict, or None if plan is complete.
        """
        plan = self.active_plan()
        if not plan:
            return None

        idx = plan["current_step"]
        step = plan["steps"][idx]
        step["status"] = "completed" if result.get("success") else "failed"
        step["result"] = result.get("message", "")

        if not result.get("success"):
            plan["status"] = "failed"
            self._save()
            return None

        plan["current_step"] += 1

        # Check if plan is complete
        if plan["current_step"] >= len(plan["steps"]):
            plan["status"] = "completed"
            self._save()
            return None

        self._save()
        return plan["steps"][plan["current_step"]]

    # --- Replan -------------------------------------------------------------

    def replan(self, new_steps: List[dict], reason: str = "") -> Optional[dict]:
        """Replace remaining steps in the active plan.

        Keeps completed steps, replaces pending ones.
        """
        plan = self.active_plan()
        if not plan:
            return None

        # Keep completed steps
        completed = [s for s in plan["steps"] if s.get("status") == "completed"]
        plan["steps"] = completed + new_steps
        plan["current_step"] = len(completed)
        plan["replan_count"] = plan.get("replan_count", 0) + 1
        plan["status"] = "active"

        if reason:
            plan.setdefault("replan_reasons", []).append({
                "at": _now_iso(),
                "reason": reason,
                "step_index": plan["current_step"],
            })

        self._save()
        return plan["steps"][plan["current_step"]] if plan["steps"] else None

    # --- Abandon ------------------------------------------------------------

    def abandon(self, reason: str = "") -> None:
        """Mark the active plan as abandoned."""
        plan = self.active_plan()
        if plan:
            plan["status"] = "abandoned"
            plan["abandon_reason"] = reason
            self._save()

    # --- History ------------------------------------------------------------

    def recent_plans(self, limit: int = 5) -> List[dict]:
        """Return recent plans (newest first)."""
        return sorted(self.plans, key=lambda p: p["created_at"], reverse=True)[:limit]

    # --- Task conversion ----------------------------------------------------

    def active_step_as_task(self) -> Optional[dict]:
        """Convert the current plan step to a task dict for the loop."""
        plan = self.active_plan()
        step = self.current_step()
        if not plan or not step:
            return None

        return _make_task(
            id=f"{plan['id']}-step{plan['current_step']}",
            title=f"[{plan['goal'][:40]}] {step['title']}",
            priority=8,  # Plans get high priority
            source="plan",
        )


# ---------------------------------------------------------------------------
# Result validation
# ---------------------------------------------------------------------------

class ResultValidator:
    """Validates task outcomes by re-scanning the system."""

    def __init__(self, scanner: SystemScanner):
        self.scanner = scanner
        self._baseline_ids: Optional[set] = None

    def take_baseline(self) -> None:
        """Capture current problem IDs before task execution."""
        self._baseline_ids = self.scanner.scan_ids()

    def validate(self, task_id: str) -> dict:
        """
        Re-scan after execution and compare with baseline.

        Returns:
            {"verified": bool, "resolved": bool, "new_problems": list}
        """
        if self._baseline_ids is None:
            return {"verified": False, "resolved": False, "new_problems": []}

        current_ids = self.scanner.scan_ids()
        resolved = task_id in self._baseline_ids and task_id not in current_ids
        new_problems = list(current_ids - self._baseline_ids)
        self._baseline_ids = None

        return {
            "verified": True,
            "resolved": resolved,
            "new_problems": new_problems,
        }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class AutonomousLoop:
    """
    Self-driving task loop.

    Usage::

        loop = AutonomousLoop(hermes_home=Path("/home/dot/.hermes"))
        loop.run(execute_callback)

    ``execute_callback`` receives a task dict and returns a result dict
    with keys: ``success`` (bool), ``message`` (str, optional).

    Features:
    - Result validation via re-scanning
    - Task decomposition (subtasks)
    - Parallel execution (max_workers)
    - Feedback to task_generator(history)
    - Self-directed planning via Planner
    """

    def __init__(
        self,
        hermes_home: Optional[Path] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
        sleep_on_success_s: int = DEFAULT_SLEEP_ON_SUCCESS_S,
        sleep_on_failure_s: int = DEFAULT_SLEEP_ON_FAILURE_S,
        scan_interval_s: int = DEFAULT_SCAN_INTERVAL_S,
        task_generator: Optional[Callable[[List[dict]], List[dict]]] = None,
        plan_generator: Optional[Callable[[List[dict]], Optional[dict]]] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ):
        self.hermes_home = hermes_home or get_hermes_home()
        self.state_path = self.hermes_home / "autonomous-state.json"
        self.backlog_path = self.hermes_home / "autonomous-backlog.md"
        self.log_path = self.hermes_home / "logs" / "autonomous.log"

        self.max_iterations = max_iterations
        self.max_consecutive_failures = max_consecutive_failures
        self.sleep_on_success_s = sleep_on_success_s
        self.sleep_on_failure_s = sleep_on_failure_s
        self.scan_interval_s = scan_interval_s

        self.scanner = SystemScanner(self.hermes_home)
        self.backlog = BacklogReader(self.backlog_path)
        self.validator = ResultValidator(self.scanner)
        self.planner = Planner(self.hermes_home)
        self.task_generator = task_generator
        self.plan_generator = plan_generator
        self.max_workers = max_workers
        self.state = self._load_state()
        self._stop_event = threading.Event()

    # --- State persistence --------------------------------------------------

    def _load_state(self) -> dict:
        default = {
            "version": 1,
            "started_at": _now_iso(),
            "last_iteration_at": "",
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "status": "idle",
            "current_task": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "consecutive_failures": 0,
            "learning_log": [],
            "execution_history": [],
        }
        loaded = _load_json(self.state_path, default)
        # Ensure all keys exist (forward-compat)
        for k, v in default.items():
            if k not in loaded:
                loaded[k] = v
        return loaded

    def _save_state(self) -> None:
        _save_json(self.state_path, self.state)

    # --- Task selection (NBA) ----------------------------------------------

    def _select_next_task(self) -> Optional[dict]:
        """Pick the highest-priority available task.

        Priority order:
        1. Active plan step (highest)
        2. System-detected problems
        3. Backlog tasks
        4. AI-generated tasks
        """
        # 0. Active plan step (priority 8)
        plan_task = self.planner.active_step_as_task()
        if plan_task is not None:
            return plan_task

        # 1. System-detected problems (priority 10)
        if self.state["iteration"] % 3 == 0:  # Scan every 3rd iteration
            detected = self.scanner.scan()
            candidates = list(detected)
        else:
            candidates = []

        # 2. Backlog tasks (priority 5)
        candidates.extend(self.backlog.read())

        # 3. AI-generated tasks (priority 4) — pass history for feedback
        if not candidates and self.task_generator is not None:
            try:
                history = self.state.get("execution_history", [])
                generated = self.task_generator(history)
                if generated:
                    candidates.extend(generated)
                    self._log("Generated %d tasks from agent", len(generated))
            except Exception as e:
                self._log("Task generator failed: %s", e)

        # 4. Try to create a plan if nothing to do
        if not candidates and self.plan_generator is not None and not self.planner.has_active_plan():
            try:
                history = self.state.get("execution_history", [])
                plan = self.plan_generator(history)
                if plan and plan.get("steps"):
                    self.planner.create_plan(
                        goal=plan["goal"],
                        steps=plan["steps"],
                        context=plan.get("context", ""),
                    )
                    self._log("Created plan: %s (%d steps)", plan["goal"], len(plan["steps"]))
                    return self.planner.active_step_as_task()
            except Exception as e:
                self._log("Plan generator failed: %s", e)

        # 5. Fallback — nothing to do
        if not candidates:
            return None

        # Sort by priority descending
        candidates.sort(key=lambda t: t["priority"], reverse=True)

        # Skip already-completed or blocked tasks
        completed_ids = {t["id"] for t in self.state["completed_tasks"]}
        blocked_ids = {t["id"] for t in self.state["failed_tasks"] if t.get("attempts", 0) >= 3}

        for c in candidates:
            if c["id"] not in completed_ids and c["id"] not in blocked_ids:
                return c
        return None

    def _select_batch(self) -> List[dict]:
        """Select multiple independent tasks for parallel execution."""
        candidates: List[dict] = []

        # Collect from scanner (always on batch iterations)
        detected = self.scanner.scan()
        candidates.extend(detected)

        # Collect from backlog
        candidates.extend(self.backlog.read())

        if not candidates and self.task_generator is not None:
            try:
                history = self.state.get("execution_history", [])
                generated = self.task_generator(history)
                if generated:
                    candidates.extend(generated)
            except Exception as e:
                self._log("Task generator failed: %s", e)

        if not candidates:
            return []

        candidates.sort(key=lambda t: t["priority"], reverse=True)

        completed_ids = {t["id"] for t in self.state["completed_tasks"]}
        blocked_ids = {t["id"] for t in self.state["failed_tasks"] if t.get("attempts", 0) >= 3}

        # Return up to max_workers independent tasks (no shared IDs)
        batch = []
        seen_ids = set()
        for c in candidates:
            if c["id"] not in completed_ids and c["id"] not in blocked_ids and c["id"] not in seen_ids:
                batch.append(c)
                seen_ids.add(c["id"])
                if len(batch) >= self.max_workers:
                    break
        return batch

    # --- Task decomposition ------------------------------------------------

    def _expand_task(self, task: dict) -> List[dict]:
        """
        If task has subtasks, return them as a sequential list.
        Otherwise return [task].
        """
        subtasks = task.get("subtasks")
        if not subtasks:
            return [task]
        # Wrap each subtask with parent metadata
        expanded = []
        for i, st in enumerate(subtasks):
            st.setdefault("id", f"{task['id']}-sub{i}")
            st.setdefault("priority", task["priority"])
            st.setdefault("source", task["source"])
            st["_parent_id"] = task["id"]
            expanded.append(st)
        return expanded

    # --- Execution wrapper --------------------------------------------------

    def _execute_task(self, task: dict, execute_fn: Callable[[dict], dict]) -> dict:
        """Run a task via the callback, with error handling and validation."""
        task["attempts"] = task.get("attempts", 0) + 1
        task["status"] = "running"
        self.state["current_task"] = task
        self._save_state()

        # Take validation baseline before execution
        self.validator.take_baseline()

        try:
            result = execute_fn(task)
            if not isinstance(result, dict):
                result = {"success": True, "message": str(result)}
        except Exception as e:
            logger.exception("Task execution failed: %s", task["id"])
            result = {"success": False, "message": f"{type(e).__name__}: {e}"}

        # Validate result
        validation = self.validator.validate(task["id"])
        result["validation"] = validation

        if validation["verified"] and not validation["resolved"] and result.get("success"):
            # Task said it succeeded but the problem persists
            result["message"] += " [verified: problem persists]"
            self._log("  ⚠ Validation: problem persists after '%s'", task["title"])

        if validation["new_problems"]:
            self._log("  ⚠ Validation: %d new problems detected", len(validation["new_problems"]))

        return result

    # --- Parallel execution -------------------------------------------------

    def _execute_batch(
        self, tasks: List[dict], execute_fn: Callable[[dict], dict]
    ) -> List[Tuple[dict, dict]]:
        """Execute independent tasks in parallel.

        Returns list of (task, result) tuples.
        """
        if len(tasks) == 1:
            return [(tasks[0], self._execute_task(tasks[0], execute_fn))]

        results = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tasks))) as pool:
            futures = {
                pool.submit(self._execute_task, task, execute_fn): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"success": False, "message": f"Thread error: {e}"}
                results.append((task, result))
        return results

    # --- Single tick (one iteration) ----------------------------------------

    def _tick(self, execute_fn: Callable[[dict], dict]) -> dict:
        """Execute one iteration. Returns the tick result dict."""
        self.state["iteration"] += 1
        self.state["last_iteration_at"] = _now_iso()

        tick_result = {
            "iteration": self.state["iteration"],
            "action": None,
            "task": None,
            "success": None,
        }

        # Decide: batch (parallel) or single
        if self.max_workers > 1 and self.state["iteration"] % 2 == 0:
            batch = self._select_batch()
            if not batch:
                tick_result["action"] = "idle"
                return tick_result

            self._log(
                "Iteration %d: batch of %d tasks (parallel)",
                self.state["iteration"], len(batch),
            )

            batch_results = self._execute_batch(batch, execute_fn)
            for task, result in batch_results:
                self._process_result(task, result)
            tick_result["action"] = "batch"
            tick_result["task"] = f"{len(batch)} tasks"
            tick_result["success"] = all(r.get("success") for _, r in batch_results)
        else:
            task = self._select_next_task()
            if task is None:
                tick_result["action"] = "idle"
                return tick_result

            # Expand subtasks
            subtasks = self._expand_task(task)
            if len(subtasks) > 1:
                self._log(
                    "Iteration %d: [%s] %s → %d subtasks",
                    self.state["iteration"],
                    task["source"], task["title"], len(subtasks),
                )
            else:
                self._log(
                    "Iteration %d: [%s] %s (priority=%d, attempt=%d)",
                    self.state["iteration"],
                    task["source"], task["title"], task["priority"],
                    task.get("attempts", 0) + 1,
                )

            # Execute subtasks sequentially
            all_success = True
            for st in subtasks:
                result = self._execute_task(st, execute_fn)
                success = result.get("success", False)
                self._process_result(st, result)
                if not success:
                    all_success = False
                    break  # Stop subtask chain on failure

            # Record parent task outcome
            if len(subtasks) > 1:
                parent_result = {
                    "success": all_success,
                    "message": f"{len(subtasks)} subtasks: {'all done' if all_success else 'stopped on failure'}",
                }
                self._record_history(task, parent_result)
                # Mark parent as completed so it's not re-selected
                task["status"] = "completed"
                self.state["completed_tasks"].append(task)

            # Advance plan if task was from a plan
            if task.get("source") == "plan" and self.planner.has_active_plan():
                step_result = {"success": all_success, "message": task.get("error", "")}
                next_step = self.planner.advance(step_result)
                if next_step:
                    self._log("  → Plan next step: %s", next_step["title"])
                elif self.planner.active_plan() is None:
                    self._log("  ✓ Plan completed!")

            tick_result["action"] = "execute"
            tick_result["task"] = task["title"]
            tick_result["success"] = all_success

        self._save_state()
        return tick_result

    # --- Daemon mode (run forever) ------------------------------------------

    def run_forever(self, execute_fn: Callable[[dict], dict]) -> None:
        """Run the autonomous loop indefinitely until stop() is called.

        This is the daemon mode — call from a daemon thread.
        Never returns (loops forever).
        """
        self.state["status"] = "running"
        self.state["started_at"] = _now_iso()
        self._stop_event.clear()
        self._save_state()

        self._log("Autonomous daemon started (max_workers=%d)", self.max_workers)

        try:
            while not self._stop_event.is_set():
                try:
                    tick = self._tick(execute_fn)

                    # If idle, sleep longer
                    if tick["action"] == "idle":
                        self._log("Nothing to do — sleeping %ds", self.sleep_on_success_s)
                        self._stop_event.wait(self.sleep_on_success_s)
                        continue

                    # Reset consecutive failures after successful tick
                    if tick["success"]:
                        self.state["consecutive_failures"] = 0

                    # Check stop conditions
                    if self.state["consecutive_failures"] >= self.max_consecutive_failures:
                        self._log("STOPPED: %d consecutive failures — pausing (not exiting)",
                                  self.state["consecutive_failures"])
                        # Reset failures after pause so daemon can retry later
                        self.state["consecutive_failures"] = 0
                        self._stop_event.wait(self.sleep_on_failure_s * 5)
                        continue

                    # Adaptive sleep between ticks
                    success_last = tick.get("success", False)
                    sleep_s = self.sleep_on_success_s if success_last else self.sleep_on_failure_s
                    self._stop_event.wait(sleep_s)

                except Exception as e:
                    self._log("Tick crashed: %s — retrying in %ds", e, self.sleep_on_failure_s)
                    self._stop_event.wait(self.sleep_on_failure_s)

        except KeyboardInterrupt:
            self._log("Interrupted by user (Ctrl+C)")
            self.state["status"] = "interrupted"
        finally:
            self.state["status"] = "stopped"
            self.state["current_task"] = None
            self._save_state()
            self._log("Autonomous daemon stopped")

    # --- Bounded mode (for tests / one-shot) --------------------------------

    def run(self, execute_fn: Callable[[dict], dict]) -> dict:
        """Run the autonomous loop for a bounded number of iterations.

        Returns a summary dict. Used by tests and one-shot execution.
        """
        self.state["status"] = "running"
        self.state["started_at"] = _now_iso()
        self._stop_event.clear()
        self._save_state()

        self._log("Autonomous loop started (max_iterations=%d, max_workers=%d)",
                  self.max_iterations, self.max_workers)

        try:
            while self.state["iteration"] < self.max_iterations and not self._stop_event.is_set():
                tick = self._tick(execute_fn)

                # If idle, exit (bounded mode)
                if tick["action"] == "idle":
                    self._log("No tasks remaining — exiting loop")
                    break

                # Check stop conditions
                if self.state["consecutive_failures"] >= self.max_consecutive_failures:
                    self._log("STOPPED: %d consecutive failures", self.state["consecutive_failures"])
                    break

                # Adaptive sleep (don't sleep after last iteration or if stopping)
                if (self.state["iteration"] < self.max_iterations
                        and not self._stop_event.is_set()):
                    success_last = tick.get("success", False)
                    sleep_s = self.sleep_on_success_s if success_last else self.sleep_on_failure_s
                    self._log("  Sleeping %ds...", sleep_s)
                    self._stop_event.wait(sleep_s)

        except KeyboardInterrupt:
            self._log("Interrupted by user (Ctrl+C)")
            self.state["status"] = "interrupted"
        finally:
            if self.state["status"] == "running":
                self.state["status"] = "completed"
            self.state["current_task"] = None
            self._save_state()

        summary = self._summary()
        self._log("Loop finished: %s", json.dumps(summary, ensure_ascii=False))
        return summary

    # --- Result processing --------------------------------------------------

    def _process_result(self, task: dict, result: dict) -> None:
        """Update state after task execution."""
        success = result.get("success", False)
        message = result.get("message", "")

        if success:
            task["status"] = "completed"
            self.state["completed_tasks"].append(task)
            self.state["consecutive_failures"] = 0
            self._log("  ✓ Success: %s", message[:100])
        else:
            task["status"] = "failed"
            task["error"] = message
            self.state["failed_tasks"].append(task)
            self.state["consecutive_failures"] += 1
            self._log("  ✗ Failed: %s", message[:100])

        # Learn
        self._learn(task, success, message)
        # Record in execution history
        self._record_history(task, result)

    def _record_history(self, task: dict, result: dict) -> None:
        """Append to execution_history for task generator feedback."""
        entry = {
            "timestamp": _now_iso(),
            "task_id": task["id"],
            "title": task["title"],
            "source": task["source"],
            "success": result.get("success", False),
            "message": result.get("message", "")[:200],
        }
        validation = result.get("validation")
        if validation:
            entry["validated"] = validation.get("verified", False)
            entry["resolved"] = validation.get("resolved", False)
        self.state.setdefault("execution_history", [])
        history = self.state["execution_history"]
        history.append(entry)
        # Keep bounded (last 50 entries)
        if len(history) > 50:
            self.state["execution_history"] = history[-50:]

    # --- Learning -----------------------------------------------------------

    def _learn(self, task: dict, success: bool, message: str) -> None:
        """Record task outcome for future learning."""
        entry = f"{_now_iso()} | {'✓' if success else '✗'} | {task['title']}"
        if not success:
            entry += f" | {message[:100]}"
        self.state.setdefault("learning_log", [])

        # Keep log bounded
        log = self.state["learning_log"]
        log.append(entry)
        if len(log) > 100:
            self.state["learning_log"] = log[-50:]

    # --- Logging ------------------------------------------------------------

    def _log(self, fmt: str, *args) -> None:
        line = fmt % args if args else fmt
        logger.info(line)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"[{_now_iso()}] {line}\n")
        except OSError:
            pass

    # --- Summary ------------------------------------------------------------

    def _summary(self) -> dict:
        return {
            "status": self.state["status"],
            "iterations": self.state["iteration"],
            "completed": len(self.state["completed_tasks"]),
            "failed": len(self.state["failed_tasks"]),
            "consecutive_failures": self.state["consecutive_failures"],
            "duration_s": self._duration_s(),
        }

    def _duration_s(self) -> int:
        try:
            start = datetime.fromisoformat(self.state["started_at"])
            now = datetime.now(timezone.utc)
            return int((now - start).total_seconds())
        except (ValueError, TypeError):
            return 0

    # --- External control ---------------------------------------------------
    def stop(self) -> None:
        """Request the loop to stop after the current iteration."""
        self.state["status"] = "stopping"
        self._stop_event.set()
        self._save_state()

    def is_running(self) -> bool:
        return self.state.get("status") == "running"

    @classmethod
    def load_status(cls, hermes_home: Optional[Path] = None) -> dict:
        """Return current state without starting the loop."""
        inst = cls(hermes_home=hermes_home)
        return inst.state
