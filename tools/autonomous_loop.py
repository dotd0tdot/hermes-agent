"""
Autonomous loop engine for Hermes Agent.

The ``AutonomousLoop`` class implements an in-process event loop that:

1. Scans the system for problems (logs, processes, git).
2. Reads an optional backlog file for user-defined tasks.
3. Picks the highest-priority task (NBA — Next Best Action).
4. Executes it via a caller-provided callback.
5. Persists state so it survives restarts.
6. Learns from outcomes (success/failure) via memory.

The loop is driven by the ``/auto`` slash command in ``cli.py``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
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
) -> dict:
    return {
        "id": id,
        "title": title,
        "priority": priority,
        "source": source,
        "attempts": attempts,
        "status": status,
        "error": error,
    }


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
                    capture_output=True, text=True, timeout=10,
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
                    capture_output=True, text=True, timeout=10,
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
                    capture_output=True, text=True, timeout=10,
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
                capture_output=True, text=True, timeout=10,
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
    """

    def __init__(
        self,
        hermes_home: Optional[Path] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
        sleep_on_success_s: int = DEFAULT_SLEEP_ON_SUCCESS_S,
        sleep_on_failure_s: int = DEFAULT_SLEEP_ON_FAILURE_S,
        scan_interval_s: int = DEFAULT_SCAN_INTERVAL_S,
        task_generator: Optional[Callable[[], List[dict]]] = None,
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
        self.task_generator = task_generator
        self.state = self._load_state()

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
        """Pick the highest-priority available task."""
        candidates: List[dict] = []

        # 1. System-detected problems (priority 10)
        if self.state["iteration"] % 3 == 0:  # Scan every 3rd iteration
            detected = self.scanner.scan()
            candidates.extend(detected)

        # 2. Backlog tasks (priority 5)
        candidates.extend(self.backlog.read())

        # 3. AI-generated tasks (priority 4) — ask agent what to do
        if not candidates and self.task_generator is not None:
            try:
                generated = self.task_generator()
                if generated:
                    candidates.extend(generated)
                    self._log("Generated %d tasks from agent", len(generated))
            except Exception as e:
                self._log("Task generator failed: %s", e)

        # 4. Fallback — nothing to do
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

    # --- Execution wrapper --------------------------------------------------

    def _execute_task(self, task: dict, execute_fn: Callable[[dict], dict]) -> dict:
        """Run a task via the callback, with error handling."""
        task["attempts"] = task.get("attempts", 0) + 1
        task["status"] = "running"
        self.state["current_task"] = task
        self._save_state()

        try:
            result = execute_fn(task)
            if not isinstance(result, dict):
                result = {"success": True, "message": str(result)}
            return result
        except Exception as e:
            logger.exception("Task execution failed: %s", task["id"])
            return {"success": False, "message": f"{type(e).__name__}: {e}"}

    # --- Main loop ----------------------------------------------------------

    def run(self, execute_fn: Callable[[dict], dict]) -> dict:
        """
        Run the autonomous loop.

        ``execute_fn(task) -> {"success": bool, "message": str}`` is called
        for each task. It has full access to Hermes tools (terminal, file, etc.)
        because it runs inside the agent's conversation loop.

        Returns a summary dict.
        """
        self.state["status"] = "running"
        self.state["started_at"] = _now_iso()
        self._save_state()

        self._log("Autonomous loop started (max_iterations=%d)", self.max_iterations)

        try:
            while self.state["iteration"] < self.max_iterations:
                self.state["iteration"] += 1
                self.state["last_iteration_at"] = _now_iso()

                task = self._select_next_task()
                if task is None:
                    self._log("No tasks remaining — exiting loop")
                    break

                self._log(
                    "Iteration %d/%d: [%s] %s (priority=%d, attempt=%d)",
                    self.state["iteration"],
                    self.max_iterations,
                    task["source"],
                    task["title"],
                    task["priority"],
                    task.get("attempts", 0) + 1,
                )

                # Execute
                result = self._execute_task(task, execute_fn)
                success = result.get("success", False)
                message = result.get("message", "")

                # Update state
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

                # Save after every iteration
                self._save_state()

                # Check stop conditions
                if self.state["consecutive_failures"] >= self.max_consecutive_failures:
                    self._log("STOPPED: %d consecutive failures", self.state["consecutive_failures"])
                    break

                # Adaptive sleep
                sleep_s = (
                    self.sleep_on_success_s if success else self.sleep_on_failure_s
                )
                # Don't sleep after the last iteration
                if self.state["iteration"] < self.max_iterations:
                    self._log("  Sleeping %ds...", sleep_s)
                    time.sleep(sleep_s)

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
        self._save_state()

    def is_running(self) -> bool:
        return self.state.get("status") == "running"

    @classmethod
    def load_status(cls, hermes_home: Optional[Path] = None) -> dict:
        """Return current state without starting the loop."""
        inst = cls(hermes_home=hermes_home)
        return inst.state
