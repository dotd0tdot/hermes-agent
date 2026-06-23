"""Tests for the autonomous loop engine."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.autonomous_loop import (
    AutonomousLoop,
    BacklogReader,
    SystemScanner,
    _make_task,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_hermes_home(tmp_path):
    """Create a temporary hermes home directory."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "logs").mkdir()
    return home


@pytest.fixture
def loop(tmp_hermes_home):
    """Create a fresh AutonomousLoop instance."""
    return AutonomousLoop(
        hermes_home=tmp_hermes_home,
        max_iterations=5,
        max_consecutive_failures=2,
        sleep_on_success_s=0,
        sleep_on_failure_s=0,
    )


# ---------------------------------------------------------------------------
# _make_task
# ---------------------------------------------------------------------------

class TestMakeTask:
    def test_defaults(self):
        t = _make_task("id1", "Test task", 10, "auto_detect")
        assert t["id"] == "id1"
        assert t["title"] == "Test task"
        assert t["priority"] == 10
        assert t["source"] == "auto_detect"
        assert t["attempts"] == 0
        assert t["status"] == "pending"

    def test_custom_fields(self):
        t = _make_task("id2", "Custom", 5, "backlog", attempts=2, status="running", error="oops")
        assert t["attempts"] == 2
        assert t["status"] == "running"
        assert t["error"] == "oops"


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class TestStateManagement:
    def test_load_default_state(self, loop):
        state = loop._load_state()
        assert state["status"] == "idle"
        assert state["iteration"] == 0
        assert state["max_iterations"] == 5

    def test_save_and_load_state(self, loop):
        loop.state["status"] = "running"
        loop.state["iteration"] = 3
        loop._save_state()

        # Reload
        loop2 = AutonomousLoop(hermes_home=loop.hermes_home, max_iterations=5)
        assert loop2.state["status"] == "running"
        assert loop2.state["iteration"] == 3

    def test_corrupt_state_falls_back_to_default(self, tmp_hermes_home):
        state_path = tmp_hermes_home / "autonomous-state.json"
        state_path.write_text("NOT JSON", encoding="utf-8")

        loop = AutonomousLoop(hermes_home=tmp_hermes_home, max_iterations=5)
        assert loop.state["status"] == "idle"


# ---------------------------------------------------------------------------
# SystemScanner
# ---------------------------------------------------------------------------

class TestSystemScanner:
    def test_no_problems_on_clean_system(self, loop):
        problems = loop.scanner.scan()
        # Should return a list (possibly empty)
        assert isinstance(problems, list)

    def test_detects_failed_systemd_service(self, tmp_hermes_home):
        scanner = SystemScanner(tmp_hermes_home)

        # Mock systemctl to return a failed service
        mock_result = subprocess.CompletedProcess(
            args=["systemctl", "--user", "list-units", "--state=failed", "--no-legend"],
            returncode=0,
            stdout="invidious.service\n",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_result):
            problems = scanner._scan_systemd_services()

        assert len(problems) == 1
        assert problems[0]["id"] == "systemd-failed-invidious.service"
        assert problems[0]["priority"] == 10

    def test_detects_git_uncommitted(self, tmp_hermes_home):
        scanner = SystemScanner(tmp_hermes_home)

        # Create a fake git repo inside the hermes_home so it's in known roots
        repo = tmp_hermes_home / "testrepo"
        repo.mkdir()
        (repo / ".git").mkdir()

        mock_status = subprocess.CompletedProcess(
            args=["git", "status", "--porcelain"],
            returncode=0,
            stdout="M file1.txt\n?? file2.txt\n" * 6,  # 12 lines > threshold
            stderr="",
        )
        mock_log = subprocess.CompletedProcess(
            args=["git", "log", "--oneline", "@{upstream}..HEAD"],
            returncode=1,
            stdout="",
            stderr="fatal: no upstream",
        )

        def mock_run(cmd, **kwargs):
            if "status" in cmd:
                return mock_status
            return mock_log

        # Patch known_roots to only scan our test repo
        scanner.known_roots = [repo]

        with patch("subprocess.run", side_effect=mock_run):
            problems = scanner._scan_git_repos()

        # Should detect the dirty repo
        dirty_problems = [p for p in problems if "git-dirty" in p["id"]]
        assert len(dirty_problems) == 1
        assert dirty_problems[0]["priority"] == 5


# ---------------------------------------------------------------------------
# BacklogReader
# ---------------------------------------------------------------------------

class TestBacklogReader:
    def test_empty_backlog(self, tmp_hermes_home):
        reader = BacklogReader(tmp_hermes_home / "autonomous-backlog.md")
        assert reader.read() == []

    def test_reads_pending_tasks(self, tmp_hermes_home):
        backlog = tmp_hermes_home / "autonomous-backlog.md"
        backlog.write_text(
            "# Backlog\n\n"
            "## Active\n"
            "- [ ] Fix the thing\n"
            "- [ ] Update the other\n\n"
            "## Done\n"
            "- [x] Already done\n",
            encoding="utf-8",
        )
        reader = BacklogReader(backlog)
        tasks = reader.read()
        assert len(tasks) == 2
        assert tasks[0]["title"] == "Fix the thing"
        assert tasks[0]["priority"] == 5
        assert tasks[0]["source"] == "backlog"

    def test_handles_missing_file(self, tmp_hermes_home):
        reader = BacklogReader(tmp_hermes_home / "nonexistent.md")
        assert reader.read() == []


# ---------------------------------------------------------------------------
# Task selection (NBA)
# ---------------------------------------------------------------------------

class TestTaskSelection:
    def test_selects_highest_priority_first(self, loop):
        # Inject mock tasks by patching scanner and backlog
        loop.scanner.scan = lambda: [
            _make_task("log-errors", "Fix log errors", 10, "auto_detect"),
        ]
        loop.backlog.read = lambda: [
            _make_task("backlog-1", "Update docs", 5, "source"),
        ]

        task = loop._select_next_task()
        assert task is not None
        assert task["priority"] == 10

    def test_skips_completed_tasks(self, loop):
        loop.state["completed_tasks"] = [
            _make_task("done-1", "Already done", 10, "auto_detect"),
        ]
        loop.scanner.scan = lambda: [
            _make_task("done-1", "Already done", 10, "auto_detect"),
        ]
        loop.backlog.read = lambda: []

        task = loop._select_next_task()
        # No tasks available — all completed
        assert task is None

    def test_skips_blocked_tasks(self, loop):
        loop.state["failed_tasks"] = [
            _make_task("blocked-1", "Always fails", 10, "auto_detect", attempts=3, status="failed"),
        ]
        loop.scanner.scan = lambda: [
            _make_task("blocked-1", "Always fails", 10, "auto_detect", attempts=3),
        ]
        loop.backlog.read = lambda: []

        task = loop._select_next_task()
        assert task is None

    def test_falls_back_to_task_generator(self, loop):
        loop.scanner.scan = lambda: []
        loop.backlog.read = lambda: []

        # With no task_generator, returns None
        task = loop._select_next_task()
        assert task is None

    def test_uses_task_generator(self, loop):
        loop.scanner.scan = lambda: []
        loop.backlog.read = lambda: []
        loop.task_generator = lambda: [
            _make_task("gen-1", "AI task", 6, "ai_generated"),
        ]

        task = loop._select_next_task()
        assert task is not None
        assert task["source"] == "ai_generated"
        assert task["title"] == "AI task"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class TestMainLoop:
    def test_loop_completes_with_no_tasks(self, loop):
        loop.scanner.scan = lambda: []
        loop.backlog.read = lambda: []

        summary = loop.run(lambda t: {"success": True, "message": "ok"})
        assert summary["status"] == "completed"
        # No tasks → loop exits immediately
        assert summary["completed"] == 0
        assert summary["iterations"] == 1

    def test_loop_executes_tasks(self, loop):
        loop.scanner.scan = lambda: [
            _make_task("task-1", "Do something", 10, "auto_detect"),
        ]
        loop.backlog.read = lambda: []

        executed = []

        def mock_execute(t):
            executed.append(t)
            return {"success": True, "message": "done"}

        # Force iteration to 3 so scanner runs on first call
        loop.state["iteration"] = 2
        summary = loop.run(mock_execute)
        assert len(executed) == 1
        assert summary["completed"] == 1

    def test_loop_stops_on_consecutive_failures(self, loop):
        loop.scanner.scan = lambda: [
            _make_task("fail-task", "Always fails", 10, "auto_detect"),
        ]
        loop.backlog.read = lambda: []

        def mock_execute(t):
            return {"success": False, "message": "nope"}

        summary = loop.run(mock_execute)
        # Should stop after max_consecutive_failures (2)
        assert summary["failed"] <= 2

    def test_loop_persists_state(self, loop):
        loop.scanner.scan = lambda: []
        loop.backlog.read = lambda: []

        loop.run(lambda t: {"success": True, "message": "ok"})

        state_path = loop.hermes_home / "autonomous-state.json"
        assert state_path.exists()
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        assert loaded["status"] == "completed"

    def test_loop_handles_execute_exception(self, loop):
        loop.scanner.scan = lambda: [
            _make_task("bad-task", "Will crash", 10, "auto_detect"),
        ]
        loop.backlog.read = lambda: []

        def mock_execute(t):
            raise RuntimeError("boom")

        # Force iteration to 3 so scanner runs on first call
        loop.state["iteration"] = 2
        summary = loop.run(mock_execute)
        assert summary["failed"] >= 1


# ---------------------------------------------------------------------------
# External control
# ---------------------------------------------------------------------------

class TestExternalControl:
    def test_stop_and_is_running(self, loop):
        assert not loop.is_running()

        loop.state["status"] = "running"
        assert loop.is_running()

        loop.stop()
        assert not loop.is_running()

    def test_load_status(self, tmp_hermes_home):
        # No state file yet
        state = AutonomousLoop.load_status(tmp_hermes_home)
        assert state["status"] == "idle"

        # Create a state file
        state_path = tmp_hermes_home / "autonomous-state.json"
        state_path.write_text(json.dumps({"status": "completed", "iteration": 5}), encoding="utf-8")

        state = AutonomousLoop.load_status(tmp_hermes_home)
        assert state["status"] == "completed"
        assert state["iteration"] == 5
