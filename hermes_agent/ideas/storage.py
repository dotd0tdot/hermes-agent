"""
SQLite storage for ideas and analysis history.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from hermes_agent.ideas.ideator import Idea

logger = logging.getLogger(__name__)

DB_PATH = Path.home() / ".hermes" / "ideas.db"


def _get_db() -> sqlite3.Connection:
    """Get or create the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ideas (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            category TEXT,
            priority INTEGER DEFAULT 3,
            complexity TEXT DEFAULT 'medium',
            target_files TEXT,  -- JSON array
            findings TEXT,      -- JSON array
            plan TEXT,          -- JSON array
            status TEXT DEFAULT 'new',
            created_at TEXT,
            applied_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            scope TEXT,
            file_count INTEGER,
            finding_count INTEGER,
            created_at TEXT
        )
    """)
    conn.commit()


class IdeaStorage:
    """Storage for ideas and scan results."""

    def save_idea(self, idea: Idea) -> None:
        """Save an idea to the database."""
        conn = _get_db()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO ideas
                   (id, project, title, description, category, priority,
                    complexity, target_files, findings, plan, status,
                    created_at, applied_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    idea.id,
                    idea.project,
                    idea.title,
                    idea.description,
                    idea.category,
                    idea.priority,
                    idea.complexity,
                    json.dumps(idea.target_files),
                    json.dumps(idea.findings),
                    json.dumps(idea.plan),
                    idea.status,
                    idea.created_at,
                    idea.applied_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_idea(self, idea_id: int, project: str = "hermes") -> Optional[dict]:
        """Get an idea by its display number (1-based)."""
        conn = _get_db()
        try:
            # idea_id is the display number, not the UUID
            rows = conn.execute(
                "SELECT * FROM ideas WHERE project = ? ORDER BY priority DESC, created_at ASC",
                (project,)
            ).fetchall()
            if 1 <= idea_id <= len(rows):
                row = rows[idea_id - 1]
                return self._row_to_dict(row)
            return None
        finally:
            conn.close()

    def list_ideas(self, project: str | None = None, status: str | None = None) -> list[dict]:
        """List ideas, optionally filtered."""
        conn = _get_db()
        try:
            query = "SELECT * FROM ideas WHERE 1=1"
            params = []
            if project:
                query += " AND project = ?"
                params.append(project)
            if status:
                query += " AND status = ?"
                params.append(status)
            query += " ORDER BY created_at DESC"

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def update_idea(self, idea_id: int, project: str = "hermes", **kwargs) -> bool:
        """Update an idea's fields."""
        conn = _get_db()
        try:
            # Find by display number
            rows = conn.execute(
                "SELECT id FROM ideas WHERE project = ? ORDER BY priority DESC, created_at ASC",
                (project,)
            ).fetchall()
            if 1 <= idea_id <= len(rows):
                real_id = rows[idea_id - 1]["id"]
                sets = ", ".join(f"{k} = ?" for k in kwargs)
                values = list(kwargs.values()) + [real_id]
                conn.execute(f"UPDATE ideas SET {sets} WHERE id = ?", values)
                conn.commit()
                return True
            return False
        finally:
            conn.close()

    def save_scan(self, project: str, scope: str, file_count: int, finding_count: int) -> None:
        """Record a scan run."""
        conn = _get_db()
        try:
            conn.execute(
                """INSERT INTO scans (project, scope, file_count, finding_count, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (project, scope, file_count, finding_count, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a database row to a dict."""
        d = dict(row)
        for key in ("target_files", "findings", "plan"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
