"""Persistent memory — SQLite-backed store for preferences, facts, and sessions.

Data is stored in data/memory.db (WAL mode for crash safety on Raspberry Pi).
Memory is injected into the system prompt so the LLM has context about the user.

On first run, automatically migrates data from the old data/memory.json if present.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/memory.db")
DEFAULT_JSON_PATH = Path("data/memory.json")
MAX_PROMPT_FACTS = 10
MAX_STORED_SESSIONS = 50

# ---------------------------------------------------------------------------
# Regex patterns for classifying remembered text as preference vs fact
# ---------------------------------------------------------------------------

# Name detection: "my name is Alex", "call me Alex"
_NAME_PATTERNS = [
    re.compile(r"\bmy name is (\w+)", re.I),
    re.compile(r"\bcall me (\w+)", re.I),
]

# Preference detection: (pattern, preference_key)
_PREFERENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Time format
    (re.compile(r"\b24[- ]?h(our)?\b.*\btime\b", re.I), "time_format"),
    (re.compile(r"\btime\b.*\b24[- ]?h(our)?\b", re.I), "time_format"),
    (re.compile(r"\b12[- ]?h(our)?\b.*\btime\b", re.I), "time_format"),
    (re.compile(r"\btime\b.*\b12[- ]?h(our)?\b", re.I), "time_format"),
    (re.compile(r"\bmilitary time\b", re.I), "time_format"),
    # Temperature units
    (re.compile(r"\b(celsius|centigrade)\b", re.I), "temperature_unit"),
    (re.compile(r"\bfahrenheit\b", re.I), "temperature_unit"),
    # Unit system
    (re.compile(r"\b(use|prefer|always)\b.*\b(metric|imperial)\b", re.I), "unit_system"),
    (re.compile(r"\b(use|prefer|always)\b.*\b(kilometers?|miles)\b", re.I), "unit_system"),
    # Generic "always/never/prefer" — catch-all for unrecognized preferences
    (re.compile(r"^(always|never|prefer)\b", re.I), "_generic_preference"),
]

# Lookup table: preference key → {value_substring → imperative instruction}
_PREFERENCE_INSTRUCTIONS: dict[str, dict[str, str]] = {
    "time_format": {
        "24": "ALWAYS use 24-hour time format (e.g., 14:30, not 2:30 PM)",
        "12": "ALWAYS use 12-hour AM/PM time format",
        "military": "ALWAYS use 24-hour time format (e.g., 14:30, not 2:30 PM)",
    },
    "temperature_unit": {
        "celsius": "ALWAYS show temperatures in Celsius",
        "centigrade": "ALWAYS show temperatures in Celsius",
        "fahrenheit": "ALWAYS show temperatures in Fahrenheit",
    },
    "unit_system": {
        "metric": "ALWAYS use metric units (km, kg, etc.)",
        "imperial": "ALWAYS use imperial units (miles, lbs, etc.)",
        "kilometer": "ALWAYS use metric units (km, kg, etc.)",
        "mile": "ALWAYS use imperial units (miles, lbs, etc.)",
    },
}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS preferences (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now','localtime'))
);

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'user_requested',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now','localtime'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    summary    TEXT,
    topics     TEXT,
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now','localtime')),
    ended_at   TEXT,
    status     TEXT NOT NULL DEFAULT 'open'
);
"""


class MemoryStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._connect()
        self._create_tables()
        self._maybe_migrate_json()

    # ------------------------------------------------------------------
    # Internal: connection and schema
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Migration from legacy JSON
    # ------------------------------------------------------------------
    def _maybe_migrate_json(self):
        """One-time import from data/memory.json if it exists and DB is empty."""
        json_path = self._db_path.parent / "memory.json"
        if not json_path.exists():
            return

        # Only migrate if DB tables are empty (idempotent)
        fact_count = self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        pref_count = self._conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
        if fact_count > 0 or pref_count > 0:
            return

        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Could not read memory.json for migration: {e}")
            return

        log.info("Migrating memory.json -> SQLite...")

        # Migrate user profile
        profile = data.get("user_profile", {})
        if profile.get("name"):
            self.set_preference("user_name", profile["name"])
        for key, val in profile.get("preferences", {}).items():
            self.set_preference(key, str(val))

        # Migrate facts
        for f in data.get("facts", []):
            self._conn.execute(
                "INSERT INTO facts (text, source, created_at) VALUES (?, ?, ?)",
                (f["text"], f.get("source", "migrated"), f.get("timestamp", datetime.now().isoformat())),
            )

        # Migrate session summaries
        for s in data.get("session_summaries", []):
            ts = s.get("timestamp", datetime.now().isoformat())
            self._conn.execute(
                "INSERT INTO sessions (summary, status, started_at, ended_at) VALUES (?, 'closed', ?, ?)",
                (s["summary"], ts, ts),
            )

        self._conn.commit()

        # Rename old file so migration won't run again
        backup = json_path.with_suffix(".json.bak")
        json_path.rename(backup)
        log.info(f"Migration complete. Old file renamed to {backup}")

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------
    def set_preference(self, key: str, value: str):
        """Store or update a user preference."""
        self._conn.execute(
            "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, datetime.now().isoformat()),
        )
        self._conn.commit()
        log.info(f"Memory: set preference {key}={value}")

    def get_preference(self, key: str) -> str | None:
        """Get a single preference value, or None if not set."""
        row = self._conn.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def get_all_preferences(self) -> dict[str, str]:
        """Get all stored preferences as a dict."""
        rows = self._conn.execute("SELECT key, value FROM preferences").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def set_user_name(self, name: str):
        """Convenience: store user name as a preference."""
        self.set_preference("user_name", name)

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------
    def add_fact(self, text: str, source: str = "user_requested") -> str:
        """Add a fact to long-term memory. Returns confirmation message."""
        self._conn.execute(
            "INSERT INTO facts (text, source) VALUES (?, ?)",
            (text, source),
        )
        self._conn.commit()
        log.info(f"Memory: stored fact '{text}'")
        return f"I'll remember that: {text}"

    def get_facts(self, limit: int = MAX_PROMPT_FACTS) -> list[dict]:
        """Get most recent facts (chronological order)."""
        rows = self._conn.execute(
            "SELECT text, source, created_at FROM facts ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def open_session(self) -> int:
        """Open a new session. Returns session ID."""
        cur = self._conn.execute("INSERT INTO sessions (status) VALUES ('open')")
        self._conn.commit()
        log.info(f"Opened session {cur.lastrowid}")
        return cur.lastrowid

    def close_session(
        self,
        session_id: int,
        summary: str | None = None,
        topics: str | None = None,
    ):
        """Close a session with optional summary and topics."""
        self._conn.execute(
            "UPDATE sessions SET status='closed', summary=?, topics=?, ended_at=? WHERE id=?",
            (summary, topics, datetime.now().isoformat(), session_id),
        )
        self._conn.commit()
        log.info(f"Session {session_id} closed. Summary: {summary}")

    def close_stale_sessions(self):
        """Close any sessions left open from a previous run (clean startup)."""
        rows = self._conn.execute(
            "SELECT id FROM sessions WHERE status = 'open'"
        ).fetchall()
        for row in rows:
            self.close_session(row["id"], summary=None)
            log.info(f"Closed stale session {row['id']} from previous run")

    def add_session_summary(self, summary: str):
        """Legacy compatibility: creates and closes a session with summary in one call."""
        sid = self.open_session()
        self.close_session(sid, summary=summary)

    def get_latest_summary(self) -> dict | None:
        """Get the most recent closed session with a summary."""
        row = self._conn.execute(
            "SELECT summary, topics, ended_at FROM sessions "
            "WHERE status='closed' AND summary IS NOT NULL "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def search_sessions(self, query: str, limit: int = 5) -> list[dict]:
        """Search session summaries and topics for keywords."""
        rows = self._conn.execute(
            "SELECT summary, topics, ended_at FROM sessions "
            "WHERE status='closed' AND summary IS NOT NULL "
            "AND (summary LIKE ? OR topics LIKE ?) "
            "ORDER BY id DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Recall tool — search past sessions
    # ------------------------------------------------------------------
    def recall(self, query: str) -> str:
        """Search past sessions for relevant summaries. Called by recall tool."""
        results = self.search_sessions(query, limit=5)
        if not results:
            return "I don't have any past conversations matching that query."

        lines = []
        for r in results:
            date = (r.get("ended_at") or "")[:10]
            summary = r.get("summary", "")
            topics = r.get("topics", "")
            entry = f"[{date}] {summary}"
            if topics:
                entry += f" (topics: {topics})"
            lines.append(entry)

        return "Past conversations:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # List memory — for the my_memory tool
    # ------------------------------------------------------------------
    def list_memory(self) -> str:
        """Format all stored preferences and facts for voice output.

        Called by the 'my_memory' tool when user asks "what do you know about me?"
        """
        parts: list[str] = []

        prefs = self.get_all_preferences()
        if prefs:
            parts.append("Your preferences:")
            for key, val in prefs.items():
                label = key.replace("_", " ")
                parts.append(f"- {label}: {val}")

        facts = self.get_facts(limit=20)
        if facts:
            parts.append("Things I remember about you:")
            for f in facts:
                parts.append(f"- {f['text']}")

        if not parts:
            return "I don't have any stored preferences or facts about you yet."

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Remember — unified entry point (classifies preference vs fact)
    # ------------------------------------------------------------------
    def remember(self, text: str) -> str:
        """Classify text as name, preference, or fact, then store accordingly.

        Called by the 'remember' tool.  Uses zero-cost regex patterns to detect
        preferences before falling back to generic fact storage.
        """
        # 1. Check for name
        for pattern in _NAME_PATTERNS:
            m = pattern.search(text)
            if m:
                name = m.group(1).strip().title()
                self.set_preference("user_name", name)
                return f"Got it, I'll remember your name is {name}."

        # 2. Check for preference
        for pattern, pref_key in _PREFERENCE_PATTERNS:
            if pattern.search(text):
                if pref_key == "_generic_preference":
                    # Store generic preferences with a hash-based key
                    pref_key = f"pref_{abs(hash(text.lower().strip())) % 10000}"
                self.set_preference(pref_key, text)
                return f"Preference saved: {text}"

        # 3. Default: store as a fact
        return self.add_fact(text, source="user_requested")

    # ------------------------------------------------------------------
    # System prompt injection
    # ------------------------------------------------------------------
    def build_prompt_section(self) -> str:
        """Build a compact memory section for the system prompt.

        Token budget target: ~150 tokens for 3B model context efficiency.
        - Preferences -> imperative instructions (highest priority)
        - Facts -> last 10
        - Session summary -> only the latest 1
        """
        parts: list[str] = []

        # 1. Preferences as imperative instructions
        prefs = self.get_all_preferences()
        user_name = prefs.pop("user_name", None)
        if user_name:
            parts.append(f"User's name: {user_name}")

        if prefs:
            parts.append("User preferences (ALWAYS follow these):")
            for key, val in prefs.items():
                instruction = self._preference_to_instruction(key, val)
                parts.append(f"- {instruction}")

        # 2. Known facts (compact)
        facts = self.get_facts()
        if facts:
            parts.append("Known facts:")
            for f in facts:
                parts.append(f"- {f['text']}")

        # 3. Latest session summary only
        latest = self.get_latest_summary()
        if latest and latest.get("summary"):
            parts.append(f"Last conversation: {latest['summary']}")

        if not parts:
            return ""
        return "\n\n## Memory\n" + "\n".join(parts)

    @staticmethod
    def _preference_to_instruction(key: str, value: str) -> str:
        """Convert a stored preference into an imperative system prompt instruction."""
        # Check lookup table for well-known preference keys
        if key in _PREFERENCE_INSTRUCTIONS:
            val_lower = value.lower()
            for sub_key, instruction in _PREFERENCE_INSTRUCTIONS[key].items():
                if sub_key in val_lower:
                    return instruction

        # For generic or unrecognized preferences, use the stored text directly
        if value.lower().startswith(("always", "never")):
            return value
        return f"ALWAYS: {value}"
