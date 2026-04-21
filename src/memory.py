"""
Memory System — Per-identity event store with session-based confidence gating.

Two components:
  1. MemoryStore  — SQLite-backed event log (async writes, cached reads).
  2. SessionManager — State machine that gates when events may be linked to a person.

Design principles:
  - Events are only attributed after 3s of continuous confident recognition.
  - The event schema is type-agnostic: transcripts, metadata, references all
    use the same table with event_type + source discriminators.
  - Extensible via metadata_json blob — no schema migrations for new fields.
"""

import os
import time
import uuid
import json
import queue
import sqlite3
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Confidence gating thresholds ─────────────────────────────────────────────
SESSION_GATE_SECONDS   = 3.0    # seconds of continuous recognition before ACTIVE
SESSION_PAUSE_TIMEOUT  = 10.0   # seconds before a paused session is closed
SESSION_MIN_CONFIDENCE = 0.70   # minimum score to start/maintain a session
# ─────────────────────────────────────────────────────────────────────────────

_MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_name   TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    content         TEXT NOT NULL,
    source          TEXT DEFAULT 'manual',
    confidence      REAL DEFAULT 0.0,
    created_at      REAL NOT NULL,
    metadata_json   TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memory_identity ON memory_events(identity_name);
CREATE INDEX IF NOT EXISTS idx_memory_session  ON memory_events(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_type     ON memory_events(event_type);
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  MEMORY STORE — async event persistence
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryStore:
    """Non-blocking SQLite store for per-identity memory events.
    Same async writer pattern as FaceDatabase."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._write_queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()

        # Recent events cache: identity_name -> list of recent event dicts
        self._recent_cache: Dict[str, List[dict]] = {}
        self._cache_lock = threading.Lock()
        self._recent_limit = 5  # keep last N events per identity in cache

        self._init_schema()
        self._reload_cache()

        self._writer = threading.Thread(
            target=self._writer_loop, daemon=True, name="MemoryWriter"
        )
        self._writer.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_event(
        self,
        identity_name: str,
        session_id:    str,
        event_type:    str,
        content:       str,
        source:        str = "manual",
        confidence:    float = 0.0,
        metadata:      Optional[dict] = None,
    ) -> None:
        """Store a memory event. Non-blocking (fire-and-forget)."""
        now = time.time()
        meta_json = json.dumps(metadata or {})
        row = {
            "identity_name": identity_name,
            "session_id": session_id,
            "event_type": event_type,
            "content": content,
            "source": source,
            "confidence": confidence,
            "created_at": now,
            "metadata_json": meta_json,
        }
        # Update cache immediately
        with self._cache_lock:
            events = self._recent_cache.setdefault(identity_name, [])
            events.insert(0, row)
            if len(events) > self._recent_limit:
                events.pop()
        # Enqueue write
        try:
            self._write_queue.put_nowait((
                "INSERT INTO memory_events "
                "(identity_name, session_id, event_type, content, source, confidence, created_at, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (identity_name, session_id, event_type, content, source, confidence, now, meta_json),
            ))
        except queue.Full:
            pass

    def get_recent_events(self, identity_name: str, limit: int = 5) -> List[dict]:
        """Read recent events from cache. Non-blocking."""
        with self._cache_lock:
            events = self._recent_cache.get(identity_name, [])
            return list(events[:limit])

    def get_event_count(self, identity_name: str) -> int:
        """Approximate count from cache."""
        with self._cache_lock:
            return len(self._recent_cache.get(identity_name, []))

    def delete_identity_events(self, identity_name: str) -> int:
        """Cascade-delete all events for an identity. Cache is wiped
        immediately; the SQL DELETE is enqueued for the writer thread.

        Returns the number of cached events removed (the on-disk count may
        be larger because the cache only tracks the most-recent N per
        identity)."""
        with self._cache_lock:
            removed = len(self._recent_cache.pop(identity_name, []))
        try:
            self._write_queue.put_nowait((
                "DELETE FROM memory_events WHERE identity_name = ?",
                (identity_name,),
            ))
        except queue.Full:
            # Same drop-on-overflow contract as add_event — caller should
            # not block on a saturated writer.
            pass
        return removed

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.executescript(_MEMORY_SCHEMA)
        conn.close()

    def _reload_cache(self) -> None:
        """Warm cache with the most recent events per identity."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM memory_events ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        with self._cache_lock:
            self._recent_cache.clear()
            for row in rows:
                name = row["identity_name"]
                events = self._recent_cache.setdefault(name, [])
                if len(events) < self._recent_limit:
                    events.append(dict(row))

    def _writer_loop(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        while not self._stop.is_set():
            try:
                sql, params = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                conn.execute(sql, params)
                conn.commit()
            except Exception as exc:
                print(f"[Memory] Write error: {exc}")
        conn.close()

    def close(self) -> None:
        self._stop.set()
        self._writer.join(timeout=3.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION MANAGER — confidence-gated identity sessions
# ═══════════════════════════════════════════════════════════════════════════════

class SessionState(Enum):
    IDLE    = auto()   # no recognized face
    GATING  = auto()   # recognized face, waiting 3s confirmation
    ACTIVE  = auto()   # session confirmed, events can be attributed
    PAUSED  = auto()   # face temporarily lost, buffering


@dataclass
class Session:
    session_id:     str
    identity_name:  str
    started_at:     float
    last_confirmed: float           # last frame where identity was confirmed
    state:          SessionState = SessionState.GATING
    gate_start:     float = 0.0     # when gating began
    pause_start:    float = 0.0     # when pause began
    event_count:    int   = 0


class SessionManager:
    """Manages identity sessions with confidence gating.

    Call tick() once per frame from the main loop with the primary face's
    DisplayState. The manager handles state transitions automatically.

    Events can only be added when a session is ACTIVE.
    """

    def __init__(self, memory_store: MemoryStore):
        self._memory = memory_store
        self._session: Optional[Session] = None
        self._state = SessionState.IDLE
        # Serializes tick() (main thread) against add_event() (speech thread).
        # Without this, the audio callback can TOCTOU-misattribute a finalized
        # phrase to a session that was just swapped out on an identity switch.
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def current_identity(self) -> Optional[str]:
        if self._session and self._state in (SessionState.ACTIVE, SessionState.PAUSED):
            return self._session.identity_name
        return None

    @property
    def current_session(self) -> Optional[Session]:
        return self._session if self._state != SessionState.IDLE else None

    def tick(
        self,
        identity: Optional[str],
        decision: Optional[str],
        score: float,
    ) -> None:
        """Called each frame from main loop. Drives session state transitions.

        Args:
            identity: The primary face's identity name, or None if no face.
            decision: The gatekeeper decision string ("MATCHED", etc.), or None.
            score:    The recognition confidence score.
        """
        with self._lock:
            self._tick_locked(identity, decision, score)

    def _tick_locked(
        self,
        identity: Optional[str],
        decision: Optional[str],
        score: float,
    ) -> None:
        now = time.time()
        is_confident = (
            identity is not None
            and decision in ("MATCHED", "REFINED")
            and score >= SESSION_MIN_CONFIDENCE
        )

        if self._state == SessionState.IDLE:
            if is_confident:
                self._state = SessionState.GATING
                self._session = Session(
                    session_id=str(uuid.uuid4())[:8],
                    identity_name=identity,
                    started_at=now,
                    last_confirmed=now,
                    state=SessionState.GATING,
                    gate_start=now,
                )

        elif self._state == SessionState.GATING:
            if not is_confident or identity != self._session.identity_name:
                # Lost confidence or identity changed during gating
                self._close_session()
            elif (now - self._session.gate_start) >= SESSION_GATE_SECONDS:
                # Gate passed — session is now active
                self._state = SessionState.ACTIVE
                self._session.state = SessionState.ACTIVE
                self._session.last_confirmed = now
                print(f"[Session] ACTIVE for '{identity}' (id={self._session.session_id})")
            else:
                self._session.last_confirmed = now

        elif self._state == SessionState.ACTIVE:
            if is_confident and identity == self._session.identity_name:
                self._session.last_confirmed = now
            elif is_confident and identity != self._session.identity_name:
                # Different identity appeared — close and start new gating
                print(f"[Session] Closed (identity switch) for '{self._session.identity_name}'")
                self._close_session()
                self._state = SessionState.GATING
                self._session = Session(
                    session_id=str(uuid.uuid4())[:8],
                    identity_name=identity,
                    started_at=now,
                    last_confirmed=now,
                    state=SessionState.GATING,
                    gate_start=now,
                )
            else:
                # Face lost or low confidence — pause
                self._state = SessionState.PAUSED
                self._session.state = SessionState.PAUSED
                self._session.pause_start = now

        elif self._state == SessionState.PAUSED:
            if is_confident and identity == self._session.identity_name:
                # Same identity returned — resume
                self._state = SessionState.ACTIVE
                self._session.state = SessionState.ACTIVE
                self._session.last_confirmed = now
            elif (now - self._session.pause_start) > SESSION_PAUSE_TIMEOUT:
                # Timeout — close session
                print(f"[Session] Closed (timeout) for '{self._session.identity_name}'")
                self._close_session()
            elif is_confident and identity != self._session.identity_name:
                # Different identity appeared — close and start new gating
                print(f"[Session] Closed (identity switch) for '{self._session.identity_name}'")
                self._close_session()
                # Immediately start gating for the new identity
                self._state = SessionState.GATING
                self._session = Session(
                    session_id=str(uuid.uuid4())[:8],
                    identity_name=identity,
                    started_at=now,
                    last_confirmed=now,
                    state=SessionState.GATING,
                    gate_start=now,
                )

    def add_event(
        self,
        event_type: str,
        content: str,
        source: str = "system",
        confidence: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Add any event to the active session. Returns False if not ACTIVE.

        Must be callable from background threads (the speech audio thread invokes
        this on phrase finalization). The lock snapshots `_state` and `_session`
        atomically so a concurrent identity switch cannot misattribute.
        """
        with self._lock:
            if self._state != SessionState.ACTIVE or self._session is None:
                return False
            session = self._session  # locked snapshot — guaranteed consistent
            self._memory.add_event(
                identity_name=session.identity_name,
                session_id=session.session_id,
                event_type=event_type,
                content=content,
                source=source,
                confidence=confidence,
                metadata=metadata,
            )
            session.event_count += 1
            return True

    def get_recent_events(self, limit: int = 5) -> List[dict]:
        """Get recent events for the current session's identity."""
        if self._session is None:
            return []
        return self._memory.get_recent_events(self._session.identity_name, limit)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _close_session(self) -> None:
        self._session = None
        self._state = SessionState.IDLE
