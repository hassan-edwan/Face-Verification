"""
FaceDatabase — Non-blocking SQLite persistence for identity metadata.

Design:
  - Single writer thread drains a queue of (sql, params) tuples.
  - All reads come from an in-memory cache (dict lookup, no SQL on read path).
  - Cache is warmed from SQLite on init; updated optimistically on each write call.
  - Worker thread calls create/update methods (fire-and-forget via queue).
  - Main thread calls get_identity() which reads only from cache.
"""

import os
import time
import queue
import sqlite3
import threading
from typing import Dict, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS identities (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    name                   TEXT NOT NULL UNIQUE,
    display_name           TEXT DEFAULT '',
    created_at             REAL NOT NULL,
    last_seen_at           REAL NOT NULL,
    total_matches          INTEGER DEFAULT 0,
    notes                  TEXT DEFAULT '',
    best_thumbnail_path    TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_name ON identities(name);
"""

# Columns added after v1 of the schema. Applied idempotently at init time
# so existing faces.db files pick them up without a manual migration step.
# Existing databases that picked up the deprecated `enrichment_saved_json`
# / `enrichment_saved_at` columns keep them dormant — SQLite has no ALTER
# TABLE DROP COLUMN before 3.35, and the columns are harmless.
_MIGRATIONS: list = []


class FaceDatabase:
    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # In-memory cache: name -> row dict (all reads come from here)
        self._cache: Dict[str, dict] = {}
        self._cache_lock = threading.Lock()

        # Async write queue: (sql, params) tuples processed by writer thread
        self._write_queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()

        # Initialize schema and warm cache
        self._init_schema()
        self._reload_cache()

        # Start writer thread
        self._writer = threading.Thread(
            target=self._writer_loop, daemon=True, name="DBWriter"
        )
        self._writer.start()

    # ── Public API (all non-blocking for caller) ──────────────────────────────

    def create_identity(self, name: str, thumbnail_path: str = "") -> None:
        """Insert a new identity. Called from worker thread on FAST_ENROLLED."""
        now = time.time()
        row = {
            "name": name,
            "display_name": "",
            "created_at": now,
            "last_seen_at": now,
            "total_matches": 0,
            "notes": "",
            "best_thumbnail_path": thumbnail_path,
        }
        # Update cache immediately (optimistic)
        with self._cache_lock:
            if name in self._cache:
                return  # already exists
            self._cache[name] = row
        # Enqueue write
        self._enqueue(
            "INSERT OR IGNORE INTO identities "
            "(name, display_name, created_at, last_seen_at, total_matches, notes, best_thumbnail_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, "", now, now, 0, "", thumbnail_path),
        )

    def update_match(self, name: str) -> None:
        """Increment match count and update last_seen. Called on MATCHED."""
        now = time.time()
        with self._cache_lock:
            entry = self._cache.get(name)
            if entry:
                entry["last_seen_at"] = now
                entry["total_matches"] = entry.get("total_matches", 0) + 1
        self._enqueue(
            "UPDATE identities SET last_seen_at = ?, total_matches = total_matches + 1 WHERE name = ?",
            (now, name),
        )

    def update_thumbnail(self, name: str, path: str) -> None:
        """Update best thumbnail path. Called on REFINED."""
        with self._cache_lock:
            entry = self._cache.get(name)
            if entry:
                entry["best_thumbnail_path"] = path
        self._enqueue(
            "UPDATE identities SET best_thumbnail_path = ? WHERE name = ?",
            (path, name),
        )

    def get_identity(self, name: str) -> Optional[dict]:
        """Read from cache only — never touches SQLite. Safe from any thread."""
        with self._cache_lock:
            entry = self._cache.get(name)
            return dict(entry) if entry else None

    def update_display_name(self, name: str, display_name: str) -> None:
        """Set an editable alias for an identity."""
        with self._cache_lock:
            entry = self._cache.get(name)
            if entry:
                entry["display_name"] = display_name
        self._enqueue(
            "UPDATE identities SET display_name = ? WHERE name = ?",
            (display_name, name),
        )

    def update_notes(self, name: str, notes: str) -> None:
        """Set free-text notes for an identity."""
        with self._cache_lock:
            entry = self._cache.get(name)
            if entry:
                entry["notes"] = notes
        self._enqueue(
            "UPDATE identities SET notes = ? WHERE name = ?",
            (notes, name),
        )

    def list_all(self) -> list:
        """Return every identity currently cached, sorted by last_seen
        descending. Used by the dashboard's overview grid. All access is
        read-only and non-blocking."""
        with self._cache_lock:
            rows = [dict(v) for v in self._cache.values()]
        rows.sort(key=lambda r: r.get("last_seen_at", 0), reverse=True)
        return rows

    def delete_identity(self, name: str) -> bool:
        """Cascade-delete entry point for the identity row.

        Returns True if the row existed in the cache (the SQL DELETE is
        enqueued either way; the boolean is for UI feedback). The cache is
        evicted synchronously so subsequent get_identity() / list_all() calls
        no longer surface this person, even before the writer thread drains.
        """
        with self._cache_lock:
            existed = self._cache.pop(name, None) is not None
        self._enqueue("DELETE FROM identities WHERE name = ?", (name,))
        return existed

    # ── Internal ──────────────────────────────────────────────────────────────

    def _enqueue(self, sql: str, params: tuple) -> None:
        try:
            self._write_queue.put_nowait((sql, params))
        except queue.Full:
            pass  # drop write silently rather than block caller

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.executescript(_SCHEMA)
        # Lightweight additive migration: add any columns listed in
        # _MIGRATIONS that don't already exist. Safe to run every startup.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(identities)")}
        for col, decl in _MIGRATIONS:
            if col not in existing:
                conn.execute(f"ALTER TABLE identities ADD COLUMN {col} {decl}")
        conn.commit()
        conn.close()

    def _reload_cache(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM identities").fetchall()
        conn.close()
        with self._cache_lock:
            self._cache.clear()
            for row in rows:
                self._cache[row["name"]] = dict(row)

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
                print(f"[DB] Write error: {exc}")
        conn.close()

    def close(self) -> None:
        self._stop.set()
        self._writer.join(timeout=3.0)
