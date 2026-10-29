from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


DATABASE_NAME = "agent.sqlite3"
SCHEMA_VERSION = 4
_transactions = threading.local()


class StorageError(RuntimeError):
    pass


class RevisionConflict(StorageError):
    pass


class Database:
    """Short synchronous transactions shared by stores on the same thread/root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / DATABASE_NAME
        if not self.path.exists() and any(self.root.glob("*/session.json")):
            raise StorageError(
                f"legacy sessions require migration: run cli.py migrate --sessions-dir {self.root}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        with self.transaction() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, 3, SCHEMA_VERSION):
                raise StorageError(f"unsupported database schema: {version}")
            connection.execute("""CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, data TEXT NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS events (
                session_id TEXT NOT NULL, sequence INTEGER NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY (session_id, sequence)
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS event_cursors (
                session_id TEXT PRIMARY KEY, reserved_through INTEGER NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS tool_attempts (
                attempt_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                call_id TEXT NOT NULL, tool_name TEXT NOT NULL, arguments TEXT NOT NULL,
                status TEXT NOT NULL, result TEXT
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS rpc_requests (
                request_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, response TEXT
            )""")
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        os.chmod(self.path, 0o600)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        active = getattr(_transactions, "active", None)
        if active is None:
            active = _transactions.active = {}
        key = str(self.path)
        if key in active:
            yield active[key][0]
            return
        connection = sqlite3.connect(self.path, isolation_level=None, timeout=5)
        callbacks: list[Callable[[], None]] = []
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            active[key] = (connection, callbacks)
            yield connection
            connection.execute("COMMIT")
        except BaseException as exc:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            for callback in reversed(callbacks):
                callback()
            if isinstance(exc, sqlite3.Error):
                raise StorageError(f"database operation failed: {exc}") from exc
            raise
        finally:
            active.pop(key, None)
            connection.close()

    def on_rollback(self, callback: Callable[[], None]) -> None:
        _transactions.active[str(self.path)][1].append(callback)
