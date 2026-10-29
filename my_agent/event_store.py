from __future__ import annotations

import json
from pathlib import Path

from .database import Database
from .protocol import AgentEventType, event_from_dict, event_to_dict


class EventStore:
    """Typed durable events in the same database as session checkpoints."""

    def __init__(self, sessions_root: str | Path) -> None:
        self.database = Database(sessions_root)
        self.root = self.database.root

    def append(self, event: AgentEventType) -> None:
        with self.database.transaction() as connection:
            latest = self.latest_sequence(event.session_id)
            if event.sequence <= latest:
                raise ValueError("event sequence must increase")
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?)",
                (event.session_id, event.sequence, json.dumps(event_to_dict(event), ensure_ascii=False)),
            )

    def replay(self, session_id: str, *, after_sequence: int = 0) -> list[AgentEventType]:
        if after_sequence < 0:
            raise ValueError("after_sequence must not be negative")
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT data FROM events WHERE session_id=? AND sequence>? ORDER BY sequence",
                (session_id, after_sequence),
            ).fetchall()
        return [event_from_dict(json.loads(row[0])) for row in rows]

    def latest_sequence(self, session_id: str) -> int:
        with self.database.transaction() as connection:
            return connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM events WHERE session_id=?", (session_id,)
            ).fetchone()[0]

    def reserved_sequence(self, session_id: str) -> int:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT reserved_through FROM event_cursors WHERE session_id=?", (session_id,)
            ).fetchone()
            return max(row[0] if row else 0, self.latest_sequence(session_id))

    def reserve_sequences(self, session_id: str, sequence: int) -> int:
        reserved = ((sequence - 1) // 1024 + 1) * 1024
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO event_cursors VALUES (?, ?) ON CONFLICT(session_id) "
                "DO UPDATE SET reserved_through=MAX(reserved_through, excluded.reserved_through)",
                (session_id, reserved),
            )
        return reserved
