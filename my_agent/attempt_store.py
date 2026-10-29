from __future__ import annotations

import json
from typing import Any

from .database import Database


class AttemptStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    def start(self, session_id: str, turn_id: str, data: dict[str, Any]) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO tool_attempts VALUES (?, ?, ?, ?, ?, ?, 'started', NULL)",
                (data["attempt_id"], session_id, turn_id, data["call_id"], data["name"],
                 json.dumps(data["arguments"], ensure_ascii=False)),
            )

    def finish(self, session_id: str, turn_id: str, data: dict[str, Any]) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO tool_attempts VALUES (?, ?, ?, ?, ?, ?, 'completed', ?) "
                "ON CONFLICT(attempt_id) DO UPDATE SET status='completed', result=excluded.result",
                (data["attempt_id"], session_id, turn_id, data["call_id"], data["name"],
                 json.dumps(data.get("arguments", {}), ensure_ascii=False),
                 json.dumps(data["result"], ensure_ascii=False)),
            )

    def unresolved(self, session_id: str, turn_id: str) -> list[dict[str, Any]]:
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT attempt_id, call_id, tool_name, arguments FROM tool_attempts "
                "WHERE session_id=? AND turn_id=? AND status='started' ORDER BY rowid",
                (session_id, turn_id),
            ).fetchall()
        return [{"attempt_id": row[0], "call_id": row[1], "tool_name": row[2],
                 "arguments": json.loads(row[3]), "status": "unknown"} for row in rows]
