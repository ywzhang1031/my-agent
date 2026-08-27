from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from .protocol import AgentEventType, event_from_dict, event_to_dict


class EventStore:
    """Append-only typed event storage scoped to a SessionStore root."""

    def __init__(self, sessions_root: str | Path) -> None:
        self.root = Path(sessions_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._latest: dict[str, int] = {}
        self._lock = threading.Lock()

    def append(self, event: AgentEventType) -> None:
        with self._lock:
            latest = self._latest_sequence_unlocked(event.session_id)
            if event.sequence <= latest:
                raise ValueError(
                    "event sequence must increase: "
                    f"session={event.session_id} latest={latest} got={event.sequence}"
                )
            path = self.path(event.session_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                json.dump(
                    event_to_dict(event),
                    stream,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                stream.write("\n")
                stream.flush()
            self._latest[event.session_id] = event.sequence

    def replay(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
    ) -> list[AgentEventType]:
        if after_sequence < 0:
            raise ValueError("after_sequence must not be negative")
        path = self.path(session_id)
        if not path.exists():
            return []
        events: list[AgentEventType] = []
        previous = 0
        with self._lock, path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    event = event_from_dict(json.loads(line))
                except (
                    AttributeError,
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    raise ValueError(
                        f"invalid event log {path}:{line_number}: {exc}"
                    ) from exc
                if event.session_id != session_id:
                    raise ValueError(
                        f"event log session mismatch: expected {session_id}, "
                        f"got {event.session_id}"
                    )
                if event.sequence <= previous:
                    raise ValueError(
                        f"event log sequence is not monotonic at {path}:{line_number}"
                    )
                previous = event.sequence
                if event.sequence > after_sequence:
                    events.append(event)
            self._latest[session_id] = previous
        return events

    def latest_sequence(self, session_id: str) -> int:
        with self._lock:
            return self._latest_sequence_unlocked(session_id)

    def path(self, session_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", session_id):
            raise ValueError(f"invalid session id: {session_id}")
        return self.root / session_id / "events.jsonl"

    def _latest_sequence_unlocked(self, session_id: str) -> int:
        cached = self._latest.get(session_id)
        if cached is not None:
            return cached
        path = self.path(session_id)
        if not path.exists():
            self._latest[session_id] = 0
            return 0
        latest = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    event = event_from_dict(json.loads(line))
                except (
                    AttributeError,
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    raise ValueError(
                        f"invalid event log {path}:{line_number}: {exc}"
                    ) from exc
                if event.session_id != session_id or event.sequence <= latest:
                    raise ValueError(f"invalid event ordering at {path}:{line_number}")
                latest = event.sequence
        self._latest[session_id] = latest
        return latest
