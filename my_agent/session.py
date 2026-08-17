from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .messages import AssistantMessage, Message, ToolCall, ToolResultMessage, UserMessage


@dataclass
class PendingTurn:
    turn_id: str
    task: str
    message_start: int
    step: int = 1
    tool_calls_executed: int = 0
    seen_calls: dict[str, int] = field(default_factory=dict)


@dataclass
class SessionState:
    session_id: str
    workspace: str
    session_dir: Path
    created_at: str
    updated_at: str
    messages: list[Message] = field(default_factory=list)
    context_summary: str = ""
    summarized_message_count: int = 0
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    provider_id: str | None = None
    model_id: str | None = None
    pending_turn: PendingTurn | None = None

    @property
    def metadata_path(self) -> Path:
        return self.session_dir / "session.json"

    @property
    def messages_path(self) -> Path:
        return self.session_dir / "messages.jsonl"

    @property
    def trace_path(self) -> Path:
        return self.session_dir / "trace.jsonl"

    @property
    def trajectory_path(self) -> Path:
        return self.session_dir / "trajectory.json"

    def abandon_pending_turn(self) -> PendingTurn:
        if self.pending_turn is None:
            raise ValueError("session has no pending turn")
        pending = self.pending_turn
        if pending.message_start < self.summarized_message_count:
            raise ValueError("pending turn overlaps compacted context")
        del self.messages[pending.message_start :]
        self.pending_turn = None
        return pending


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    workspace: str
    updated_at: str
    message_count: int


class SessionStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def create(self, workspace: str | Path) -> SessionState:
        self.root.mkdir(parents=True, exist_ok=True)
        while True:
            session_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
            session_dir = self.root / session_id
            try:
                session_dir.mkdir()
            except FileExistsError:
                continue
            break

        now = _now()
        state = SessionState(
            session_id=session_id,
            workspace=str(Path(workspace).expanduser().resolve()),
            session_dir=session_dir,
            created_at=now,
            updated_at=now,
        )
        self.save(state)
        return state

    def load(self, session_id: str) -> SessionState:
        session_dir = self._session_dir(session_id)
        metadata_path = session_dir / "session.json"
        messages_path = session_dir / "messages.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"session not found: {session_id}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        messages: list[Message] = []
        if messages_path.exists():
            for line_number, line in enumerate(
                messages_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    messages.append(message_from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid message in {messages_path} line {line_number}: {exc}"
                    ) from exc

        return SessionState(
            session_id=metadata["session_id"],
            workspace=metadata["workspace"],
            session_dir=session_dir,
            created_at=metadata["created_at"],
            updated_at=metadata["updated_at"],
            messages=messages,
            context_summary=metadata.get("context_summary", ""),
            summarized_message_count=int(metadata.get("summarized_message_count", 0)),
            last_input_tokens=int(metadata.get("last_input_tokens", 0)),
            last_output_tokens=int(metadata.get("last_output_tokens", 0)),
            provider_id=metadata.get("provider_id"),
            model_id=metadata.get("model_id"),
            pending_turn=_pending_turn_from_dict(metadata.get("pending_turn")),
        )

    def save(self, state: SessionState) -> None:
        state.session_dir.mkdir(parents=True, exist_ok=True)
        state.updated_at = _now()
        metadata = {
            "session_id": state.session_id,
            "workspace": state.workspace,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "message_count": len(state.messages),
            "context_summary": state.context_summary,
            "summarized_message_count": state.summarized_message_count,
            "last_input_tokens": state.last_input_tokens,
            "last_output_tokens": state.last_output_tokens,
            "provider_id": state.provider_id,
            "model_id": state.model_id,
            "pending_turn": _pending_turn_to_dict(state.pending_turn),
        }
        state.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        state.messages_path.write_text(
            "".join(
                json.dumps(message_to_dict(message), ensure_ascii=False) + "\n"
                for message in state.messages
            ),
            encoding="utf-8",
        )

    def list_sessions(self) -> list[SessionSummary]:
        if not self.root.exists():
            return []
        sessions: list[SessionSummary] = []
        for metadata_path in self.root.glob("*/session.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            sessions.append(
                SessionSummary(
                    session_id=metadata["session_id"],
                    workspace=metadata["workspace"],
                    updated_at=metadata["updated_at"],
                    message_count=int(metadata.get("message_count", 0)),
                )
            )
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def _session_dir(self, session_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", session_id):
            raise ValueError(f"invalid session id: {session_id}")
        return self.root / session_id


def message_to_dict(message: Message) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": message.content,
            "reasoning_content": message.reasoning_content,
            "tool_calls": [call.to_dict() for call in message.tool_calls],
        }
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "content": message.content,
        }
    raise TypeError(f"unsupported message type: {type(message).__name__}")


def message_from_dict(data: dict[str, Any]) -> Message:
    role = data.get("role")
    if role == "user":
        return UserMessage(content=data.get("content", ""))
    if role == "assistant":
        return AssistantMessage(
            content=data.get("content", ""),
            reasoning_content=data.get("reasoning_content", ""),
            tool_calls=[
                ToolCall(
                    call_id=call.get("call_id", ""),
                    name=call.get("name", ""),
                    arguments=call.get("arguments", {}),
                )
                for call in data.get("tool_calls", [])
            ],
        )
    if role == "tool":
        return ToolResultMessage(
            tool_call_id=data.get("tool_call_id", ""),
            tool_name=data.get("tool_name", ""),
            content=data.get("content", ""),
        )
    raise ValueError(f"unknown message role: {role!r}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pending_turn_to_dict(pending: PendingTurn | None) -> dict[str, Any] | None:
    if pending is None:
        return None
    return {
        "turn_id": pending.turn_id,
        "task": pending.task,
        "message_start": pending.message_start,
        "step": pending.step,
        "tool_calls_executed": pending.tool_calls_executed,
        "seen_calls": pending.seen_calls,
    }


def _pending_turn_from_dict(data: Any) -> PendingTurn | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("pending_turn must be an object")
    return PendingTurn(
        turn_id=str(data["turn_id"]),
        task=str(data["task"]),
        message_start=int(data["message_start"]),
        step=int(data.get("step", 1)),
        tool_calls_executed=int(data.get("tool_calls_executed", 0)),
        seen_calls={str(key): int(value) for key, value in data.get("seen_calls", {}).items()},
    )
