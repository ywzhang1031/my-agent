from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .database import Database, RevisionConflict
from .attempt_store import AttemptStore
from .goals import GoalState, goal_from_dict
from .messages import AssistantMessage, Message, ToolCall, ToolResultMessage, UserMessage
from .modes import BUILD_MODE
from .permissions import DEFAULT_PERMISSION_PROFILE_ID, get_permission_profile
from .sandbox import DEFAULT_SANDBOX_MODE, SANDBOX_MODES


SESSION_TITLE_MAX_LENGTH = 80


@dataclass
class PendingTurn:
    turn_id: str
    task: str
    message_start: int
    step: int = 1
    tool_calls_executed: int = 0
    seen_calls: dict[str, int] = field(default_factory=dict)
    mode_id: str = BUILD_MODE.mode_id
    approved_plan_id: str | None = None
    last_error: str | None = None
    last_error_retryable: bool | None = None
    last_error_details: dict[str, Any] = field(default_factory=dict)
    pause_reason: str | None = None
    pause_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposedPlan:
    plan_id: str
    source_turn_id: str
    content: str
    status: str = "proposed"


@dataclass
class SessionState:
    session_id: str
    workspace: str
    session_dir: Path
    created_at: str
    updated_at: str
    revision: int = 0
    turn_counter: int = 0
    checkpoint_status: str = "idle"
    title: str = ""
    messages: list[Message] = field(default_factory=list)
    context_summary: str = ""
    summarized_message_count: int = 0
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    provider_id: str | None = None
    model_id: str | None = None
    mode_id: str = BUILD_MODE.mode_id
    permission_profile_id: str = DEFAULT_PERMISSION_PROFILE_ID
    sandbox_mode: str = DEFAULT_SANDBOX_MODE
    goal: GoalState | None = None
    proposed_plan: ProposedPlan | None = None
    pending_turn: PendingTurn | None = None

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
    title: str
    workspace: str
    updated_at: str
    message_count: int


class SessionStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.database = Database(self.root)

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

    def load(self, session_id: str, *, recover: bool = True) -> SessionState:
        session_dir = self._session_dir(session_id)
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT revision, data FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"session not found: {session_id}")
        state = session_from_dict(json.loads(row[1]), session_dir, revision=row[0])
        if recover and state.checkpoint_status == "running" and state.pending_turn is not None:
            state.pending_turn.pause_reason = "recovery_required"
            state.pending_turn.last_error = "Previous process stopped during execution; inspect workspace before continuing."
            state.pending_turn.pause_details = {
                "unknown_attempts": AttemptStore(self.database).unresolved(session_id, state.pending_turn.turn_id)
            }
        return state

    def save(self, state: SessionState, *, touch_updated_at: bool = True) -> None:
        state.session_dir.mkdir(parents=True, exist_ok=True)
        updated_at = _now() if touch_updated_at else state.updated_at
        data = session_to_dict(state)
        data["updated_at"] = updated_at
        with self.database.transaction() as connection:
            if state.revision == 0:
                row = connection.execute(
                    "INSERT OR IGNORE INTO sessions VALUES (?, 1, ?)",
                    (state.session_id, json.dumps(data, ensure_ascii=False)),
                )
            else:
                row = connection.execute(
                    "UPDATE sessions SET revision=revision+1, data=? "
                    "WHERE session_id=? AND revision=?",
                    (json.dumps(data, ensure_ascii=False), state.session_id, state.revision),
                )
            if row.rowcount != 1:
                raise RevisionConflict(f"stale session revision: {state.session_id}")
            old_revision, old_updated_at = state.revision, state.updated_at
            def restore() -> None:
                state.revision, state.updated_at = old_revision, old_updated_at
            self.database.on_rollback(restore)
            state.revision += 1
            state.updated_at = updated_at

    def list_sessions(self) -> list[SessionSummary]:
        with self.database.transaction() as connection:
            rows = connection.execute("SELECT data FROM sessions").fetchall()
        sessions = []
        for row in rows:
            metadata = json.loads(row[0])
            sessions.append(SessionSummary(
                session_id=metadata["session_id"], title=metadata.get("title", ""),
                workspace=metadata["workspace"], updated_at=metadata["updated_at"],
                message_count=len(metadata["messages"]),
            ))
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def _session_dir(self, session_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", session_id):
            raise ValueError(f"invalid session id: {session_id}")
        return self.root / session_id


def session_from_dict(metadata: dict[str, Any], session_dir: Path, *, revision: int = 0) -> SessionState:
    messages = [message_from_dict(message) for message in metadata["messages"]]
    return SessionState(
        session_id=metadata["session_id"],
        workspace=metadata["workspace"],
        session_dir=session_dir,
        created_at=metadata["created_at"],
        updated_at=metadata["updated_at"],
        revision=revision,
        checkpoint_status=str(metadata.get("checkpoint_status", "idle")),
        turn_counter=int(metadata.get("turn_counter", sum(isinstance(m, UserMessage) for m in messages))),
        title=str(metadata.get("title", "")),
        messages=messages,
        context_summary=metadata.get("context_summary", ""),
        summarized_message_count=int(metadata.get("summarized_message_count", 0)),
        last_input_tokens=int(metadata.get("last_input_tokens", 0)),
        last_output_tokens=int(metadata.get("last_output_tokens", 0)),
        provider_id=metadata.get("provider_id"),
        model_id=metadata.get("model_id"),
        mode_id=str(metadata.get("mode_id", BUILD_MODE.mode_id)),
        permission_profile_id=get_permission_profile(
            str(
                metadata.get(
                    "permission_profile_id",
                    DEFAULT_PERMISSION_PROFILE_ID,
                )
            )
        ).profile_id,
        sandbox_mode=_sandbox_mode(metadata.get("sandbox_mode")),
        goal=goal_from_dict(metadata.get("goal")),
        proposed_plan=_proposed_plan_from_dict(metadata.get("proposed_plan")),
        pending_turn=_pending_turn_from_dict(metadata.get("pending_turn")),
    )


def session_to_dict(state: SessionState) -> dict[str, Any]:
    return {
        "session_id": state.session_id,
        "title": state.title,
        "workspace": state.workspace,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "message_count": len(state.messages),
        "messages": [message_to_dict(message) for message in state.messages],
        "turn_counter": state.turn_counter,
        "checkpoint_status": state.checkpoint_status,
        "context_summary": state.context_summary,
        "summarized_message_count": state.summarized_message_count,
        "last_input_tokens": state.last_input_tokens,
        "last_output_tokens": state.last_output_tokens,
        "provider_id": state.provider_id,
        "model_id": state.model_id,
        "mode_id": state.mode_id,
        "permission_profile_id": state.permission_profile_id,
        "sandbox_mode": state.sandbox_mode,
        "goal": state.goal.to_dict() if state.goal is not None else None,
        "proposed_plan": _proposed_plan_to_dict(state.proposed_plan),
        "pending_turn": _pending_turn_to_dict(state.pending_turn),
    }


def normalize_session_title(value: str) -> str:
    title = value.strip()
    if not title:
        raise ValueError("title must be a non-empty string")
    if len(title) > SESSION_TITLE_MAX_LENGTH:
        raise ValueError(
            f"title must be at most {SESSION_TITLE_MAX_LENGTH} characters"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in title):
        raise ValueError("title must not contain control characters")
    return title


def message_to_dict(message: Message) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": message.content,
            "reasoning_content": message.reasoning_content,
            "tool_calls": [call.to_dict() for call in message.tool_calls],
            "provider_data": message.provider_data,
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
            provider_data=data.get("provider_data", {}),
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


def _sandbox_mode(value: Any) -> str:
    selected = str(value or DEFAULT_SANDBOX_MODE)
    if selected not in SANDBOX_MODES:
        raise ValueError(f"unknown sandbox mode: {selected}")
    return selected


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
        "mode_id": pending.mode_id,
        "approved_plan_id": pending.approved_plan_id,
        "last_error": pending.last_error,
        "last_error_retryable": pending.last_error_retryable,
        "last_error_details": pending.last_error_details,
        "pause_reason": pending.pause_reason,
        "pause_details": pending.pause_details,
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
        mode_id=str(data.get("mode_id", BUILD_MODE.mode_id)),
        approved_plan_id=data.get("approved_plan_id"),
        last_error=data.get("last_error"),
        last_error_retryable=data.get("last_error_retryable"),
        last_error_details=dict(data.get("last_error_details", {})),
        pause_reason=data.get("pause_reason"),
        pause_details=dict(data.get("pause_details", {})),
    )


def _proposed_plan_to_dict(plan: ProposedPlan | None) -> dict[str, str] | None:
    if plan is None:
        return None
    return {
        "plan_id": plan.plan_id,
        "source_turn_id": plan.source_turn_id,
        "content": plan.content,
        "status": plan.status,
    }


def _proposed_plan_from_dict(data: Any) -> ProposedPlan | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("proposed_plan must be an object")
    status = str(data.get("status", "proposed"))
    if status not in {"proposed", "revising", "approved", "completed"}:
        raise ValueError(f"invalid proposed_plan status: {status}")
    return ProposedPlan(
        plan_id=str(data["plan_id"]),
        source_turn_id=str(data["source_turn_id"]),
        content=str(data["content"]),
        status=status,
    )
