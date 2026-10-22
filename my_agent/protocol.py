from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, ClassVar, Mapping, TypeAlias
from uuid import uuid4

from .control import ApprovalDecision


PROTOCOL_VERSION = 2


class AgentState(str, Enum):
    IDLE = "idle"
    RUNNING_MODEL = "running_model"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_USER = "waiting_user"
    RUNNING_TOOL = "running_tool"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class CommandType(str, Enum):
    RENAME_SESSION = "session.rename"
    START_TURN = "turn.start"
    INTERRUPT_TURN = "turn.interrupt"
    RESOLVE_APPROVAL = "approval.resolve"
    SWITCH_MODEL = "model.switch"
    SWITCH_MODE = "mode.switch"
    SWITCH_PERMISSION_PROFILE = "permission.switch"
    EXECUTE_PLAN = "plan.execute"
    REVISE_PLAN = "plan.revise"
    ABORT_TURN = "turn.abort"


class EventType(str, Enum):
    SESSION_RENAMED = "session.renamed"
    TURN_STARTED = "turn.started"
    TURN_RESUMED = "turn.resumed"
    CONTEXT_COMPACTED = "context.compacted"
    MODEL_STARTED = "model.started"
    PROVIDER_RETRY = "provider.retry"
    ITEM_STARTED = "item.started"
    ITEM_DELTA = "item.delta"
    ITEM_COMPLETED = "item.completed"
    ITEM_FAILED = "item.failed"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_RESOLVED = "approval.resolved"
    TURN_ERROR = "turn.error"
    TURN_CONTINUATION_REQUIRED = "turn.continuation_required"
    TURN_COMPLETED = "turn.completed"
    TURN_FAILED = "turn.failed"
    TURN_INTERRUPTED = "turn.interrupted"
    MODEL_CHANGED = "model.changed"
    MODE_CHANGED = "mode.changed"
    PERMISSION_PROFILE_CHANGED = "permission.changed"
    PLAN_APPROVED = "plan.approved"
    PLAN_REVISION_REQUESTED = "plan.revision_requested"
    TURN_ABORTED = "turn.aborted"


@dataclass(frozen=True, kw_only=True)
class AgentCommand:
    session_id: str
    command_id: str = field(default_factory=lambda: uuid4().hex)
    command_type: ClassVar[CommandType]


@dataclass(frozen=True, kw_only=True)
class RenameSessionCommand(AgentCommand):
    title: str
    command_type: ClassVar[CommandType] = CommandType.RENAME_SESSION


@dataclass(frozen=True, kw_only=True)
class StartTurnCommand(AgentCommand):
    user_input: str = ""
    retry: bool = False
    approved_plan_id: str | None = None
    command_type: ClassVar[CommandType] = CommandType.START_TURN


@dataclass(frozen=True, kw_only=True)
class InterruptTurnCommand(AgentCommand):
    command_type: ClassVar[CommandType] = CommandType.INTERRUPT_TURN


@dataclass(frozen=True, kw_only=True)
class ResolveApprovalCommand(AgentCommand):
    request_id: str
    decision: ApprovalDecision
    command_type: ClassVar[CommandType] = CommandType.RESOLVE_APPROVAL


@dataclass(frozen=True, kw_only=True)
class SwitchModelCommand(AgentCommand):
    model_ref: str
    command_type: ClassVar[CommandType] = CommandType.SWITCH_MODEL


@dataclass(frozen=True, kw_only=True)
class SwitchModeCommand(AgentCommand):
    mode_id: str
    command_type: ClassVar[CommandType] = CommandType.SWITCH_MODE


@dataclass(frozen=True, kw_only=True)
class SwitchPermissionProfileCommand(AgentCommand):
    profile_id: str
    command_type: ClassVar[CommandType] = CommandType.SWITCH_PERMISSION_PROFILE


@dataclass(frozen=True, kw_only=True)
class ExecutePlanCommand(AgentCommand):
    command_type: ClassVar[CommandType] = CommandType.EXECUTE_PLAN


@dataclass(frozen=True, kw_only=True)
class RevisePlanCommand(AgentCommand):
    command_type: ClassVar[CommandType] = CommandType.REVISE_PLAN


@dataclass(frozen=True, kw_only=True)
class AbortTurnCommand(AgentCommand):
    command_type: ClassVar[CommandType] = CommandType.ABORT_TURN


AgentCommandType: TypeAlias = (
    RenameSessionCommand
    | StartTurnCommand
    | InterruptTurnCommand
    | ResolveApprovalCommand
    | SwitchModelCommand
    | SwitchModeCommand
    | SwitchPermissionProfileCommand
    | ExecutePlanCommand
    | RevisePlanCommand
    | AbortTurnCommand
)


@dataclass(frozen=True, kw_only=True)
class AgentEvent:
    session_id: str
    turn_id: str | None
    sequence: int
    state: AgentState
    durable: bool = True
    event_id: str = field(default_factory=lambda: uuid4().hex)
    event_type: ClassVar[EventType]


@dataclass(frozen=True, kw_only=True)
class SessionRenamedEvent(AgentEvent):
    title: str
    event_type: ClassVar[EventType] = EventType.SESSION_RENAMED


@dataclass(frozen=True, kw_only=True)
class TurnStartedEvent(AgentEvent):
    task: str
    event_type: ClassVar[EventType] = EventType.TURN_STARTED


@dataclass(frozen=True, kw_only=True)
class TurnResumedEvent(AgentEvent):
    step: int
    event_type: ClassVar[EventType] = EventType.TURN_RESUMED


@dataclass(frozen=True, kw_only=True)
class ContextCompactedEvent(AgentEvent):
    step: int
    snapshot: dict[str, Any]
    event_type: ClassVar[EventType] = EventType.CONTEXT_COMPACTED


@dataclass(frozen=True, kw_only=True)
class ModelStartedEvent(AgentEvent):
    step: int
    event_type: ClassVar[EventType] = EventType.MODEL_STARTED


@dataclass(frozen=True, kw_only=True)
class ItemStartedEvent(AgentEvent):
    item_id: str
    item_type: str
    step: int
    data: dict[str, Any] = field(default_factory=dict)
    event_type: ClassVar[EventType] = EventType.ITEM_STARTED


@dataclass(frozen=True, kw_only=True)
class ItemDeltaEvent(AgentEvent):
    item_id: str
    item_type: str
    step: int
    delta: str
    data: dict[str, Any] = field(default_factory=dict)
    event_type: ClassVar[EventType] = EventType.ITEM_DELTA


@dataclass(frozen=True, kw_only=True)
class ItemCompletedEvent(AgentEvent):
    item_id: str
    item_type: str
    step: int
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    event_type: ClassVar[EventType] = EventType.ITEM_COMPLETED


@dataclass(frozen=True, kw_only=True)
class ItemFailedEvent(AgentEvent):
    item_id: str
    item_type: str
    step: int
    status: str
    error: str
    data: dict[str, Any] = field(default_factory=dict)
    event_type: ClassVar[EventType] = EventType.ITEM_FAILED


@dataclass(frozen=True, kw_only=True)
class ProviderRetryEvent(AgentEvent):
    step: int
    error: str
    attempt: int
    delay_seconds: float
    details: dict[str, Any]
    event_type: ClassVar[EventType] = EventType.PROVIDER_RETRY


@dataclass(frozen=True, kw_only=True)
class ApprovalRequestedEvent(AgentEvent):
    step: int
    call_id: str
    tool_name: str
    request_id: str
    action: str
    resource: str
    description: str
    details: dict[str, Any]
    event_type: ClassVar[EventType] = EventType.APPROVAL_REQUESTED


@dataclass(frozen=True, kw_only=True)
class ApprovalResolvedEvent(AgentEvent):
    step: int
    call_id: str
    tool_name: str
    request_id: str
    decision: str
    prompted: bool
    source: str
    event_type: ClassVar[EventType] = EventType.APPROVAL_RESOLVED


@dataclass(frozen=True, kw_only=True)
class TurnErrorEvent(AgentEvent):
    step: int
    error: str
    retryable: bool
    details: dict[str, Any] = field(default_factory=dict)
    event_type: ClassVar[EventType] = EventType.TURN_ERROR


@dataclass(frozen=True, kw_only=True)
class TurnContinuationRequiredEvent(AgentEvent):
    steps_completed: int
    step_budget: int
    next_step: int
    event_type: ClassVar[EventType] = EventType.TURN_CONTINUATION_REQUIRED


@dataclass(frozen=True, kw_only=True)
class TurnCompletedEvent(AgentEvent):
    answer: str
    steps: int
    stopped_by: str
    mode_id: str
    plan_id: str | None = None
    event_type: ClassVar[EventType] = EventType.TURN_COMPLETED


@dataclass(frozen=True, kw_only=True)
class TurnFailedEvent(AgentEvent):
    error: str
    event_type: ClassVar[EventType] = EventType.TURN_FAILED


@dataclass(frozen=True, kw_only=True)
class TurnInterruptedEvent(AgentEvent):
    reason: str
    event_type: ClassVar[EventType] = EventType.TURN_INTERRUPTED


@dataclass(frozen=True, kw_only=True)
class ModelChangedEvent(AgentEvent):
    provider_id: str
    model_id: str
    event_type: ClassVar[EventType] = EventType.MODEL_CHANGED


@dataclass(frozen=True, kw_only=True)
class ModeChangedEvent(AgentEvent):
    mode_id: str
    event_type: ClassVar[EventType] = EventType.MODE_CHANGED


@dataclass(frozen=True, kw_only=True)
class PermissionProfileChangedEvent(AgentEvent):
    profile_id: str
    event_type: ClassVar[EventType] = EventType.PERMISSION_PROFILE_CHANGED


@dataclass(frozen=True, kw_only=True)
class PlanApprovedEvent(AgentEvent):
    plan_id: str
    mode_id: str
    event_type: ClassVar[EventType] = EventType.PLAN_APPROVED


@dataclass(frozen=True, kw_only=True)
class PlanRevisionRequestedEvent(AgentEvent):
    plan_id: str
    event_type: ClassVar[EventType] = EventType.PLAN_REVISION_REQUESTED


@dataclass(frozen=True, kw_only=True)
class TurnAbortedEvent(AgentEvent):
    tool_calls_executed: int
    event_type: ClassVar[EventType] = EventType.TURN_ABORTED


AgentEventType: TypeAlias = (
    SessionRenamedEvent
    | TurnStartedEvent
    | TurnResumedEvent
    | ContextCompactedEvent
    | ModelStartedEvent
    | ItemStartedEvent
    | ItemDeltaEvent
    | ItemCompletedEvent
    | ItemFailedEvent
    | ProviderRetryEvent
    | ApprovalRequestedEvent
    | ApprovalResolvedEvent
    | TurnErrorEvent
    | TurnContinuationRequiredEvent
    | TurnCompletedEvent
    | TurnFailedEvent
    | TurnInterruptedEvent
    | ModelChangedEvent
    | ModeChangedEvent
    | PermissionProfileChangedEvent
    | PlanApprovedEvent
    | PlanRevisionRequestedEvent
    | TurnAbortedEvent
)


_COMMAND_TYPES = {
    command_type.command_type.value: command_type
    for command_type in (
        RenameSessionCommand,
        StartTurnCommand,
        InterruptTurnCommand,
        ResolveApprovalCommand,
        SwitchModelCommand,
        SwitchModeCommand,
        SwitchPermissionProfileCommand,
        ExecutePlanCommand,
        RevisePlanCommand,
        AbortTurnCommand,
    )
}

_EVENT_TYPES = {
    event_type.event_type.value: event_type
    for event_type in (
        SessionRenamedEvent,
        TurnStartedEvent,
        TurnResumedEvent,
        ContextCompactedEvent,
        ModelStartedEvent,
        ItemStartedEvent,
        ItemDeltaEvent,
        ItemCompletedEvent,
        ItemFailedEvent,
        ProviderRetryEvent,
        ApprovalRequestedEvent,
        ApprovalResolvedEvent,
        TurnErrorEvent,
        TurnContinuationRequiredEvent,
        TurnCompletedEvent,
        TurnFailedEvent,
        TurnInterruptedEvent,
        ModelChangedEvent,
        ModeChangedEvent,
        PermissionProfileChangedEvent,
        PlanApprovedEvent,
        PlanRevisionRequestedEvent,
        TurnAbortedEvent,
    )
}


def command_to_dict(command: AgentCommandType) -> dict[str, Any]:
    payload = asdict(command)
    session_id = payload.pop("session_id")
    command_id = payload.pop("command_id")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": command.command_type.value,
        "command_id": command_id,
        "session_id": session_id,
        "payload": _json_value(payload),
    }


def command_from_dict(data: Mapping[str, Any]) -> AgentCommandType:
    _require_version(data)
    command_type = _COMMAND_TYPES.get(str(data.get("type", "")))
    if command_type is None:
        raise ValueError(f"unknown command type: {data.get('type')}")
    payload = _payload(data)
    if command_type is ResolveApprovalCommand:
        payload["decision"] = ApprovalDecision(payload["decision"])
    return command_type(
        session_id=str(data["session_id"]),
        command_id=str(data["command_id"]),
        **payload,
    )


def event_to_dict(event: AgentEventType) -> dict[str, Any]:
    payload = asdict(event)
    session_id = payload.pop("session_id")
    turn_id = payload.pop("turn_id")
    sequence = payload.pop("sequence")
    state = payload.pop("state")
    durable = payload.pop("durable")
    event_id = payload.pop("event_id")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": event.event_type.value,
        "event_id": event_id,
        "session_id": session_id,
        "turn_id": turn_id,
        "sequence": sequence,
        "state": state.value,
        "durable": durable,
        "payload": _json_value(payload),
    }


def event_from_dict(data: Mapping[str, Any]) -> AgentEventType:
    _require_version(data)
    event_type = _EVENT_TYPES.get(str(data.get("type", "")))
    if event_type is None:
        raise ValueError(f"unknown event type: {data.get('type')}")
    return event_type(
        session_id=str(data["session_id"]),
        turn_id=(str(data["turn_id"]) if data.get("turn_id") is not None else None),
        sequence=int(data["sequence"]),
        state=AgentState(data["state"]),
        durable=bool(data["durable"]),
        event_id=str(data["event_id"]),
        **_payload(data),
    )


def _require_version(data: Mapping[str, Any]) -> None:
    if data.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported protocol version: {data.get('protocol_version')}"
        )


def _payload(data: Mapping[str, Any]) -> dict[str, Any]:
    payload = data.get("payload", {})
    if not isinstance(payload, Mapping):
        raise ValueError("protocol payload must be an object")
    return dict(payload)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
