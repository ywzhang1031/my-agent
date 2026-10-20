from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, ClassVar, Mapping, TypeAlias
from uuid import uuid4

from .control import ApprovalDecision


PROTOCOL_VERSION = 1


class AgentState(str, Enum):
    IDLE = "idle"
    RUNNING_MODEL = "running_model"
    WAITING_APPROVAL = "waiting_approval"
    RUNNING_TOOL = "running_tool"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class CommandType(str, Enum):
    START_TURN = "turn.start"
    INTERRUPT_TURN = "turn.interrupt"
    RESOLVE_APPROVAL = "approval.resolve"
    SWITCH_MODEL = "model.switch"
    SWITCH_MODE = "mode.switch"
    EXECUTE_PLAN = "plan.execute"
    REVISE_PLAN = "plan.revise"
    ABORT_TURN = "turn.abort"
    REPLAY_EVENTS = "events.replay"


class EventType(str, Enum):
    TURN_STARTED = "turn.started"
    TURN_RESUMED = "turn.resumed"
    CONTEXT_COMPACTED = "context.compacted"
    MODEL_STARTED = "model.started"
    CONTENT_DELTA = "model.content_delta"
    REASONING_DELTA = "model.reasoning_delta"
    PROVIDER_RETRY = "provider.retry"
    TOOL_PROPOSED = "tool.proposed"
    TOOL_STARTED = "tool.started"
    TOOL_FINISHED = "tool.finished"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_RESOLVED = "approval.resolved"
    TURN_ERROR = "turn.error"
    TURN_COMPLETED = "turn.completed"
    TURN_FAILED = "turn.failed"
    TURN_INTERRUPTED = "turn.interrupted"
    MODEL_CHANGED = "model.changed"
    MODE_CHANGED = "mode.changed"
    PLAN_APPROVED = "plan.approved"
    PLAN_REVISION_REQUESTED = "plan.revision_requested"
    TURN_ABORTED = "turn.aborted"
    REPLAY_COMPLETED = "events.replay_completed"


@dataclass(frozen=True, kw_only=True)
class AgentCommand:
    session_id: str
    command_id: str = field(default_factory=lambda: uuid4().hex)
    command_type: ClassVar[CommandType]


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
class ExecutePlanCommand(AgentCommand):
    command_type: ClassVar[CommandType] = CommandType.EXECUTE_PLAN


@dataclass(frozen=True, kw_only=True)
class RevisePlanCommand(AgentCommand):
    command_type: ClassVar[CommandType] = CommandType.REVISE_PLAN


@dataclass(frozen=True, kw_only=True)
class AbortTurnCommand(AgentCommand):
    command_type: ClassVar[CommandType] = CommandType.ABORT_TURN


@dataclass(frozen=True, kw_only=True)
class ReplayEventsCommand(AgentCommand):
    after_sequence: int = 0
    command_type: ClassVar[CommandType] = CommandType.REPLAY_EVENTS

    def __post_init__(self) -> None:
        if self.after_sequence < 0:
            raise ValueError("after_sequence must not be negative")


AgentCommandType: TypeAlias = (
    StartTurnCommand
    | InterruptTurnCommand
    | ResolveApprovalCommand
    | SwitchModelCommand
    | SwitchModeCommand
    | ExecutePlanCommand
    | RevisePlanCommand
    | AbortTurnCommand
    | ReplayEventsCommand
)


@dataclass(frozen=True, kw_only=True)
class AgentEvent:
    session_id: str
    sequence: int
    state: AgentState
    event_id: str = field(default_factory=lambda: uuid4().hex)
    event_type: ClassVar[EventType]


@dataclass(frozen=True, kw_only=True)
class TurnStartedEvent(AgentEvent):
    turn_id: str
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
class ContentDeltaEvent(AgentEvent):
    step: int
    text: str
    event_type: ClassVar[EventType] = EventType.CONTENT_DELTA


@dataclass(frozen=True, kw_only=True)
class ReasoningDeltaEvent(AgentEvent):
    step: int
    text: str
    event_type: ClassVar[EventType] = EventType.REASONING_DELTA


@dataclass(frozen=True, kw_only=True)
class ProviderRetryEvent(AgentEvent):
    step: int
    error: str
    attempt: int
    delay_seconds: float
    details: dict[str, Any]
    event_type: ClassVar[EventType] = EventType.PROVIDER_RETRY


@dataclass(frozen=True, kw_only=True)
class ToolProposedEvent(AgentEvent):
    step: int
    call_id: str
    name: str
    arguments: dict[str, Any]
    event_type: ClassVar[EventType] = EventType.TOOL_PROPOSED


@dataclass(frozen=True, kw_only=True)
class ToolStartedEvent(AgentEvent):
    step: int
    call_id: str
    name: str
    arguments: dict[str, Any]
    event_type: ClassVar[EventType] = EventType.TOOL_STARTED


@dataclass(frozen=True, kw_only=True)
class ToolFinishedEvent(AgentEvent):
    step: int
    call_id: str
    name: str
    ok: bool
    status: str
    duration_ms: int | None = None
    event_type: ClassVar[EventType] = EventType.TOOL_FINISHED


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
    turn_id: str
    tool_calls_executed: int
    event_type: ClassVar[EventType] = EventType.TURN_ABORTED


@dataclass(frozen=True, kw_only=True)
class ReplayCompletedEvent(AgentEvent):
    after_sequence: int
    replayed_events: int
    latest_sequence: int
    event_type: ClassVar[EventType] = EventType.REPLAY_COMPLETED


AgentEventType: TypeAlias = (
    TurnStartedEvent
    | TurnResumedEvent
    | ContextCompactedEvent
    | ModelStartedEvent
    | ContentDeltaEvent
    | ReasoningDeltaEvent
    | ProviderRetryEvent
    | ToolProposedEvent
    | ToolStartedEvent
    | ToolFinishedEvent
    | ApprovalRequestedEvent
    | ApprovalResolvedEvent
    | TurnErrorEvent
    | TurnCompletedEvent
    | TurnFailedEvent
    | TurnInterruptedEvent
    | ModelChangedEvent
    | ModeChangedEvent
    | PlanApprovedEvent
    | PlanRevisionRequestedEvent
    | TurnAbortedEvent
    | ReplayCompletedEvent
)


_COMMAND_TYPES = {
    command_type.command_type.value: command_type
    for command_type in (
        StartTurnCommand,
        InterruptTurnCommand,
        ResolveApprovalCommand,
        SwitchModelCommand,
        SwitchModeCommand,
        ExecutePlanCommand,
        RevisePlanCommand,
        AbortTurnCommand,
        ReplayEventsCommand,
    )
}

_EVENT_TYPES = {
    event_type.event_type.value: event_type
    for event_type in (
        TurnStartedEvent,
        TurnResumedEvent,
        ContextCompactedEvent,
        ModelStartedEvent,
        ContentDeltaEvent,
        ReasoningDeltaEvent,
        ProviderRetryEvent,
        ToolProposedEvent,
        ToolStartedEvent,
        ToolFinishedEvent,
        ApprovalRequestedEvent,
        ApprovalResolvedEvent,
        TurnErrorEvent,
        TurnCompletedEvent,
        TurnFailedEvent,
        TurnInterruptedEvent,
        ModelChangedEvent,
        ModeChangedEvent,
        PlanApprovedEvent,
        PlanRevisionRequestedEvent,
        TurnAbortedEvent,
        ReplayCompletedEvent,
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
    sequence = payload.pop("sequence")
    state = payload.pop("state")
    event_id = payload.pop("event_id")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": event.event_type.value,
        "event_id": event_id,
        "session_id": session_id,
        "sequence": sequence,
        "state": state.value,
        "payload": _json_value(payload),
    }


def event_from_dict(data: Mapping[str, Any]) -> AgentEventType:
    _require_version(data)
    event_type = _EVENT_TYPES.get(str(data.get("type", "")))
    if event_type is None:
        raise ValueError(f"unknown event type: {data.get('type')}")
    return event_type(
        session_id=str(data["session_id"]),
        sequence=int(data["sequence"]),
        state=AgentState(data["state"]),
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
