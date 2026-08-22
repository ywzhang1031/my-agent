from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

from .model_invoker import ModelInvoker
from .modes import BUILD_MODE, CollaborationMode, get_mode
from .protocol import (
    AbortTurnCommand,
    AgentCommandType,
    AgentEventType,
    AgentState,
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    ContentDeltaEvent,
    ContextCompactedEvent,
    ExecutePlanCommand,
    InterruptTurnCommand,
    ModeChangedEvent,
    ModelChangedEvent,
    ModelStartedEvent,
    PlanApprovedEvent,
    PlanRevisionRequestedEvent,
    ProviderRetryEvent,
    ReasoningDeltaEvent,
    ResolveApprovalCommand,
    RevisePlanCommand,
    StartTurnCommand,
    SwitchModeCommand,
    SwitchModelCommand,
    ToolFinishedEvent,
    ToolProposedEvent,
    ToolStartedEvent,
    TurnAbortedEvent,
    TurnCompletedEvent,
    TurnErrorEvent,
    TurnFailedEvent,
    TurnInterruptedEvent,
    TurnResumedEvent,
    TurnStartedEvent,
)
from .runtime import (
    AgentRuntime,
    InterruptTurn,
    ResolveApproval,
    RuntimeEvent,
    StartTurn,
)
from .session import SessionState
from .tools import ToolRegistry


EXECUTE_APPROVED_PLAN_PROMPT = (
    "Execute the approved plan from the previous assistant message. "
    "Re-check the current workspace state, implement it, and validate the result."
)

ModelResolver = Callable[[str], ModelInvoker]
ToolRegistryFactory = Callable[[CollaborationMode], ToolRegistry]
SessionSaver = Callable[[SessionState], None]


@dataclass(frozen=True)
class ServiceSessionView:
    session_id: str
    workspace: str
    provider_id: str
    model_id: str
    mode_id: str
    pending_turn_id: str | None
    proposed_plan_id: str | None
    proposed_plan_status: str | None


@dataclass
class _SessionHandle:
    runtime: AgentRuntime
    sequence: int = 0
    forwarding: bool = False
    forwarder: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class AgentService:
    """Application boundary shared by CLI and future remote transports."""

    def __init__(
        self,
        *,
        tool_registry_factory: ToolRegistryFactory,
        model_resolver: ModelResolver | None = None,
        save_session: SessionSaver | None = None,
    ) -> None:
        self._tool_registry_factory = tool_registry_factory
        self._model_resolver = model_resolver
        self._save_session = save_session or (lambda state: None)
        self._sessions: dict[str, _SessionHandle] = {}
        self._events: queue.Queue[AgentEventType] = queue.Queue()
        self._lock = threading.Lock()

    def register(self, runtime: AgentRuntime) -> None:
        session_id = runtime.session.session_id
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"session already registered: {session_id}")
            self._sessions[session_id] = _SessionHandle(runtime=runtime)

    def dispatch(self, command: AgentCommandType) -> None:
        handle = self._handle(command.session_id)
        runtime = handle.runtime
        if isinstance(command, StartTurnCommand):
            self._start_turn(
                handle,
                StartTurn(
                    user_input=command.user_input,
                    retry=command.retry,
                    approved_plan_id=command.approved_plan_id,
                ),
            )
            return
        if isinstance(command, InterruptTurnCommand):
            runtime.send(InterruptTurn())
            return
        if isinstance(command, ResolveApprovalCommand):
            runtime.send(
                ResolveApproval(
                    request_id=command.request_id,
                    decision=command.decision,
                )
            )
            return
        if isinstance(command, SwitchModelCommand):
            self._switch_model(handle, command.model_ref)
            return
        if isinstance(command, SwitchModeCommand):
            self._switch_mode(handle, command.mode_id)
            return
        if isinstance(command, ExecutePlanCommand):
            self._execute_plan(handle)
            return
        if isinstance(command, RevisePlanCommand):
            self._revise_plan(handle)
            return
        if isinstance(command, AbortTurnCommand):
            self._abort_turn(handle)
            return
        raise TypeError(f"unsupported service command: {type(command).__name__}")

    def next_event(self, timeout: float | None = None) -> AgentEventType:
        return self._events.get(timeout=timeout)

    def inspect_context(self, session_id: str):
        handle = self._handle(session_id)
        return handle.runtime.loop.inspect_context(handle.runtime.session)

    def session_view(self, session_id: str) -> ServiceSessionView:
        runtime = self._handle(session_id).runtime
        state = runtime.session
        plan = state.proposed_plan
        return ServiceSessionView(
            session_id=state.session_id,
            workspace=state.workspace,
            provider_id=runtime.loop.model.profile.provider_id,
            model_id=runtime.loop.model.profile.model_id,
            mode_id=runtime.loop.mode.mode_id,
            pending_turn_id=(state.pending_turn.turn_id if state.pending_turn else None),
            proposed_plan_id=(plan.plan_id if plan else None),
            proposed_plan_status=(plan.status if plan else None),
        )

    def tool_names(self, session_id: str) -> list[str]:
        return self._handle(session_id).runtime.loop.tools.names()

    def wait(self, session_id: str, timeout: float = 5.0) -> None:
        handle = self._handle(session_id)
        handle.runtime.wait(timeout)
        forwarder = handle.forwarder
        if forwarder is not None:
            forwarder.join(timeout)
            if forwarder.is_alive():
                raise RuntimeError("service event forwarder did not finish")

    def close(self, timeout: float = 5.0) -> None:
        with self._lock:
            handles = list(self._sessions.values())
            self._sessions.clear()
        for handle in handles:
            handle.runtime.close(timeout)
            if handle.forwarder is not None:
                handle.forwarder.join(timeout)

    def _handle(self, session_id: str) -> _SessionHandle:
        with self._lock:
            handle = self._sessions.get(session_id)
        if handle is None:
            raise ValueError(f"session is not registered: {session_id}")
        return handle

    def _start_turn(self, handle: _SessionHandle, command: StartTurn) -> None:
        with handle.lock:
            if handle.forwarding:
                raise RuntimeError("session already has an active event stream")
            handle.runtime.send(command)
            handle.forwarding = True
            handle.forwarder = threading.Thread(
                target=self._forward_runtime_events,
                args=(handle,),
                name=f"agent-service-{handle.runtime.session.session_id}",
                daemon=True,
            )
            handle.forwarder.start()

    def _forward_runtime_events(self, handle: _SessionHandle) -> None:
        try:
            while True:
                event = handle.runtime.next_event()
                self._publish_runtime_event(handle, event)
                if event.kind in {
                    "turn_completed",
                    "turn_failed",
                    "turn_interrupted",
                }:
                    break
        finally:
            with handle.lock:
                handle.forwarding = False
            self._save_session(handle.runtime.session)

    def _switch_model(self, handle: _SessionHandle, model_ref: str) -> None:
        if self._model_resolver is None:
            raise RuntimeError("agent service has no model resolver")
        model = self._model_resolver(model_ref)
        handle.runtime.switch_model(model)
        state = handle.runtime.session
        state.provider_id = model.profile.provider_id
        state.model_id = model.profile.model_id
        self._save_session(state)
        self._publish(
            handle,
            ModelChangedEvent,
            AgentState.IDLE,
            provider_id=model.profile.provider_id,
            model_id=model.profile.model_id,
        )

    def _switch_mode(self, handle: _SessionHandle, mode_id: str) -> None:
        mode = get_mode(mode_id)
        tools = self._tool_registry_factory(mode)
        handle.runtime.switch_mode(mode, tools)
        handle.runtime.session.mode_id = mode.mode_id
        self._save_session(handle.runtime.session)
        self._publish(
            handle,
            ModeChangedEvent,
            AgentState.IDLE,
            mode_id=mode.mode_id,
        )

    def _execute_plan(self, handle: _SessionHandle) -> None:
        runtime = handle.runtime
        plan = runtime.loop.approve_plan(runtime.session)
        tools = self._tool_registry_factory(BUILD_MODE)
        runtime.switch_mode(BUILD_MODE, tools)
        runtime.session.mode_id = BUILD_MODE.mode_id
        self._save_session(runtime.session)
        self._publish(
            handle,
            PlanApprovedEvent,
            AgentState.IDLE,
            plan_id=plan.plan_id,
            mode_id=BUILD_MODE.mode_id,
        )
        self._start_turn(
            handle,
            StartTurn(
                user_input=EXECUTE_APPROVED_PLAN_PROMPT,
                approved_plan_id=plan.plan_id,
            ),
        )

    def _revise_plan(self, handle: _SessionHandle) -> None:
        runtime = handle.runtime
        plan = runtime.loop.request_plan_revision(runtime.session)
        self._save_session(runtime.session)
        self._publish(
            handle,
            PlanRevisionRequestedEvent,
            AgentState.IDLE,
            plan_id=plan.plan_id,
        )

    def _abort_turn(self, handle: _SessionHandle) -> None:
        runtime = handle.runtime
        pending = runtime.loop.abort_turn(runtime.session)
        self._save_session(runtime.session)
        self._publish(
            handle,
            TurnAbortedEvent,
            AgentState.IDLE,
            turn_id=pending.turn_id,
            tool_calls_executed=pending.tool_calls_executed,
        )

    def _publish_runtime_event(
        self,
        handle: _SessionHandle,
        event: RuntimeEvent,
    ) -> None:
        data = event.data
        state = AgentState(event.state.value)
        if event.kind == "turn_started":
            self._publish(
                handle,
                TurnStartedEvent,
                state,
                turn_id=str(data["turn_id"]),
                task=str(data["task"]),
            )
        elif event.kind == "turn_resumed":
            self._publish(handle, TurnResumedEvent, state, step=int(data["step"]))
        elif event.kind == "context_compacted":
            self._publish(
                handle,
                ContextCompactedEvent,
                state,
                step=int(data["step"]),
                snapshot={key: value for key, value in data.items() if key != "step"},
            )
        elif event.kind == "model_started":
            self._publish(handle, ModelStartedEvent, state, step=int(data["step"]))
        elif event.kind == "content_delta":
            self._publish(
                handle,
                ContentDeltaEvent,
                state,
                step=int(data["step"]),
                text=event.text,
            )
        elif event.kind == "reasoning_delta":
            self._publish(
                handle,
                ReasoningDeltaEvent,
                state,
                step=int(data["step"]),
                text=event.text,
            )
        elif event.kind == "provider_retry":
            self._publish(
                handle,
                ProviderRetryEvent,
                state,
                step=int(data["step"]),
                error=event.text,
                attempt=int(data["attempt"]),
                delay_seconds=float(data["delay_seconds"]),
                details=dict(data),
            )
        elif event.kind == "tool_proposed":
            self._publish(
                handle,
                ToolProposedEvent,
                state,
                step=int(data["step"]),
                call_id=str(data["call_id"]),
                name=str(data["name"]),
                arguments=dict(data["arguments"]),
            )
        elif event.kind == "tool_started":
            self._publish(
                handle,
                ToolStartedEvent,
                state,
                step=int(data["step"]),
                call_id=str(data["call_id"]),
                name=str(data["name"]),
                arguments=dict(data["arguments"]),
            )
        elif event.kind == "tool_finished":
            self._publish(
                handle,
                ToolFinishedEvent,
                state,
                step=int(data["step"]),
                call_id=str(data["call_id"]),
                name=str(data["name"]),
                ok=bool(data["ok"]),
                status=str(data["status"]),
                duration_ms=data.get("duration_ms"),
            )
        elif event.kind == "approval_requested":
            self._publish(
                handle,
                ApprovalRequestedEvent,
                state,
                step=int(data["step"]),
                call_id=str(data["call_id"]),
                tool_name=str(data["tool_name"]),
                request_id=str(data["request_id"]),
                action=str(data["action"]),
                resource=str(data["resource"]),
                description=str(data["description"]),
                details=dict(data.get("details", {})),
            )
        elif event.kind == "approval_resolved":
            self._publish(
                handle,
                ApprovalResolvedEvent,
                state,
                step=int(data["step"]),
                call_id=str(data["call_id"]),
                tool_name=str(data["tool_name"]),
                request_id=str(data["request_id"]),
                decision=str(data["decision"]),
                prompted=bool(data["prompted"]),
                source=str(data["source"]),
            )
        elif event.kind == "turn_error":
            self._publish(
                handle,
                TurnErrorEvent,
                state,
                step=int(data["step"]),
                error=event.text,
                retryable=bool(data["retryable"]),
            )
        elif event.kind == "turn_completed":
            self._publish(
                handle,
                TurnCompletedEvent,
                state,
                answer=event.text,
                steps=int(data["steps"]),
                stopped_by=str(data["stopped_by"]),
                mode_id=str(data["mode_id"]),
                plan_id=data.get("plan_id"),
            )
        elif event.kind == "turn_failed":
            self._publish(handle, TurnFailedEvent, state, error=event.text)
        elif event.kind == "turn_interrupted":
            self._publish(handle, TurnInterruptedEvent, state, reason=event.text)
        else:
            raise ValueError(f"unsupported runtime event: {event.kind}")

    def _publish(self, handle, event_type, state: AgentState, **payload) -> None:
        with handle.lock:
            handle.sequence += 1
            sequence = handle.sequence
        self._events.put(
            event_type(
                session_id=handle.runtime.session.session_id,
                sequence=sequence,
                state=state,
                **payload,
            )
        )
