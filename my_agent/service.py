from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .control import ApprovalDecision, ApprovalGrant
from .event_broker import EventBroker
from .event_store import EventStore
from .messages import UserMessage
from .goals import GoalState
from .model_invoker import ModelInvoker
from .modes import BUILD_MODE, CollaborationMode, get_mode
from .protocol import (
    AbortTurnCommand,
    AgentCommandType,
    AgentEventType,
    AgentState,
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    ClearGoalCommand,
    ClearPermissionGrantsCommand,
    ContextCompactedEvent,
    ExecutePlanCommand,
    GoalClearedEvent,
    GoalUpdatedEvent,
    InterruptTurnCommand,
    ItemCompletedEvent,
    ItemDeltaEvent,
    ItemFailedEvent,
    ItemStartedEvent,
    ModeChangedEvent,
    ModelChangedEvent,
    ModelStartedEvent,
    PlanApprovedEvent,
    PlanRevisionRequestedEvent,
    PermissionProfileChangedEvent,
    PermissionGrantsChangedEvent,
    ProviderRetryEvent,
    RenameSessionCommand,
    RevokePermissionGrantCommand,
    ResolveApprovalCommand,
    RevisePlanCommand,
    SandboxChangedEvent,
    StartTurnCommand,
    SessionRenamedEvent,
    SwitchModeCommand,
    SwitchModelCommand,
    SwitchPermissionProfileCommand,
    SwitchSandboxCommand,
    TurnAbortedEvent,
    TurnCompletedEvent,
    TurnContinuationRequiredEvent,
    TurnErrorEvent,
    TurnFailedEvent,
    TurnInterruptedEvent,
    TurnResumedEvent,
    TurnStartedEvent,
)
from .projection import SessionProjection
from .runtime import (
    AgentRuntime,
    InterruptTurn,
    ResolveApproval,
    RuntimeEvent,
    StartTurn,
)
from .session import SessionState, normalize_session_title
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
    title: str
    workspace: str
    provider_id: str
    model_id: str
    mode_id: str
    permission_profile_id: str
    sandbox_mode: str
    goal: dict[str, Any] | None
    runtime_state: str
    pending_turn_id: str | None
    proposed_plan_id: str | None
    proposed_plan_status: str | None


@dataclass
class _SessionHandle:
    runtime: AgentRuntime
    projection: SessionProjection
    sequence: int = 0
    current_turn_id: str | None = None
    active_items: dict[str, tuple[str, int, str]] = field(default_factory=dict)
    completed_items: set[str] = field(default_factory=set)
    forwarding: bool = False
    forwarder: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    operation_lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False


class AgentService:
    """Application boundary shared by CLI and future remote transports."""

    def __init__(
        self,
        *,
        tool_registry_factory: ToolRegistryFactory,
        model_resolver: ModelResolver | None = None,
        save_session: SessionSaver | None = None,
        event_store: EventStore | None = None,
    ) -> None:
        self._tool_registry_factory = tool_registry_factory
        self._model_resolver = model_resolver
        self._save_session = save_session or (lambda state: None)
        self._event_store = event_store
        self._sessions: dict[str, _SessionHandle] = {}
        self._event_broker = EventBroker()
        self._default_subscription_id: str | None = None
        self._lock = threading.Lock()

    def register(self, runtime: AgentRuntime) -> None:
        session_id = runtime.session.session_id
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"session already registered: {session_id}")
            projection = SessionProjection(session_id)
            persisted_events = (
                self._event_store.replay(session_id)
                if self._event_store is not None
                else []
            )
            for event in persisted_events:
                projection.apply(event)
            self._sessions[session_id] = _SessionHandle(
                runtime=runtime,
                projection=projection,
                sequence=projection.latest_sequence,
            )

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def registered_session_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._sessions)

    def unregister(self, session_id: str, timeout: float = 5.0) -> None:
        with self._lock:
            handle = self._sessions.get(session_id)
            if handle is None:
                raise ValueError(f"session is not registered: {session_id}")
            with handle.operation_lock:
                handle.closed = True
                del self._sessions[session_id]
        self._close_handle(handle, timeout)

    def dispatch(self, command: AgentCommandType) -> None:
        handle = self._handle(command.session_id)
        with handle.operation_lock:
            if handle.closed:
                raise ValueError(f"session is not registered: {command.session_id}")
            self._dispatch_to_handle(handle, command)

    def _dispatch_to_handle(
        self,
        handle: _SessionHandle,
        command: AgentCommandType,
    ) -> None:
        runtime = handle.runtime
        if isinstance(command, RenameSessionCommand):
            self._rename_session(handle, command.title)
            return
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
        if isinstance(command, SwitchPermissionProfileCommand):
            self._switch_permission_profile(handle, command.profile_id)
            return
        if isinstance(command, RevokePermissionGrantCommand):
            self._revoke_permission_grant(handle, command.action, command.resource)
            return
        if isinstance(command, ClearPermissionGrantsCommand):
            self._clear_permission_grants(handle)
            return
        if isinstance(command, ClearGoalCommand):
            self._clear_goal(handle)
            return
        if isinstance(command, SwitchSandboxCommand):
            self._switch_sandbox(handle, command.mode)
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

    def subscribe_events(
        self,
        *,
        session_ids: Iterable[str] | None = None,
        after_sequences: Mapping[str, int] | None = None,
    ) -> tuple[str, dict[str, int]]:
        selected = (
            tuple(sorted(session_ids))
            if session_ids is not None
            else self.registered_session_ids()
        )
        handles = [(session_id, self._handle(session_id)) for session_id in selected]
        for _, handle in handles:
            handle.lock.acquire()
        try:
            subscription_id = self._event_broker.subscribe(
                None if session_ids is None else selected
            )
            if after_sequences and self._event_store is not None:
                for session_id, after_sequence in after_sequences.items():
                    if session_id not in dict(handles):
                        raise ValueError(
                            f"event replay session is not subscribed: {session_id}"
                        )
                    for event in self._event_store.replay(
                        session_id,
                        after_sequence=after_sequence,
                    ):
                        self._event_broker.deliver(subscription_id, event)
            cursors = {session_id: handle.sequence for session_id, handle in handles}
        except Exception:
            if "subscription_id" in locals():
                self._event_broker.unsubscribe(subscription_id)
            raise
        finally:
            for _, handle in reversed(handles):
                handle.lock.release()
        return subscription_id, cursors

    def unsubscribe_events(self, subscription_id: str) -> bool:
        return self._event_broker.unsubscribe(subscription_id)

    def open_default_event_stream(
        self,
        *,
        session_ids: Iterable[str] | None = None,
    ) -> str:
        if self._default_subscription_id is not None:
            raise RuntimeError("default event stream is already open")
        subscription_id, _ = self.subscribe_events(session_ids=session_ids)
        self._default_subscription_id = subscription_id
        return subscription_id

    def next_event(
        self,
        subscription_id: str | None = None,
        timeout: float | None = None,
    ) -> AgentEventType:
        if subscription_id is None:
            subscription_id = self._default_subscription_id
        if subscription_id is None:
            raise RuntimeError("event subscription is required")
        return self._event_broker.next_event(subscription_id, timeout=timeout)

    def projection(self, session_id: str) -> dict:
        with self._lock:
            handle = self._sessions.get(session_id)
        if handle is not None:
            with handle.lock:
                return handle.projection.to_dict()
        projection = SessionProjection(session_id)
        if self._event_store is not None:
            for event in self._event_store.replay(session_id):
                projection.apply(event)
        return projection.to_dict()

    def inspect_context(self, session_id: str):
        handle = self._handle(session_id)
        return handle.runtime.loop.inspect_context(handle.runtime.session)

    def permission_grants(self, session_id: str) -> tuple[ApprovalGrant, ...]:
        return self._handle(session_id).runtime.permission_grants()

    def current_goal(self, session_id: str) -> GoalState | None:
        return self._handle(session_id).runtime.session.goal

    def session_view(self, session_id: str) -> ServiceSessionView:
        runtime = self._handle(session_id).runtime
        state = runtime.session
        plan = state.proposed_plan
        return ServiceSessionView(
            session_id=state.session_id,
            title=state.title,
            workspace=state.workspace,
            provider_id=runtime.loop.model.profile.provider_id,
            model_id=runtime.loop.model.profile.model_id,
            mode_id=runtime.loop.mode.mode_id,
            permission_profile_id=state.permission_profile_id,
            sandbox_mode=state.sandbox_mode,
            goal=state.goal.to_dict() if state.goal is not None else None,
            runtime_state=runtime.state.value,
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
            for handle in handles:
                with handle.operation_lock:
                    handle.closed = True
            self._sessions.clear()
        for handle in handles:
            self._close_handle(handle, timeout)
        self._event_broker.close()

    def _close_handle(self, handle: _SessionHandle, timeout: float) -> None:
        handle.runtime.close(timeout)
        if handle.forwarder is not None:
            handle.forwarder.join(timeout)
            if handle.forwarder.is_alive():
                raise RuntimeError("service event forwarder did not finish")
        self._save_session(handle.runtime.session)

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
            if command.retry and handle.runtime.session.pending_turn is not None:
                handle.current_turn_id = handle.runtime.session.pending_turn.turn_id
            elif not command.retry:
                turn_number = (
                    sum(
                        isinstance(message, UserMessage)
                        for message in handle.runtime.session.messages
                    )
                    + 1
                )
                handle.current_turn_id = (
                    f"{handle.runtime.session.session_id}:{turn_number}"
                )
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
        terminal_saved = False
        try:
            while True:
                event = handle.runtime.next_event()
                terminal = event.kind in {
                    "turn_completed",
                    "turn_continuation_required",
                    "turn_failed",
                    "turn_interrupted",
                }
                if terminal:
                    self._save_session(handle.runtime.session)
                    terminal_saved = True
                self._publish_runtime_event(handle, event)
                if terminal:
                    break
        finally:
            with handle.lock:
                handle.forwarding = False
            if not terminal_saved:
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
            turn_id=None,
            provider_id=model.profile.provider_id,
            model_id=model.profile.model_id,
        )

    def _rename_session(self, handle: _SessionHandle, title: str) -> None:
        state = handle.runtime.session
        state.title = normalize_session_title(title)
        self._save_session(state)
        self._publish(
            handle,
            SessionRenamedEvent,
            AgentState.IDLE,
            turn_id=None,
            title=state.title,
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
            turn_id=None,
            mode_id=mode.mode_id,
        )

    def _switch_permission_profile(
        self,
        handle: _SessionHandle,
        profile_id: str,
    ) -> None:
        handle.runtime.switch_permission_profile(profile_id)
        profile = handle.runtime.loop.permissions.profile
        handle.runtime.session.permission_profile_id = profile.profile_id
        self._save_session(handle.runtime.session)
        self._publish(
            handle,
            PermissionProfileChangedEvent,
            AgentState.IDLE,
            turn_id=None,
            profile_id=profile.profile_id,
        )

    def _switch_sandbox(self, handle: _SessionHandle, mode: str) -> None:
        handle.runtime.switch_sandbox(mode)
        selected_mode = handle.runtime.loop.sandbox_policy.mode
        handle.runtime.session.sandbox_mode = selected_mode
        self._save_session(handle.runtime.session)
        self._publish(
            handle,
            SandboxChangedEvent,
            AgentState.IDLE,
            turn_id=None,
            mode=selected_mode,
        )

    def _revoke_permission_grant(
        self,
        handle: _SessionHandle,
        action: str,
        resource: str,
    ) -> None:
        removed = handle.runtime.revoke_permission_grant(action, resource)
        self._publish_permission_grants_changed(
            handle,
            reason="revoked",
            removed_count=int(removed),
        )

    def _clear_permission_grants(self, handle: _SessionHandle) -> None:
        removed_count = handle.runtime.clear_permission_grants()
        self._publish_permission_grants_changed(
            handle,
            reason="cleared",
            removed_count=removed_count,
        )

    def _clear_goal(self, handle: _SessionHandle) -> None:
        goal_id = handle.runtime.clear_goal()
        self._save_session(handle.runtime.session)
        self._publish(
            handle,
            GoalClearedEvent,
            AgentState.IDLE,
            turn_id=None,
            goal_id=goal_id,
        )

    def _publish_permission_grants_changed(
        self,
        handle: _SessionHandle,
        *,
        reason: str,
        removed_count: int = 0,
        turn_id: str | None = None,
        state: AgentState | None = None,
    ) -> None:
        self._publish(
            handle,
            PermissionGrantsChangedEvent,
            state or AgentState(handle.runtime.state.value),
            turn_id=turn_id,
            durable=False,
            grants=[grant.to_dict() for grant in handle.runtime.permission_grants()],
            reason=reason,
            removed_count=removed_count,
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
            turn_id=plan.source_turn_id,
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
            turn_id=plan.source_turn_id,
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
            turn_id = str(data["turn_id"])
            handle.current_turn_id = turn_id
            self._publish(
                handle,
                TurnStartedEvent,
                state,
                turn_id=turn_id,
                task=str(data["task"]),
            )
            user_item_id = f"{turn_id}:user"
            self._ensure_item(
                handle,
                turn_id=turn_id,
                item_id=user_item_id,
                item_type="user_message",
                step=0,
                state=state,
            )
            self._complete_item(
                handle,
                turn_id=turn_id,
                item_id=user_item_id,
                item_type="user_message",
                step=0,
                state=state,
                content=str(data["task"]),
            )
        elif event.kind == "turn_resumed":
            turn_id = self._current_turn_id(handle)
            self._publish(
                handle,
                TurnResumedEvent,
                state,
                turn_id=turn_id,
                step=int(data["step"]),
            )
        elif event.kind == "context_compacted":
            self._publish(
                handle,
                ContextCompactedEvent,
                state,
                turn_id=self._current_turn_id(handle),
                step=int(data["step"]),
                snapshot={key: value for key, value in data.items() if key != "step"},
            )
        elif event.kind == "model_started":
            self._publish(
                handle,
                ModelStartedEvent,
                state,
                turn_id=self._current_turn_id(handle),
                step=int(data["step"]),
            )
        elif event.kind == "content_delta":
            self._publish_item_delta(
                handle,
                turn_id=self._current_turn_id(handle),
                item_type="assistant_message",
                step=int(data["step"]),
                delta=event.text,
                state=state,
            )
        elif event.kind == "reasoning_delta":
            self._publish_item_delta(
                handle,
                turn_id=self._current_turn_id(handle),
                item_type="reasoning",
                step=int(data["step"]),
                delta=event.text,
                state=state,
            )
        elif event.kind == "model_completed":
            turn_id = self._current_turn_id(handle)
            step = int(data["step"])
            for item_type, content in (
                ("reasoning", str(data.get("reasoning") or "")),
                ("assistant_message", str(data.get("content") or "")),
            ):
                item_id = self._model_item_id(turn_id, step, item_type)
                if content or item_id in handle.active_items:
                    self._ensure_item(
                        handle,
                        turn_id=turn_id,
                        item_id=item_id,
                        item_type=item_type,
                        step=step,
                        state=state,
                    )
                    self._complete_item(
                        handle,
                        turn_id=turn_id,
                        item_id=item_id,
                        item_type=item_type,
                        step=step,
                        state=state,
                        content=content,
                        data={
                            "finish_reason": data.get("finish_reason"),
                            "usage": data.get("usage"),
                            "metadata": data.get("metadata"),
                        },
                    )
        elif event.kind == "provider_retry":
            self._publish(
                handle,
                ProviderRetryEvent,
                state,
                turn_id=self._current_turn_id(handle),
                step=int(data["step"]),
                error=event.text,
                attempt=int(data["attempt"]),
                delay_seconds=float(data["delay_seconds"]),
                details=dict(data),
            )
        elif event.kind == "tool_proposed":
            turn_id = self._current_turn_id(handle)
            self._ensure_item(
                handle,
                turn_id=turn_id,
                item_id=self._tool_item_id(turn_id, str(data["call_id"])),
                item_type="tool_call",
                step=int(data["step"]),
                state=state,
                data={
                    "call_id": str(data["call_id"]),
                    "name": str(data["name"]),
                    "arguments": dict(data["arguments"]),
                    "phase": "proposed",
                },
            )
        elif event.kind == "tool_started":
            turn_id = self._current_turn_id(handle)
            item_id = self._tool_item_id(turn_id, str(data["call_id"]))
            self._ensure_item(
                handle,
                turn_id=turn_id,
                item_id=item_id,
                item_type="tool_call",
                step=int(data["step"]),
                state=state,
                data={
                    "call_id": str(data["call_id"]),
                    "name": str(data["name"]),
                    "arguments": dict(data["arguments"]),
                },
            )
            self._publish(
                handle,
                ItemDeltaEvent,
                state,
                turn_id=turn_id,
                durable=False,
                item_id=item_id,
                item_type="tool_call",
                step=int(data["step"]),
                delta="",
                data={"phase": "running"},
            )
        elif event.kind == "tool_finished":
            turn_id = self._current_turn_id(handle)
            item_id = self._tool_item_id(turn_id, str(data["call_id"]))
            self._ensure_item(
                handle,
                turn_id=turn_id,
                item_id=item_id,
                item_type="tool_call",
                step=int(data["step"]),
                state=state,
                data={
                    "call_id": str(data["call_id"]),
                    "name": str(data["name"]),
                },
            )
            self._complete_item(
                handle,
                turn_id=turn_id,
                item_id=item_id,
                item_type="tool_call",
                step=int(data["step"]),
                state=state,
                data={
                    "call_id": str(data["call_id"]),
                    "name": str(data["name"]),
                    "arguments": dict(data.get("arguments", {})),
                    "ok": bool(data["ok"]),
                    "status": str(data["status"]),
                    "duration_ms": data.get("duration_ms"),
                    "result": dict(data.get("result", {})),
                    "phase": "completed",
                },
            )
        elif event.kind == "approval_requested":
            self._publish(
                handle,
                ApprovalRequestedEvent,
                state,
                turn_id=self._current_turn_id(handle),
                step=int(data["step"]),
                call_id=str(data["call_id"]),
                tool_name=str(data["tool_name"]),
                request_id=str(data["request_id"]),
                action=str(data["action"]),
                resource=str(data["resource"]),
                description=str(data["description"]),
                details=dict(data.get("details", {})),
            )
        elif event.kind == "goal_updated":
            goal = dict(data["goal"])
            self._save_session(handle.runtime.session)
            self._publish(
                handle,
                GoalUpdatedEvent,
                state,
                turn_id=self._current_turn_id(handle),
                goal=goal,
            )
        elif event.kind == "approval_resolved":
            self._publish(
                handle,
                ApprovalResolvedEvent,
                state,
                turn_id=self._current_turn_id(handle),
                step=int(data["step"]),
                call_id=str(data["call_id"]),
                tool_name=str(data["tool_name"]),
                request_id=str(data["request_id"]),
                decision=str(data["decision"]),
                prompted=bool(data["prompted"]),
                source=str(data["source"]),
            )
            if (
                str(data["decision"]) == ApprovalDecision.ALLOW_SESSION.value
                and str(data["source"]) == "prompt"
            ):
                self._publish_permission_grants_changed(
                    handle,
                    reason="granted",
                    turn_id=self._current_turn_id(handle),
                    state=state,
                )
        elif event.kind == "turn_error":
            self._publish(
                handle,
                TurnErrorEvent,
                state,
                turn_id=self._current_turn_id(handle),
                step=int(data["step"]),
                error=event.text,
                retryable=bool(data["retryable"]),
                details=dict(data.get("details", {})),
            )
        elif event.kind == "turn_continuation_required":
            self._publish(
                handle,
                TurnContinuationRequiredEvent,
                state,
                turn_id=self._current_turn_id(handle),
                steps_completed=int(data["steps_completed"]),
                step_budget=int(data["step_budget"]),
                next_step=int(data["next_step"]),
            )
        elif event.kind == "turn_completed":
            turn_id = self._current_turn_id(handle)
            if data.get("plan_id"):
                plan_item_id = str(data["plan_id"])
                self._ensure_item(
                    handle,
                    turn_id=turn_id,
                    item_id=plan_item_id,
                    item_type="plan",
                    step=int(data["steps"]),
                    state=state,
                )
                self._complete_item(
                    handle,
                    turn_id=turn_id,
                    item_id=plan_item_id,
                    item_type="plan",
                    step=int(data["steps"]),
                    state=state,
                    content=event.text,
                )
            self._publish(
                handle,
                TurnCompletedEvent,
                state,
                turn_id=turn_id,
                answer=event.text,
                steps=int(data["steps"]),
                stopped_by=str(data["stopped_by"]),
                mode_id=str(data["mode_id"]),
                plan_id=data.get("plan_id"),
            )
            handle.current_turn_id = None
        elif event.kind == "turn_failed":
            turn_id = self._current_turn_id(handle)
            self._fail_active_items(handle, turn_id, state, "failed", event.text)
            self._publish(
                handle,
                TurnFailedEvent,
                state,
                turn_id=turn_id,
                error=event.text,
            )
            handle.current_turn_id = None
        elif event.kind == "turn_interrupted":
            turn_id = self._current_turn_id(handle)
            self._fail_active_items(handle, turn_id, state, "interrupted", event.text)
            self._publish(
                handle,
                TurnInterruptedEvent,
                state,
                turn_id=turn_id,
                reason=event.text,
            )
            handle.current_turn_id = None
        else:
            raise ValueError(f"unsupported runtime event: {event.kind}")

    def _publish_item_delta(
        self,
        handle: _SessionHandle,
        *,
        turn_id: str,
        item_type: str,
        step: int,
        delta: str,
        state: AgentState,
    ) -> None:
        item_id = self._model_item_id(turn_id, step, item_type)
        self._ensure_item(
            handle,
            turn_id=turn_id,
            item_id=item_id,
            item_type=item_type,
            step=step,
            state=state,
        )
        self._publish(
            handle,
            ItemDeltaEvent,
            state,
            turn_id=turn_id,
            durable=False,
            item_id=item_id,
            item_type=item_type,
            step=step,
            delta=delta,
        )

    def _ensure_item(
        self,
        handle: _SessionHandle,
        *,
        turn_id: str,
        item_id: str,
        item_type: str,
        step: int,
        state: AgentState,
        data: dict | None = None,
    ) -> None:
        if item_id in handle.active_items or item_id in handle.completed_items:
            return
        handle.active_items[item_id] = (item_type, step, turn_id)
        self._publish(
            handle,
            ItemStartedEvent,
            state,
            turn_id=turn_id,
            durable=False,
            item_id=item_id,
            item_type=item_type,
            step=step,
            data=dict(data or {}),
        )

    def _complete_item(
        self,
        handle: _SessionHandle,
        *,
        turn_id: str,
        item_id: str,
        item_type: str,
        step: int,
        state: AgentState,
        content: str = "",
        data: dict | None = None,
    ) -> None:
        if item_id in handle.completed_items:
            return
        handle.active_items.pop(item_id, None)
        handle.completed_items.add(item_id)
        self._publish(
            handle,
            ItemCompletedEvent,
            state,
            turn_id=turn_id,
            item_id=item_id,
            item_type=item_type,
            step=step,
            content=content,
            data=dict(data or {}),
        )

    def _fail_active_items(
        self,
        handle: _SessionHandle,
        turn_id: str,
        state: AgentState,
        status: str,
        error: str,
    ) -> None:
        active = [
            (item_id, item_type, step)
            for item_id, (item_type, step, item_turn_id) in handle.active_items.items()
            if item_turn_id == turn_id
        ]
        for item_id, item_type, step in active:
            handle.active_items.pop(item_id, None)
            handle.completed_items.add(item_id)
            self._publish(
                handle,
                ItemFailedEvent,
                state,
                turn_id=turn_id,
                item_id=item_id,
                item_type=item_type,
                step=step,
                status=status,
                error=error,
            )

    @staticmethod
    def _model_item_id(turn_id: str, step: int, item_type: str) -> str:
        return f"{turn_id}:step:{step}:{item_type}"

    @staticmethod
    def _tool_item_id(turn_id: str, call_id: str) -> str:
        return f"{turn_id}:tool:{call_id}"

    @staticmethod
    def _current_turn_id(handle: _SessionHandle) -> str:
        if handle.current_turn_id is None:
            pending = handle.runtime.session.pending_turn
            if pending is not None:
                handle.current_turn_id = pending.turn_id
        if handle.current_turn_id is None:
            raise RuntimeError("runtime event is not associated with an active turn")
        return handle.current_turn_id

    def _publish(
        self,
        handle: _SessionHandle,
        event_type,
        state: AgentState,
        *,
        turn_id: str | None,
        durable: bool = True,
        **payload,
    ) -> None:
        with handle.lock:
            handle.sequence += 1
            event = event_type(
                session_id=handle.runtime.session.session_id,
                turn_id=turn_id,
                sequence=handle.sequence,
                state=state,
                durable=durable,
                **payload,
            )
            handle.projection.apply(event)
            if durable and self._event_store is not None:
                self._event_store.append(event)
            self._event_broker.publish(event)
