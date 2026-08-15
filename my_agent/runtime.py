from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .agent_loop import AgentEvent, AgentLoop, TurnResult
from .control import (
    ApprovalDecision,
    ApprovalNotifier,
    AutoApproveBroker,
    PermissionRequest,
    TurnInterrupted,
    raise_if_cancelled,
)
from .session import SessionState


class RuntimeState(str, Enum):
    IDLE = "idle"
    RUNNING_MODEL = "running_model"
    WAITING_APPROVAL = "waiting_approval"
    RUNNING_TOOL = "running_tool"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class StartTurn:
    user_input: str = ""
    retry: bool = False


@dataclass(frozen=True)
class InterruptTurn:
    pass


@dataclass(frozen=True)
class ResolveApproval:
    request_id: str
    decision: ApprovalDecision


RuntimeCommand = StartTurn | InterruptTurn | ResolveApproval


@dataclass(frozen=True)
class RuntimeEvent:
    kind: str
    state: RuntimeState
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    result: TurnResult | None = None


@dataclass
class _PendingApproval:
    request: PermissionRequest
    ready: threading.Event = field(default_factory=threading.Event)
    decision: ApprovalDecision | None = None


class RuntimeApprovalBroker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: _PendingApproval | None = None
        self._session_approvals: set[tuple[str, str]] = set()

    def authorize(
        self,
        request: PermissionRequest,
        *,
        cancel_event: threading.Event | None,
        on_waiting: ApprovalNotifier,
    ) -> ApprovalDecision:
        raise_if_cancelled(cancel_event)
        with self._lock:
            if request.scope_key in self._session_approvals:
                return ApprovalDecision.ALLOW_SESSION
            if self._pending is not None:
                raise RuntimeError("another approval request is already pending")
            pending = _PendingApproval(request=request)
            self._pending = pending

        try:
            on_waiting()
            while not pending.ready.wait(0.05):
                raise_if_cancelled(cancel_event)
            raise_if_cancelled(cancel_event)
        except Exception:
            with self._lock:
                if self._pending is pending:
                    self._pending = None
            raise

        with self._lock:
            decision = pending.decision
            if self._pending is pending:
                self._pending = None
            if decision == ApprovalDecision.ALLOW_SESSION:
                self._session_approvals.add(request.scope_key)
        if decision is None:
            raise RuntimeError("approval resolved without a decision")
        return decision

    def resolve(self, request_id: str, decision: ApprovalDecision) -> None:
        if not isinstance(decision, ApprovalDecision):
            raise TypeError("approval decision must be an ApprovalDecision")
        with self._lock:
            pending = self._pending
            if pending is None or pending.request.request_id != request_id:
                raise ValueError(f"approval request is not pending: {request_id}")
            pending.decision = decision
            pending.ready.set()


class AgentRuntime:
    def __init__(
        self,
        loop: AgentLoop,
        session: SessionState,
        *,
        interactive_approvals: bool,
    ) -> None:
        self.loop = loop
        self.session = session
        self._events: queue.Queue[RuntimeEvent] = queue.Queue()
        self._lock = threading.Lock()
        self._state = RuntimeState.IDLE
        self._worker: threading.Thread | None = None
        self._cancel_event: threading.Event | None = None
        self._approvals = (
            RuntimeApprovalBroker() if interactive_approvals else AutoApproveBroker()
        )

    @property
    def state(self) -> RuntimeState:
        with self._lock:
            return self._state

    def send(self, command: RuntimeCommand) -> None:
        if isinstance(command, StartTurn):
            self._start_turn(command)
            return
        if isinstance(command, InterruptTurn):
            with self._lock:
                cancel_event = self._cancel_event
            if cancel_event is not None:
                cancel_event.set()
            return
        if isinstance(command, ResolveApproval):
            if not isinstance(self._approvals, RuntimeApprovalBroker):
                raise RuntimeError("runtime does not accept interactive approvals")
            self._approvals.resolve(command.request_id, command.decision)
            return
        raise TypeError(f"unsupported runtime command: {type(command).__name__}")

    def next_event(self, timeout: float | None = None) -> RuntimeEvent:
        return self._events.get(timeout=timeout)

    def wait(self, timeout: float = 5.0) -> None:
        with self._lock:
            worker = self._worker
        if worker is None:
            return
        worker.join(timeout)
        if worker.is_alive():
            raise RuntimeError("runtime worker did not finish")

    def close(self, timeout: float = 5.0) -> None:
        with self._lock:
            worker = self._worker
            cancel_event = self._cancel_event
        if worker is None or not worker.is_alive():
            return
        if cancel_event is not None:
            cancel_event.set()
        self.wait(timeout)

    def _start_turn(self, command: StartTurn) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("runtime already has an active turn")
            if command.retry:
                if self.session.pending_turn is None:
                    raise ValueError("session has no pending turn to retry")
            elif not command.user_input.strip():
                raise ValueError("user_input is required for a new turn")
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            self._state = RuntimeState.RUNNING_MODEL
            self._worker = threading.Thread(
                target=self._run_turn,
                args=(command, cancel_event),
                name=f"agent-turn-{self.session.session_id}",
                daemon=True,
            )
            self._worker.start()

    def _run_turn(self, command: StartTurn, cancel_event: threading.Event) -> None:
        terminal_kind: str
        terminal_state: RuntimeState
        terminal_text: str
        terminal_data: dict[str, Any] = {}
        terminal_result: TurnResult | None = None
        try:
            if command.retry:
                result = self.loop.retry_turn(
                    self.session,
                    event_handler=self._forward_agent_event,
                    cancel_event=cancel_event,
                    approvals=self._approvals,
                )
            else:
                result = self.loop.run_turn(
                    self.session,
                    command.user_input,
                    event_handler=self._forward_agent_event,
                    cancel_event=cancel_event,
                    approvals=self._approvals,
                )
        except TurnInterrupted as exc:
            terminal_kind = "turn_interrupted"
            terminal_state = RuntimeState.INTERRUPTED
            terminal_text = str(exc)
        except Exception as exc:
            terminal_kind = "turn_failed"
            terminal_state = RuntimeState.FAILED
            terminal_text = str(exc)
        else:
            terminal_kind = "turn_completed"
            terminal_state = RuntimeState.COMPLETED
            terminal_text = result.answer
            terminal_data = {"steps": result.steps, "stopped_by": result.stopped_by}
            terminal_result = result
        finally:
            with self._lock:
                self._cancel_event = None
        self._publish(
            terminal_kind,
            terminal_state,
            text=terminal_text,
            data=terminal_data,
            result=terminal_result,
        )

    def _forward_agent_event(self, event: AgentEvent) -> None:
        if event.kind == "turn_interrupted":
            return
        state = {
            "model_started": RuntimeState.RUNNING_MODEL,
            "approval_requested": RuntimeState.WAITING_APPROVAL,
            "approval_resolved": RuntimeState.RUNNING_TOOL,
            "tool_started": RuntimeState.RUNNING_TOOL,
        }.get(event.kind)
        if state is None:
            state = self.state
        self._publish(event.kind, state, text=event.text, data=event.data)

    def _publish(
        self,
        kind: str,
        state: RuntimeState,
        *,
        text: str = "",
        data: dict[str, Any] | None = None,
        result: TurnResult | None = None,
    ) -> None:
        with self._lock:
            self._state = state
        self._events.put(
            RuntimeEvent(
                kind=kind,
                state=state,
                text=text,
                data=data or {},
                result=result,
            )
        )
