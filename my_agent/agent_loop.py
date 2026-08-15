from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .context import ContextManager, ContextSnapshot
from .control import (
    ApprovalBroker,
    ApprovalDecision,
    AutoApproveBroker,
    PermissionRequest,
    TurnInterrupted,
    raise_if_cancelled,
)
from .messages import AssistantMessage, Message, ToolCall, ToolResultMessage, UserMessage
from .permissions import PermissionPolicy
from .provider import Provider, ProviderError, ProviderResponse
from .session import PendingTurn, SessionState
from .tools import ToolContext, ToolRegistry, ToolResult
from .trace import TraceRecorder
from .workspace import Workspace


DEFAULT_INSTRUCTIONS = """You are a coding agent operating inside one workspace.
You may inspect files, search the repository, edit files with apply_patch, and run allowlisted
test, check, or build processes with exec_command.
Each apply_patch call must change exactly one workspace-relative file.
Do not modify .git, .my-agent, or paths outside the workspace.
Pass exec_command one structured argv array; shell operators and command strings are unsupported.
After editing, inspect the actual changes with git_diff and run the narrowest relevant validation.
Use read_file when you need the contents of an untracked file reported by git_diff.
Return a concise result only after checking the changes and test evidence.
"""


@dataclass
class TurnResult:
    answer: str
    messages: list[Message]
    steps: int
    stopped_by: str


@dataclass(frozen=True)
class AgentEvent:
    kind: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[AgentEvent], None]


class AgentLoop:
    def __init__(
        self,
        workspace: Workspace,
        provider: Provider,
        tools: ToolRegistry,
        permissions: PermissionPolicy,
        trace: TraceRecorder,
        max_steps: int = 12,
        instructions: str = DEFAULT_INSTRUCTIONS,
        context_manager: ContextManager | None = None,
    ) -> None:
        self.workspace = workspace
        self.provider = provider
        self.tools = tools
        self.permissions = permissions
        self.trace = trace
        self.max_steps = max_steps
        self.instructions = instructions
        self.context_manager = context_manager or ContextManager(
            context_window_tokens=provider.context_window_tokens,
            reserve_output_tokens=provider.max_output_tokens,
        )

    def run_turn(
        self,
        state: SessionState,
        user_input: str,
        event_handler: EventHandler | None = None,
        cancel_event: threading.Event | None = None,
        approvals: ApprovalBroker | None = None,
    ) -> TurnResult:
        if state.workspace != str(self.workspace.root):
            raise ValueError(
                f"session workspace {state.workspace!r} does not match loop workspace "
                f"{str(self.workspace.root)!r}"
            )
        if state.pending_turn is not None:
            raise ValueError("session has a pending turn; use retry_turn() or abandon it")
        turn_number = sum(isinstance(message, UserMessage) for message in state.messages) + 1
        message_start = len(state.messages)
        state.messages.append(UserMessage(content=user_input))
        state.pending_turn = PendingTurn(
            turn_id=f"{state.session_id}:{turn_number}",
            task=user_input,
            message_start=message_start,
        )
        event_context = {
            "session_id": state.session_id,
            "turn_id": state.pending_turn.turn_id,
        }

        self.trace.write(
            "turn_started",
            {
                **event_context,
                "task": user_input,
                "workspace": str(self.workspace.root),
                "max_steps": self.max_steps,
                "messages_before_request": len(state.messages),
            },
        )
        self._emit(
            event_handler,
            AgentEvent(kind="turn_started", data={**event_context, "task": user_input}),
        )

        return self._continue_turn(state, event_handler, cancel_event, approvals)

    def retry_turn(
        self,
        state: SessionState,
        event_handler: EventHandler | None = None,
        cancel_event: threading.Event | None = None,
        approvals: ApprovalBroker | None = None,
    ) -> TurnResult:
        pending = state.pending_turn
        if pending is None:
            raise ValueError("session has no pending turn to retry")
        self.trace.write(
            "turn_resumed",
            {
                "session_id": state.session_id,
                "turn_id": pending.turn_id,
                "step": pending.step,
                "task": pending.task,
            },
        )
        self._emit(
            event_handler,
            AgentEvent(kind="turn_resumed", data={"step": pending.step}),
        )
        return self._continue_turn(state, event_handler, cancel_event, approvals)

    def inspect_context(self, state: SessionState) -> ContextSnapshot:
        return self.context_manager.inspect(
            state=state,
            tools=self.tools.specs(),
            system_prompt=self.instructions,
        )

    def abort_turn(self, state: SessionState) -> PendingTurn:
        pending = state.abandon_pending_turn()
        self.trace.write(
            "turn_aborted",
            {
                "session_id": state.session_id,
                "turn_id": pending.turn_id,
                "step": pending.step,
                "tool_calls_executed": pending.tool_calls_executed,
            },
        )
        return pending

    def _continue_turn(
        self,
        state: SessionState,
        event_handler: EventHandler | None,
        cancel_event: threading.Event | None,
        approvals: ApprovalBroker | None,
    ) -> TurnResult:
        pending = state.pending_turn
        if pending is None:
            raise ValueError("session has no pending turn")
        messages = state.messages
        ctx = ToolContext(
            workspace=self.workspace,
            permissions=self.permissions,
            approvals=approvals or AutoApproveBroker(),
            cancel_event=cancel_event,
        )
        event_context = {
            "session_id": state.session_id,
            "turn_id": pending.turn_id,
        }

        for step in range(pending.step, self.max_steps + 1):
            pending.step = step
            try:
                raise_if_cancelled(cancel_event)
                snapshot = self.context_manager.prepare(
                    state=state,
                    tools=self.tools.specs(),
                    system_prompt=self.instructions,
                )
                if snapshot.compacted_messages:
                    self.trace.write(
                        "context_compacted",
                        {**event_context, "step": step, **snapshot.to_dict()},
                    )
                    self._emit(
                        event_handler,
                        AgentEvent(kind="context_compacted", data=snapshot.to_dict()),
                    )
                self.trace.write(
                    "model_request",
                    {
                        **event_context,
                        "step": step,
                        "messages": len(snapshot.messages),
                        "tools": self.tools.names(),
                        "context": snapshot.to_dict(),
                    },
                )
                reply = self._stream_model_response(
                    snapshot=snapshot,
                    step=step,
                    event_context=event_context,
                    event_handler=event_handler,
                    cancel_event=cancel_event,
                )
            except TurnInterrupted as exc:
                self._record_turn_interrupted(
                    pending,
                    step,
                    event_context,
                    str(exc),
                    event_handler,
                )
                raise
            except KeyboardInterrupt:
                self._record_turn_error(
                    pending,
                    step,
                    event_context,
                    "turn interrupted by user",
                    retryable=True,
                    event_handler=event_handler,
                )
                raise
            except Exception as exc:
                self._record_turn_error(
                    pending,
                    step,
                    event_context,
                    str(exc),
                    retryable=isinstance(exc, ProviderError) and exc.retryable,
                    event_handler=event_handler,
                )
                raise

            if reply.usage is not None:
                state.last_input_tokens = reply.usage.input_tokens
                state.last_output_tokens = reply.usage.output_tokens
            self.trace.write(
                "model_response",
                {
                    **event_context,
                    "step": step,
                    "content": reply.content,
                    "tool_calls": [call.to_dict() for call in reply.tool_calls],
                    "finish_reason": reply.finish_reason,
                    "usage": {
                        "input_tokens": state.last_input_tokens,
                        "output_tokens": state.last_output_tokens,
                    },
                },
            )

            if not reply.tool_calls:
                answer = reply.content.strip() or "(model returned no final content)"
                messages.append(
                    AssistantMessage(
                        content=answer,
                        reasoning_content=reply.reasoning_content,
                    )
                )
                self.trace.write(
                    "final_answer",
                    {
                        **event_context,
                        "step": step,
                        "answer": answer,
                        "finish_reason": reply.finish_reason,
                    },
                )
                state.pending_turn = None
                return TurnResult(
                    answer=answer,
                    messages=list(messages),
                    steps=step,
                    stopped_by=(
                        "model_length"
                        if reply.finish_reason == "length"
                        else "final_answer"
                    ),
                )

            messages.append(
                AssistantMessage(
                    content=reply.content,
                    reasoning_content=reply.reasoning_content,
                    tool_calls=reply.tool_calls,
                )
            )

            for call in reply.tool_calls:
                self.trace.write(
                    "tool_call",
                    {**event_context, "step": step, "call": call.to_dict()},
                )
                self._emit(
                    event_handler,
                    AgentEvent(
                        kind="tool_proposed",
                        data={"name": call.name, "arguments": call.arguments},
                    ),
                )

            for call_index, call in enumerate(reply.tool_calls):
                observation_recorded = False
                try:
                    raise_if_cancelled(cancel_event)
                    call_key = json.dumps(
                        {"name": call.name, "arguments": call.arguments},
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    pending.seen_calls[call_key] = pending.seen_calls.get(call_key, 0) + 1
                    if pending.seen_calls[call_key] > 3:
                        tool_result = ToolResult(
                            ok=False,
                            stderr="repeated identical tool call stopped",
                            metadata={"status": "repeated_call"},
                        )
                    else:
                        request = self.tools.permission_request(call, ctx)
                        tool_result = self._authorize_tool_call(
                            request=request,
                            call_id=call.call_id,
                            call_name=call.name,
                            step=step,
                            event_context=event_context,
                            ctx=ctx,
                            event_handler=event_handler,
                        )
                        if tool_result is None:
                            self._emit(
                                event_handler,
                                AgentEvent(
                                    kind="tool_started",
                                    data={"name": call.name, "arguments": call.arguments},
                                ),
                            )
                            tool_result = self.tools.run(call, ctx)
                            pending.tool_calls_executed += 1

                    result_payload = tool_result.to_dict()
                    self.trace.write(
                        "tool_result",
                        {
                            **event_context,
                            "step": step,
                            "call": call.to_dict(),
                            "result": result_payload,
                        },
                    )
                    self._emit(
                        event_handler,
                        AgentEvent(
                            kind="tool_finished",
                            data={
                                "name": call.name,
                                "ok": tool_result.ok,
                                "status": tool_result.metadata.get("status")
                                or ("success" if tool_result.ok else "failed"),
                                "duration_ms": tool_result.metadata.get("duration_ms"),
                            },
                        ),
                    )
                    messages.append(
                        ToolResultMessage(
                            tool_call_id=call.call_id,
                            tool_name=call.name,
                            content=json.dumps(result_payload, ensure_ascii=False),
                        )
                    )
                    observation_recorded = True
                    raise_if_cancelled(cancel_event)
                except TurnInterrupted as exc:
                    first_unobserved = call_index + int(observation_recorded)
                    for interrupted_call in reply.tool_calls[first_unobserved:]:
                        self._record_interrupted_tool_result(
                            interrupted_call,
                            messages,
                            step,
                            event_context,
                            event_handler,
                        )
                    self._record_turn_interrupted(
                        pending,
                        step,
                        event_context,
                        str(exc),
                        event_handler,
                    )
                    raise
            pending.step = step + 1

        answer = f"Stopped after reaching max_steps={self.max_steps} without a final answer."
        messages.append(AssistantMessage(content=answer))
        self.trace.write(
            "final_answer",
            {**event_context, "step": self.max_steps, "answer": answer},
        )
        state.pending_turn = None
        return TurnResult(
            answer=answer,
            messages=list(messages),
            steps=self.max_steps,
            stopped_by="max_steps",
        )

    def _authorize_tool_call(
        self,
        *,
        request: PermissionRequest | None,
        call_id: str,
        call_name: str,
        step: int,
        event_context: dict[str, str],
        ctx: ToolContext,
        event_handler: EventHandler | None,
    ) -> ToolResult | None:
        if request is None:
            return None

        waited = False

        def on_waiting() -> None:
            nonlocal waited
            waited = True
            payload = {
                **event_context,
                "step": step,
                "call_id": call_id,
                "tool_name": call_name,
                **request.to_dict(),
            }
            self.trace.write("approval_requested", payload)
            self._emit(
                event_handler,
                AgentEvent(kind="approval_requested", data=payload),
            )

        decision = ctx.approvals.authorize(
            request,
            cancel_event=ctx.cancel_event,
            on_waiting=on_waiting,
        )
        payload = {
            **event_context,
            "step": step,
            "call_id": call_id,
            "tool_name": call_name,
            "request_id": request.request_id,
            "action": request.action,
            "resource": request.resource,
            "description": request.description,
            "details": request.details,
            "decision": decision.value,
            "prompted": waited,
            "source": (
                "prompt"
                if waited
                else "session"
                if decision == ApprovalDecision.ALLOW_SESSION
                else "automatic"
            ),
        }
        self.trace.write("approval_resolved", payload)
        self._emit(
            event_handler,
            AgentEvent(kind="approval_resolved", data=payload),
        )
        if decision == ApprovalDecision.DENY:
            return ToolResult(
                ok=False,
                stderr=f"user denied {request.description}",
                metadata={
                    "status": "denied",
                    "request_id": request.request_id,
                    "action": request.action,
                    "resource": request.resource,
                },
            )
        return None

    def _record_interrupted_tool_result(
        self,
        call: ToolCall,
        messages: list[Message],
        step: int,
        event_context: dict[str, str],
        event_handler: EventHandler | None,
    ) -> None:
        tool_result = ToolResult(
            ok=False,
            stderr="tool call cancelled because the turn was interrupted",
            metadata={"status": "interrupted"},
        )
        result_payload = tool_result.to_dict()
        self.trace.write(
            "tool_result",
            {
                **event_context,
                "step": step,
                "call": call.to_dict(),
                "result": result_payload,
            },
        )
        self._emit(
            event_handler,
            AgentEvent(
                kind="tool_finished",
                data={"name": call.name, "ok": False, "status": "interrupted"},
            ),
        )
        messages.append(
            ToolResultMessage(
                tool_call_id=call.call_id,
                tool_name=call.name,
                content=json.dumps(result_payload, ensure_ascii=False),
            )
        )

    def _stream_model_response(
        self,
        *,
        snapshot: ContextSnapshot,
        step: int,
        event_context: dict[str, str],
        event_handler: EventHandler | None,
        cancel_event: threading.Event | None,
    ) -> ProviderResponse:
        raise_if_cancelled(cancel_event)
        self._emit(event_handler, AgentEvent(kind="model_started", data={"step": step}))
        response: ProviderResponse | None = None
        for event in self.provider.stream(
            messages=snapshot.messages,
            tools=self.tools.specs(),
            system_prompt=snapshot.system_prompt,
            cancel_event=cancel_event,
        ):
            raise_if_cancelled(cancel_event)
            if event.kind in {"content_delta", "reasoning_delta"}:
                self._emit(
                    event_handler,
                    AgentEvent(kind=event.kind, text=event.text, data={"step": step}),
                )
            elif event.kind == "retry":
                self.trace.write(
                    "provider_retry",
                    {**event_context, "step": step, "error": event.text, **event.data},
                )
                self._emit(
                    event_handler,
                    AgentEvent(kind="provider_retry", text=event.text, data=event.data),
                )
            elif event.kind == "completed":
                response = event.response
        if response is None:
            raise ProviderError("provider stream completed without a response", retryable=True)
        return response

    def _record_turn_interrupted(
        self,
        pending: PendingTurn,
        step: int,
        event_context: dict[str, str],
        reason: str,
        event_handler: EventHandler | None,
    ) -> None:
        pending.step = step
        self.trace.write(
            "turn_interrupted",
            {**event_context, "step": step, "reason": reason},
        )
        self._emit(
            event_handler,
            AgentEvent(
                kind="turn_interrupted",
                text=reason,
                data={"step": step},
            ),
        )

    def _record_turn_error(
        self,
        pending: PendingTurn,
        step: int,
        event_context: dict[str, str],
        error: str,
        *,
        retryable: bool,
        event_handler: EventHandler | None,
    ) -> None:
        pending.step = step
        self.trace.write(
            "turn_error",
            {
                **event_context,
                "step": step,
                "error": error,
                "retryable": retryable,
            },
        )
        self._emit(
            event_handler,
            AgentEvent(
                kind="turn_error",
                text=error,
                data={"step": step, "retryable": retryable},
            ),
        )

    @staticmethod
    def _emit(event_handler: EventHandler | None, event: AgentEvent) -> None:
        if event_handler is not None:
            event_handler(event)
