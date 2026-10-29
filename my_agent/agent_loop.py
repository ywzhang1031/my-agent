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
from .goals import GoalState, format_goal_context, update_goal
from .messages import AssistantMessage, Message, ToolCall, ToolResultMessage, UserMessage
from .model_invoker import ModelInvoker, RetryScheduled
from .modes import BUILD_MODE, CollaborationMode, extract_proposed_plan
from .permissions import PermissionEffect, PermissionPolicy
from .provider import (
    ModelCompleted,
    ModelError,
    ModelRequest,
    ModelResponse,
    ReasoningDelta,
    StopReason,
    TextDelta,
)
from .session import PendingTurn, ProposedPlan, SessionState
from .sandbox import DEFAULT_SANDBOX_MODE, SandboxPolicy
from .tools import ToolContext, ToolRegistry, ToolResult
from .trace import TraceRecorder
from .workspace import Workspace


@dataclass
class TurnResult:
    answer: str
    messages: list[Message]
    steps: int
    stopped_by: str
    mode_id: str = BUILD_MODE.mode_id
    plan_id: str | None = None


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
        model: ModelInvoker,
        tools: ToolRegistry,
        permissions: PermissionPolicy,
        trace: TraceRecorder,
        max_steps: int = 12,
        mode: CollaborationMode = BUILD_MODE,
        context_manager: ContextManager | None = None,
        sandbox_policy: SandboxPolicy | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.workspace = workspace
        self.model = model
        self.tools = tools
        self.permissions = permissions
        self.trace = trace
        self.max_steps = max_steps
        self.mode = mode
        self.sandbox_policy = sandbox_policy or SandboxPolicy(
            mode=DEFAULT_SANDBOX_MODE,
            workspace=workspace.root,
        )
        self.context_manager = context_manager or ContextManager(
            context_window_tokens=model.profile.context_window_tokens,
            reserve_output_tokens=model.profile.max_output_tokens,
        )

    @property
    def instructions(self) -> str:
        return self.mode.instructions

    def _instructions_for(self, state: SessionState) -> str:
        goal_context = format_goal_context(state.goal)
        if not goal_context:
            return self.instructions
        return f"{self.instructions.rstrip()}\n\n{goal_context}\n"

    def run_turn(
        self,
        state: SessionState,
        user_input: str,
        event_handler: EventHandler | None = None,
        cancel_event: threading.Event | None = None,
        approvals: ApprovalBroker | None = None,
        approved_plan_id: str | None = None,
    ) -> TurnResult:
        if state.workspace != str(self.workspace.root):
            raise ValueError(
                f"session workspace {state.workspace!r} does not match loop workspace "
                f"{str(self.workspace.root)!r}"
            )
        if state.pending_turn is not None:
            raise ValueError("session has a pending turn; use retry_turn() or abandon it")
        if approved_plan_id is not None:
            plan = state.proposed_plan
            if self.mode.mode_id != BUILD_MODE.mode_id:
                raise ValueError("approved plans can only execute in build mode")
            if plan is None or plan.plan_id != approved_plan_id or plan.status != "approved":
                raise ValueError(f"approved plan is not ready to execute: {approved_plan_id}")
        turn_number = sum(isinstance(message, UserMessage) for message in state.messages) + 1
        message_start = len(state.messages)
        state.messages.append(UserMessage(content=user_input))
        state.pending_turn = PendingTurn(
            turn_id=f"{state.session_id}:{turn_number}",
            task=user_input,
            message_start=message_start,
            mode_id=self.mode.mode_id,
            approved_plan_id=approved_plan_id,
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
                "mode_id": self.mode.mode_id,
                "approved_plan_id": approved_plan_id,
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
        if pending.mode_id != self.mode.mode_id:
            raise ValueError(
                f"pending turn requires mode {pending.mode_id}; current mode is {self.mode.mode_id}"
            )
        self.trace.write(
            "turn_resumed",
            {
                "session_id": state.session_id,
                "turn_id": pending.turn_id,
                "step": pending.step,
                "task": pending.task,
                "mode_id": pending.mode_id,
                "approved_plan_id": pending.approved_plan_id,
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
            system_prompt=self._instructions_for(state),
        )

    def switch_model(self, model: ModelInvoker) -> None:
        current = self.context_manager
        self.model = model
        self.context_manager = ContextManager(
            context_window_tokens=model.profile.context_window_tokens,
            reserve_output_tokens=model.profile.max_output_tokens,
            compact_threshold=current.compact_threshold,
            compact_target=current.compact_target,
            summary_tokens=current.summary_tokens,
        )

    def switch_mode(self, mode: CollaborationMode, tools: ToolRegistry) -> None:
        self.mode = mode
        self.tools = tools

    def switch_sandbox(self, mode: str) -> None:
        self.sandbox_policy = SandboxPolicy(mode=mode, workspace=self.workspace.root)

    def clear_goal(self, state: SessionState) -> GoalState | None:
        goal = state.goal
        state.goal = None
        if goal is not None:
            self.trace.write(
                "goal_cleared",
                {
                    "session_id": state.session_id,
                    "goal_id": goal.goal_id,
                },
            )
        return goal

    def approve_plan(self, state: SessionState) -> ProposedPlan:
        if state.pending_turn is not None:
            raise ValueError("cannot approve a plan while a turn is pending")
        plan = state.proposed_plan
        if plan is None or plan.status != "proposed":
            raise ValueError("session has no proposed plan to approve")
        plan.status = "approved"
        self.trace.write(
            "plan_approved",
            {
                "session_id": state.session_id,
                "turn_id": plan.source_turn_id,
                "plan_id": plan.plan_id,
            },
        )
        return plan

    def request_plan_revision(self, state: SessionState) -> ProposedPlan:
        if state.pending_turn is not None:
            raise ValueError("cannot revise a plan while a turn is pending")
        plan = state.proposed_plan
        if plan is None or plan.status != "proposed":
            raise ValueError("session has no proposed plan to revise")
        plan.status = "revising"
        self.trace.write(
            "plan_revision_requested",
            {
                "session_id": state.session_id,
                "turn_id": plan.source_turn_id,
                "plan_id": plan.plan_id,
            },
        )
        return plan

    def abort_turn(self, state: SessionState) -> PendingTurn:
        pending = state.abandon_pending_turn()
        self._finish_plan_execution(
            state,
            pending,
            {"session_id": state.session_id, "turn_id": pending.turn_id},
            step=pending.step,
            successful=False,
            reason="aborted",
        )
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
            sandbox_policy=self.sandbox_policy,
            goal_updater=lambda arguments: self._update_goal(state, arguments),
        )
        event_context = {
            "session_id": state.session_id,
            "turn_id": pending.turn_id,
        }
        batch_start = pending.step
        batch_end = batch_start + self.max_steps - 1
        pending.pause_reason = None
        pending.pause_details = {}

        for step in range(batch_start, batch_end + 1):
            pending.step = step
            try:
                raise_if_cancelled(cancel_event)
                snapshot = self.context_manager.prepare(
                    state=state,
                    tools=self.tools.specs(),
                    system_prompt=self._instructions_for(state),
                )
                if snapshot.compacted_messages:
                    self.trace.write(
                        "context_compacted",
                        {**event_context, "step": step, **snapshot.to_dict()},
                    )
                    self._emit(
                        event_handler,
                        AgentEvent(
                            kind="context_compacted",
                            data={"step": step, **snapshot.to_dict()},
                        ),
                    )
                self.trace.write(
                    "model_request",
                    {
                        **event_context,
                        "step": step,
                        "provider_id": self.model.profile.provider_id,
                        "model_id": self.model.profile.model_id,
                        "messages": len(snapshot.messages),
                        "tools": self.tools.names(),
                        "mode_id": self.mode.mode_id,
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
                error_details = exc.to_dict() if isinstance(exc, ModelError) else {}
                if isinstance(exc, ModelError) and exc.kind.value in {
                    "configuration",
                    "authentication",
                }:
                    error_details["action_required"] = "configure_api_key"
                self._record_turn_error(
                    pending,
                    step,
                    event_context,
                    str(exc),
                    retryable=isinstance(exc, ModelError) and exc.retryable,
                    details=error_details,
                    event_handler=event_handler,
                )
                raise

            if reply.usage is not None:
                if reply.usage.input_tokens is not None:
                    state.last_input_tokens = reply.usage.input_tokens
                if reply.usage.output_tokens is not None:
                    state.last_output_tokens = reply.usage.output_tokens
            response_usage = (
                {
                    "input_tokens": reply.usage.input_tokens,
                    "output_tokens": reply.usage.output_tokens,
                    "total_tokens": reply.usage.total_tokens,
                    "reasoning_tokens": reply.usage.reasoning_tokens,
                    "cache_read_tokens": reply.usage.cache_read_tokens,
                    "cache_write_tokens": reply.usage.cache_write_tokens,
                }
                if reply.usage is not None
                else None
            )
            response_metadata = {
                "provider_id": reply.metadata.provider_id,
                "model_id": reply.metadata.model_id,
                "response_model_id": reply.metadata.response_model_id,
                "response_id": reply.metadata.response_id,
                "request_id": reply.metadata.request_id,
                "native": reply.metadata.native,
            }
            self.trace.write(
                "model_response",
                {
                    **event_context,
                    "step": step,
                    "content": reply.text,
                    "tool_calls": [call.to_dict() for call in reply.tool_calls],
                    "finish_reason": reply.stop_reason.value,
                    "usage": response_usage,
                    "metadata": response_metadata,
                },
            )
            self._emit(
                event_handler,
                AgentEvent(
                    kind="model_completed",
                    data={
                        "step": step,
                        "content": reply.text,
                        "reasoning": reply.reasoning,
                        "finish_reason": reply.stop_reason.value,
                        "tool_calls": [call.to_dict() for call in reply.tool_calls],
                        "usage": response_usage,
                        "metadata": response_metadata,
                    },
                ),
            )

            if not reply.tool_calls:
                answer = reply.text.strip() or "(model returned no final content)"
                stopped_by = (
                    "model_length"
                    if reply.stop_reason is StopReason.LENGTH
                    else "final_answer"
                )
                plan_id = None
                if self.mode.produces_plan and reply.stop_reason is not StopReason.LENGTH:
                    plan_content = extract_proposed_plan(answer)
                    if plan_content is not None:
                        plan_id = f"{pending.turn_id}:plan"
                        state.proposed_plan = ProposedPlan(
                            plan_id=plan_id,
                            source_turn_id=pending.turn_id,
                            content=plan_content,
                        )
                        answer = plan_content
                        stopped_by = "proposed_plan"
                        self.trace.write(
                            "plan_proposed",
                            {
                                **event_context,
                                "step": step,
                                "plan_id": plan_id,
                                "content": plan_content,
                            },
                        )
                messages.append(
                    AssistantMessage(
                        content=answer,
                        reasoning_content=reply.reasoning,
                        provider_data=reply.provider_data,
                    )
                )
                self.trace.write(
                    "final_answer",
                    {
                        **event_context,
                        "step": step,
                        "answer": answer,
                        "finish_reason": reply.stop_reason.value,
                        "mode_id": self.mode.mode_id,
                        "stopped_by": stopped_by,
                        "plan_id": plan_id,
                    },
                )
                if pending.approved_plan_id is not None:
                    self._finish_plan_execution(
                        state,
                        pending,
                        event_context,
                        step=step,
                        successful=stopped_by == "final_answer",
                        reason=stopped_by,
                    )
                state.pending_turn = None
                return TurnResult(
                    answer=answer,
                    messages=list(messages),
                    steps=step,
                    stopped_by=stopped_by,
                    mode_id=self.mode.mode_id,
                    plan_id=plan_id,
                )

            messages.append(
                AssistantMessage(
                    content=reply.text,
                    reasoning_content=reply.reasoning,
                    tool_calls=reply.tool_calls,
                    provider_data=reply.provider_data,
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
                        data={
                            "step": step,
                            "call_id": call.call_id,
                            "name": call.name,
                            "arguments": call.arguments,
                        },
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
                                    data={
                                        "step": step,
                                        "call_id": call.call_id,
                                        "name": call.name,
                                        "arguments": call.arguments,
                                    },
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
                                "step": step,
                                "call_id": call.call_id,
                                "name": call.name,
                                "arguments": call.arguments,
                                "ok": tool_result.ok,
                                "status": tool_result.metadata.get("status")
                                or ("success" if tool_result.ok else "failed"),
                                "duration_ms": tool_result.metadata.get("duration_ms"),
                                "result": result_payload,
                            },
                        ),
                    )
                    goal_payload = tool_result.metadata.get("goal")
                    if (
                        tool_result.ok
                        and tool_result.metadata.get("status") == "goal_updated"
                        and isinstance(goal_payload, dict)
                    ):
                        goal_event = {
                            **event_context,
                            "step": step,
                            "call_id": call.call_id,
                            "goal": goal_payload,
                        }
                        self.trace.write("goal_updated", goal_event)
                        self._emit(
                            event_handler,
                            AgentEvent(kind="goal_updated", data=goal_event),
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

        steps_completed = pending.step - 1
        pause_details = {
            "steps_completed": steps_completed,
            "step_budget": self.max_steps,
            "next_step": pending.step,
        }
        pending.pause_reason = "step_budget"
        pending.pause_details = pause_details
        self.trace.write(
            "turn_continuation_required",
            {**event_context, **pause_details},
        )
        return TurnResult(
            answer="",
            messages=list(messages),
            steps=steps_completed,
            stopped_by="continuation_required",
            mode_id=self.mode.mode_id,
        )

    @staticmethod
    def _update_goal(
        state: SessionState,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        goal = update_goal(state.goal, arguments)
        state.goal = goal
        return goal.to_dict()

    def _finish_plan_execution(
        self,
        state: SessionState,
        pending: PendingTurn,
        event_context: dict[str, str],
        *,
        step: int,
        successful: bool,
        reason: str,
    ) -> None:
        plan = state.proposed_plan
        if (
            pending.approved_plan_id is None
            or plan is None
            or plan.plan_id != pending.approved_plan_id
        ):
            return
        plan.status = "completed" if successful else "proposed"
        self.trace.write(
            "plan_completed" if successful else "plan_execution_incomplete",
            {
                **event_context,
                "step": step,
                "plan_id": plan.plan_id,
                "reason": reason,
            },
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

        policy_effect = ctx.permissions.decide_permission(
            request.action,
            request.resource,
        )
        if policy_effect == PermissionEffect.ALLOW:
            decision = ApprovalDecision.ALLOW_ONCE
            source = f"profile:{ctx.permissions.profile.profile_id}"
        elif policy_effect == PermissionEffect.DENY:
            decision = ApprovalDecision.DENY
            source = f"profile:{ctx.permissions.profile.profile_id}"
        else:
            approval = ctx.approvals.authorize(
                request,
                cancel_event=ctx.cancel_event,
                on_waiting=on_waiting,
            )
            decision = approval.decision
            source = approval.source
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
            "source": source,
        }
        self.trace.write("approval_resolved", payload)
        self._emit(
            event_handler,
            AgentEvent(kind="approval_resolved", data=payload),
        )
        if decision == ApprovalDecision.DENY:
            return ToolResult(
                ok=False,
                stderr=(
                    f"permission profile {ctx.permissions.profile.profile_id} denied "
                    f"{request.description}"
                    if policy_effect == PermissionEffect.DENY
                    else f"approval unavailable for {request.description}"
                    if source == "unavailable"
                    else f"user denied {request.description}"
                ),
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
                data={
                    "step": step,
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "ok": False,
                    "status": "interrupted",
                    "result": result_payload,
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

    def _stream_model_response(
        self,
        *,
        snapshot: ContextSnapshot,
        step: int,
        event_context: dict[str, str],
        event_handler: EventHandler | None,
        cancel_event: threading.Event | None,
    ) -> ModelResponse:
        raise_if_cancelled(cancel_event)
        self._emit(event_handler, AgentEvent(kind="model_started", data={"step": step}))
        response: ModelResponse | None = None
        request = ModelRequest(
            messages=tuple(snapshot.messages),
            tools=tuple(self.tools.specs()),
            system_prompt=snapshot.system_prompt,
        )
        for event in self.model.stream(request, cancel_event=cancel_event):
            raise_if_cancelled(cancel_event)
            if isinstance(event, TextDelta):
                self._emit(
                    event_handler,
                    AgentEvent(kind="content_delta", text=event.text, data={"step": step}),
                )
            elif isinstance(event, ReasoningDelta):
                self._emit(
                    event_handler,
                    AgentEvent(kind="reasoning_delta", text=event.text, data={"step": step}),
                )
            elif isinstance(event, RetryScheduled):
                retry_data = {
                    "attempt": event.next_attempt,
                    "failed_attempt": event.failed_attempt,
                    "delay_seconds": event.delay_seconds,
                    **event.error.to_dict(),
                }
                self.trace.write(
                    "provider_retry",
                    {
                        **event_context,
                        "step": step,
                        "error": str(event.error),
                        **retry_data,
                    },
                )
                self._emit(
                    event_handler,
                    AgentEvent(
                        kind="provider_retry",
                        text=str(event.error),
                        data={"step": step, **retry_data},
                    ),
                )
            elif isinstance(event, ModelCompleted):
                response = event.response
        if response is None:
            raise ModelError("model invocation completed without a response", retryable=True)
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
        details: dict[str, Any] | None = None,
        event_handler: EventHandler | None,
    ) -> None:
        error_details = dict(details or {})
        pending.step = step
        pending.last_error = error
        pending.last_error_retryable = retryable
        pending.last_error_details = error_details
        self.trace.write(
            "turn_error",
            {
                **event_context,
                "step": step,
                "error": error,
                "retryable": retryable,
                "details": error_details,
            },
        )
        self._emit(
            event_handler,
            AgentEvent(
                kind="turn_error",
                text=error,
                data={
                    "step": step,
                    "retryable": retryable,
                    "details": error_details,
                },
            ),
        )

    @staticmethod
    def _emit(event_handler: EventHandler | None, event: AgentEvent) -> None:
        if event_handler is not None:
            event_handler(event)
