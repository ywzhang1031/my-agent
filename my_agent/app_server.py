from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import ConfiguredModel
from .control import ApprovalDecision
from .modes import CollaborationMode
from .protocol import (
    AbortTurnCommand,
    AgentCommandType,
    AgentEventType,
    ExecutePlanCommand,
    InterruptTurnCommand,
    ReplayEventsCommand,
    ResolveApprovalCommand,
    RevisePlanCommand,
    StartTurnCommand,
    SwitchModeCommand,
    SwitchModelCommand,
)
from .rpc import RequestEnvelope, ResponseEnvelope
from .runtime import AgentRuntime
from .service import AgentService, ServiceSessionView
from .session import SessionState, SessionStore, message_to_dict


RuntimeFactory = Callable[[SessionState, str | None, str | None], AgentRuntime]


class _RpcCallError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.data = data or {}


class AppServer:
    """Multi-session application service behind local or remote transports."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        session_store: SessionStore,
        agent_service: AgentService,
        runtime_factory: RuntimeFactory,
        models: Iterable[ConfiguredModel],
        modes: Iterable[CollaborationMode],
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.session_store = session_store
        self.agent_service = agent_service
        self._runtime_factory = runtime_factory
        self._models = tuple(models)
        self._modes = tuple(modes)
        self._lifecycle_lock = threading.Lock()

    def handle(self, request: RequestEnvelope) -> ResponseEnvelope:
        try:
            result = self._dispatch(request)
        except _RpcCallError as exc:
            return ResponseEnvelope.failure(
                request.request_id,
                code=exc.code,
                message=str(exc),
                data=exc.data,
            )
        except FileNotFoundError as exc:
            return ResponseEnvelope.failure(
                request.request_id,
                code="not_found",
                message=str(exc),
            )
        except ValueError as exc:
            return ResponseEnvelope.failure(
                request.request_id,
                code="invalid_params",
                message=str(exc),
            )
        except RuntimeError as exc:
            return ResponseEnvelope.failure(
                request.request_id,
                code="conflict",
                message=str(exc),
            )
        except Exception as exc:
            return ResponseEnvelope.failure(
                request.request_id,
                code="internal_error",
                message=str(exc),
            )
        return ResponseEnvelope.success(request.request_id, result)

    def next_event(self, timeout: float | None = None) -> AgentEventType:
        return self.agent_service.next_event(timeout=timeout)

    def close(self, timeout: float = 5.0) -> None:
        self.agent_service.close(timeout)

    def _dispatch(self, request: RequestEnvelope) -> dict[str, Any]:
        handlers = {
            "session.create": self._create_session,
            "session.list": self._list_sessions,
            "session.read": self._read_session,
            "session.resume": self._resume_session,
            "session.close": self._close_session,
            "model.list": self._list_models,
            "mode.list": self._list_modes,
            "turn.start": self._start_turn,
            "turn.retry": self._retry_turn,
            "turn.interrupt": self._interrupt_turn,
            "approval.resolve": self._resolve_approval,
            "model.switch": self._switch_model,
            "mode.switch": self._switch_mode,
            "plan.execute": self._execute_plan,
            "plan.revise": self._revise_plan,
            "turn.abort": self._abort_turn,
            "events.replay": self._replay_events,
        }
        handler = handlers.get(request.method)
        if handler is None:
            raise _RpcCallError(
                "method_not_found",
                f"unknown RPC method: {request.method}",
            )
        return handler(request.params)

    def _create_session(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params, optional={"model_ref", "mode_id"})
        model_ref = self._optional_string(params, "model_ref")
        mode_id = self._optional_string(params, "mode_id")
        if model_ref is not None and model_ref not in {model.ref for model in self._models}:
            raise ValueError(f"unknown model: {model_ref}")
        if mode_id is not None and mode_id not in {mode.mode_id for mode in self._modes}:
            raise ValueError(f"unknown collaboration mode: {mode_id}")
        with self._lifecycle_lock:
            state = self.session_store.create(self.workspace)
            self._attach(state, model_ref=model_ref, mode_id=mode_id)
        return {"session": self._active_session_to_dict(state.session_id)}

    def _list_sessions(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params)
        sessions = []
        for summary in self.session_store.list_sessions():
            if Path(summary.workspace).expanduser().resolve() != self.workspace:
                continue
            item = asdict(summary)
            item["active"] = self.agent_service.has_session(summary.session_id)
            sessions.append(item)
        return {"sessions": sessions}

    def _read_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = self._session_id(params)
        state = self.session_store.load(session_id)
        self._require_workspace(state)
        return {
            "session": self._state_to_dict(
                state,
                active=self.agent_service.has_session(session_id),
            ),
            "messages": [message_to_dict(message) for message in state.messages],
        }

    def _resume_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = self._session_id(params)
        with self._lifecycle_lock:
            if not self.agent_service.has_session(session_id):
                state = self.session_store.load(session_id)
                self._require_workspace(state)
                self._attach(state)
        return {"session": self._active_session_to_dict(session_id)}

    def _close_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = self._session_id(params)
        with self._lifecycle_lock:
            state = self.session_store.load(session_id)
            self._require_workspace(state)
            active = self.agent_service.has_session(session_id)
            if active:
                self.agent_service.unregister(session_id)
        return {"session_id": session_id, "closed": active}

    def _list_models(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params)
        return {
            "models": [
                {
                    "ref": model.ref,
                    "provider_id": model.provider_id,
                    "model_id": model.model_id,
                    "description": model.description,
                    "context_window_tokens": model.context_window_tokens,
                    "max_output_tokens": model.max_output_tokens,
                }
                for model in self._models
            ]
        }

    def _list_modes(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params)
        return {
            "modes": [
                {
                    "mode_id": mode.mode_id,
                    "description": mode.description,
                    "tool_names": list(mode.tool_names),
                    "produces_plan": mode.produces_plan,
                }
                for mode in self._modes
            ]
        }

    def _start_turn(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params, required={"session_id", "user_input"})
        command = StartTurnCommand(
            session_id=self._required_string(params, "session_id"),
            user_input=self._required_string(params, "user_input"),
        )
        return self._dispatch_command(command)

    def _retry_turn(self, params: dict[str, Any]) -> dict[str, Any]:
        command = StartTurnCommand(session_id=self._session_id(params), retry=True)
        return self._dispatch_command(command)

    def _interrupt_turn(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._dispatch_command(
            InterruptTurnCommand(session_id=self._session_id(params))
        )

    def _resolve_approval(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(
            params,
            required={"session_id", "approval_request_id", "decision"},
        )
        command = ResolveApprovalCommand(
            session_id=self._required_string(params, "session_id"),
            request_id=self._required_string(params, "approval_request_id"),
            decision=ApprovalDecision(self._required_string(params, "decision")),
        )
        return self._dispatch_command(command)

    def _switch_model(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params, required={"session_id", "model_ref"})
        command = SwitchModelCommand(
            session_id=self._required_string(params, "session_id"),
            model_ref=self._required_string(params, "model_ref"),
        )
        return self._dispatch_command(command)

    def _switch_mode(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params, required={"session_id", "mode_id"})
        command = SwitchModeCommand(
            session_id=self._required_string(params, "session_id"),
            mode_id=self._required_string(params, "mode_id"),
        )
        return self._dispatch_command(command)

    def _execute_plan(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._dispatch_command(
            ExecutePlanCommand(session_id=self._session_id(params))
        )

    def _revise_plan(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._dispatch_command(
            RevisePlanCommand(session_id=self._session_id(params))
        )

    def _abort_turn(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._dispatch_command(
            AbortTurnCommand(session_id=self._session_id(params))
        )

    def _replay_events(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(
            params,
            required={"session_id"},
            optional={"after_sequence"},
        )
        after_sequence = params.get("after_sequence", 0)
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise ValueError("after_sequence must be an integer")
        command = ReplayEventsCommand(
            session_id=self._required_string(params, "session_id"),
            after_sequence=after_sequence,
        )
        return self._dispatch_command(command)

    def _dispatch_command(self, command: AgentCommandType) -> dict[str, Any]:
        if not self.agent_service.has_session(command.session_id):
            raise _RpcCallError(
                "session_not_active",
                f"session is not active: {command.session_id}; call session.resume first",
            )
        try:
            self.agent_service.dispatch(command)
        except ValueError as exc:
            if not self.agent_service.has_session(command.session_id):
                raise _RpcCallError(
                    "session_not_active",
                    f"session is not active: {command.session_id}; call session.resume first",
                ) from exc
            raise
        return {
            "accepted": True,
            "command_id": command.command_id,
            "session_id": command.session_id,
        }

    def _attach(
        self,
        state: SessionState,
        *,
        model_ref: str | None = None,
        mode_id: str | None = None,
    ) -> None:
        self._require_workspace(state)
        runtime = self._runtime_factory(state, model_ref, mode_id)
        self.agent_service.register(runtime)
        self.session_store.save(state)

    def _active_session_to_dict(self, session_id: str) -> dict[str, Any]:
        return self._view_to_dict(self.agent_service.session_view(session_id))

    @staticmethod
    def _view_to_dict(view: ServiceSessionView) -> dict[str, Any]:
        data = asdict(view)
        data["active"] = True
        return data

    @staticmethod
    def _state_to_dict(state: SessionState, *, active: bool) -> dict[str, Any]:
        plan = state.proposed_plan
        return {
            "session_id": state.session_id,
            "workspace": state.workspace,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "message_count": len(state.messages),
            "summarized_message_count": state.summarized_message_count,
            "last_input_tokens": state.last_input_tokens,
            "last_output_tokens": state.last_output_tokens,
            "provider_id": state.provider_id,
            "model_id": state.model_id,
            "mode_id": state.mode_id,
            "pending_turn_id": state.pending_turn.turn_id if state.pending_turn else None,
            "proposed_plan_id": plan.plan_id if plan else None,
            "proposed_plan_status": plan.status if plan else None,
            "active": active,
        }

    def _require_workspace(self, state: SessionState) -> None:
        if Path(state.workspace).expanduser().resolve() != self.workspace:
            raise _RpcCallError(
                "workspace_mismatch",
                f"session {state.session_id} belongs to a different workspace",
                data={
                    "session_workspace": state.workspace,
                    "server_workspace": str(self.workspace),
                },
            )

    def _session_id(self, params: dict[str, Any]) -> str:
        self._check_params(params, required={"session_id"})
        return self._required_string(params, "session_id")

    @staticmethod
    def _check_params(
        params: dict[str, Any],
        *,
        required: set[str] | None = None,
        optional: set[str] | None = None,
    ) -> None:
        required = required or set()
        allowed = required | (optional or set())
        missing = required - set(params)
        if missing:
            raise ValueError(f"missing RPC params: {', '.join(sorted(missing))}")
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unknown RPC params: {', '.join(sorted(unknown))}")

    @staticmethod
    def _required_string(params: dict[str, Any], name: str) -> str:
        value = params.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    @classmethod
    def _optional_string(cls, params: dict[str, Any], name: str) -> str | None:
        if name not in params or params[name] is None:
            return None
        return cls._required_string(params, name)
