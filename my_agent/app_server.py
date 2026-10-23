from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

from .config import ConfiguredModel
from .control import ApprovalDecision
from .modes import CollaborationMode
from .permissions import PERMISSION_PROFILES, get_permission_profile
from .protocol import (
    AbortTurnCommand,
    AgentCommandType,
    AgentEventType,
    ClearPermissionGrantsCommand,
    ExecutePlanCommand,
    InterruptTurnCommand,
    RenameSessionCommand,
    RevokePermissionGrantCommand,
    ResolveApprovalCommand,
    RevisePlanCommand,
    StartTurnCommand,
    SwitchModeCommand,
    SwitchModelCommand,
    SwitchPermissionProfileCommand,
    SwitchSandboxCommand,
)
from .rpc import RequestEnvelope, ResponseEnvelope
from .runtime import AgentRuntime
from .sandbox import SANDBOX_PROFILES, get_sandbox_profile, sandbox_availability
from .service import AgentService, ServiceSessionView
from .server_instance import ServerInstanceLease
from .session import SessionState, SessionStore, message_to_dict


RuntimeFactory = Callable[
    [SessionState, str | None, str | None, str | None, str | None],
    AgentRuntime,
]
CredentialSetter = Callable[[str, str], Mapping[str, Any] | None]


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
        credentials_path: str | Path | None = None,
        credential_setter: CredentialSetter | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.session_store = session_store
        self.agent_service = agent_service
        self._runtime_factory = runtime_factory
        self._models = tuple(models)
        self._modes = tuple(modes)
        self._credentials_path = (
            Path(credentials_path).expanduser().resolve()
            if credentials_path is not None
            else None
        )
        self._credential_setter = credential_setter
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self.instance_id = uuid4().hex
        self._instance_lease = ServerInstanceLease.acquire(
            self.session_store.root,
            instance_id=self.instance_id,
            workspace=self.workspace,
        )

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

    def next_event(
        self,
        subscription_id: str,
        timeout: float | None = None,
    ) -> AgentEventType:
        return self.agent_service.next_event(subscription_id, timeout=timeout)

    def subscribe_events(self) -> str:
        subscription_id, _ = self.agent_service.subscribe_events()
        return subscription_id

    def unsubscribe_events(self, subscription_id: str) -> bool:
        return self.agent_service.unsubscribe_events(subscription_id)

    def close(self, timeout: float = 5.0) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.agent_service.close(timeout)
        finally:
            self._instance_lease.release()

    def _dispatch(self, request: RequestEnvelope) -> dict[str, Any]:
        handlers = {
            "session.create": self._create_session,
            "session.list": self._list_sessions,
            "session.read": self._read_session,
            "session.resume": self._resume_session,
            "session.close": self._close_session,
            "session.rename": self._rename_session,
            "model.list": self._list_models,
            "provider.credential.set": self._set_provider_credential,
            "mode.list": self._list_modes,
            "permission.list": self._list_permission_profiles,
            "permission.grants.list": self._list_permission_grants,
            "sandbox.list": self._list_sandbox_profiles,
            "context.inspect": self._inspect_context,
            "turn.start": self._start_turn,
            "turn.retry": self._retry_turn,
            "turn.interrupt": self._interrupt_turn,
            "approval.resolve": self._resolve_approval,
            "model.switch": self._switch_model,
            "mode.switch": self._switch_mode,
            "permission.switch": self._switch_permission_profile,
            "permission.grant.revoke": self._revoke_permission_grant,
            "permission.grants.clear": self._clear_permission_grants,
            "sandbox.switch": self._switch_sandbox,
            "plan.execute": self._execute_plan,
            "plan.revise": self._revise_plan,
            "turn.abort": self._abort_turn,
            "events.subscribe": self._subscribe_events,
            "events.unsubscribe": self._unsubscribe_events,
        }
        handler = handlers.get(request.method)
        if handler is None:
            raise _RpcCallError(
                "method_not_found",
                f"unknown RPC method: {request.method}",
            )
        return handler(request.params)

    def _create_session(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(
            params,
            optional={
                "model_ref",
                "mode_id",
                "permission_profile_id",
                "sandbox_mode",
            },
        )
        model_ref = self._optional_string(params, "model_ref")
        mode_id = self._optional_string(params, "mode_id")
        permission_profile_id = self._optional_string(
            params,
            "permission_profile_id",
        )
        sandbox_mode = self._optional_string(params, "sandbox_mode")
        if model_ref is not None and model_ref not in {model.ref for model in self._models}:
            raise ValueError(f"unknown model: {model_ref}")
        if mode_id is not None and mode_id not in {mode.mode_id for mode in self._modes}:
            raise ValueError(f"unknown collaboration mode: {mode_id}")
        if permission_profile_id is not None:
            get_permission_profile(permission_profile_id)
        if sandbox_mode is not None:
            get_sandbox_profile(sandbox_mode)
            availability = sandbox_availability(sandbox_mode)
            if not availability.available:
                raise RuntimeError(
                    f"sandbox mode {sandbox_mode!r} is unavailable: "
                    f"{availability.unavailable_reason}"
                )
        with self._lifecycle_lock:
            state = self.session_store.create(self.workspace)
            if permission_profile_id is not None:
                state.permission_profile_id = permission_profile_id
            if sandbox_mode is not None:
                state.sandbox_mode = sandbox_mode
            self._attach(
                state,
                model_ref=model_ref,
                mode_id=mode_id,
                permission_profile_id=permission_profile_id,
                sandbox_mode=sandbox_mode,
            )
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
        active = self.agent_service.has_session(session_id)
        return {
            "session": self._state_to_dict(
                state,
                active=active,
                runtime_state=(
                    self.agent_service.session_view(session_id).runtime_state
                    if active
                    else "inactive"
                ),
            ),
            "messages": [message_to_dict(message) for message in state.messages],
            "projection": self.agent_service.projection(session_id),
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

    def _rename_session(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params, required={"session_id", "title"})
        return self._dispatch_command(
            RenameSessionCommand(
                session_id=self._required_string(params, "session_id"),
                title=self._required_string(params, "title"),
            )
        )

    def _list_models(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params)
        result: dict[str, Any] = {
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
        if self._credentials_path is not None:
            result["credentials_path"] = str(self._credentials_path)
        return result

    def _set_provider_credential(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params, required={"session_id", "api_key"})
        session_id = self._required_string(params, "session_id")
        api_key = self._required_string(params, "api_key")
        if len(api_key) > 16_384:
            raise ValueError("api_key is too long")
        if self._credential_setter is None:
            raise RuntimeError("provider credential storage is not configured")

        with self._lifecycle_lock:
            self._require_active_session(session_id)
            state = self.session_store.load(session_id)
            self._require_workspace(state)
            provider_id = state.provider_id
            if provider_id is None:
                raise RuntimeError("session has no provider to configure")
            if provider_id not in {model.provider_id for model in self._models}:
                raise ValueError(f"unknown provider: {provider_id}")

            metadata = dict(self._credential_setter(provider_id, api_key) or {})
            self.agent_service.unregister(session_id)
            refreshed_state = self.session_store.load(session_id)
            self._attach(refreshed_state)

        result = {
            "provider_id": provider_id,
            "saved": True,
            "session": self._active_session_to_dict(session_id),
        }
        if self._credentials_path is not None:
            result["credentials_path"] = str(self._credentials_path)
        result.update(metadata)
        return result

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

    def _list_permission_profiles(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params)
        return {
            "profiles": [
                {
                    "profile_id": profile.profile_id,
                    "description": profile.description,
                    "rules": [
                        {
                            "action": rule.action,
                            "resource_pattern": rule.resource_pattern,
                            "effect": rule.effect.value,
                        }
                        for rule in profile.rules
                    ],
                }
                for profile in PERMISSION_PROFILES
            ]
        }

    def _list_permission_grants(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = self._session_id(params)
        self._require_active_session(session_id)
        return {
            "grants": [
                grant.to_dict()
                for grant in self.agent_service.permission_grants(session_id)
            ],
            "lifetime": "runtime",
            "persistent": False,
        }

    def _list_sandbox_profiles(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params)
        profiles = []
        for profile in SANDBOX_PROFILES:
            availability = sandbox_availability(profile.mode)
            profiles.append(
                {
                    "mode": profile.mode,
                    "description": profile.description,
                    "filesystem_read": profile.filesystem_read,
                    "filesystem_write": profile.filesystem_write,
                    "network": profile.network,
                    "dangerous": profile.dangerous,
                    "available": availability.available,
                    "backend": availability.backend,
                    "unavailable_reason": availability.unavailable_reason,
                }
            )
        return {"profiles": profiles}

    def _inspect_context(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = self._session_id(params)
        self._require_active_session(session_id)
        return {"context": self.agent_service.inspect_context(session_id).to_dict()}

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

    def _switch_permission_profile(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params, required={"session_id", "profile_id"})
        profile_id = self._required_string(params, "profile_id")
        get_permission_profile(profile_id)
        return self._dispatch_command(
            SwitchPermissionProfileCommand(
                session_id=self._required_string(params, "session_id"),
                profile_id=profile_id,
            )
        )

    def _revoke_permission_grant(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(
            params,
            required={"session_id", "action", "resource"},
        )
        return self._dispatch_command(
            RevokePermissionGrantCommand(
                session_id=self._required_string(params, "session_id"),
                action=self._required_string(params, "action"),
                resource=self._required_string(params, "resource"),
            )
        )

    def _clear_permission_grants(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._dispatch_command(
            ClearPermissionGrantsCommand(session_id=self._session_id(params))
        )

    def _switch_sandbox(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params, required={"session_id", "mode"})
        mode = self._required_string(params, "mode")
        get_sandbox_profile(mode)
        availability = sandbox_availability(mode)
        if not availability.available:
            raise RuntimeError(
                f"sandbox mode {mode!r} is unavailable: "
                f"{availability.unavailable_reason}"
            )
        return self._dispatch_command(
            SwitchSandboxCommand(
                session_id=self._required_string(params, "session_id"),
                mode=mode,
            )
        )

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

    def _subscribe_events(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params, optional={"session_ids", "after_sequences"})
        raw_session_ids = params.get("session_ids")
        session_ids: tuple[str, ...] | None = None
        if raw_session_ids is not None:
            if not isinstance(raw_session_ids, list) or not raw_session_ids:
                raise ValueError("session_ids must be a non-empty array")
            if any(not isinstance(value, str) or not value for value in raw_session_ids):
                raise ValueError("session_ids must contain non-empty strings")
            session_ids = tuple(raw_session_ids)

        raw_sequences = params.get("after_sequences")
        after_sequences: dict[str, int] | None = None
        if raw_sequences is not None:
            if not isinstance(raw_sequences, Mapping):
                raise ValueError("after_sequences must be an object")
            after_sequences = {}
            for session_id, sequence in raw_sequences.items():
                if not isinstance(session_id, str) or not session_id:
                    raise ValueError("after_sequences keys must be session ids")
                if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
                    raise ValueError("after_sequences values must be non-negative integers")
                after_sequences[session_id] = sequence

        subscription_id, cursors = self.agent_service.subscribe_events(
            session_ids=session_ids,
            after_sequences=after_sequences,
        )
        return {"subscription_id": subscription_id, "cursors": cursors}

    def _unsubscribe_events(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_params(params, required={"subscription_id"})
        subscription_id = self._required_string(params, "subscription_id")
        return {
            "subscription_id": subscription_id,
            "unsubscribed": self.agent_service.unsubscribe_events(subscription_id),
        }

    def _dispatch_command(self, command: AgentCommandType) -> dict[str, Any]:
        self._require_active_session(command.session_id)
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

    def _require_active_session(self, session_id: str) -> None:
        if not self.agent_service.has_session(session_id):
            raise _RpcCallError(
                "session_not_active",
                f"session is not active: {session_id}; call session.resume first",
            )

    def _attach(
        self,
        state: SessionState,
        *,
        model_ref: str | None = None,
        mode_id: str | None = None,
        permission_profile_id: str | None = None,
        sandbox_mode: str | None = None,
    ) -> None:
        self._require_workspace(state)
        runtime = self._runtime_factory(
            state,
            model_ref,
            mode_id,
            permission_profile_id,
            sandbox_mode,
        )
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
    def _state_to_dict(
        state: SessionState,
        *,
        active: bool,
        runtime_state: str = "inactive",
    ) -> dict[str, Any]:
        plan = state.proposed_plan
        return {
            "session_id": state.session_id,
            "title": state.title,
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
            "permission_profile_id": state.permission_profile_id,
            "sandbox_mode": state.sandbox_mode,
            "runtime_state": runtime_state,
            "pending_turn_id": state.pending_turn.turn_id if state.pending_turn else None,
            "pending_turn_error": (
                state.pending_turn.last_error if state.pending_turn else None
            ),
            "pending_turn_retryable": (
                state.pending_turn.last_error_retryable if state.pending_turn else None
            ),
            "pending_turn_error_details": (
                dict(state.pending_turn.last_error_details)
                if state.pending_turn
                else {}
            ),
            "pending_turn_pause_reason": (
                state.pending_turn.pause_reason if state.pending_turn else None
            ),
            "pending_turn_pause_details": (
                dict(state.pending_turn.pause_details) if state.pending_turn else {}
            ),
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
