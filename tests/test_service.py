from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from pathlib import Path

from my_agent.agent_loop import TurnResult
from my_agent.control import ApprovalDecision, ApprovalGrant
from my_agent.goals import GoalState, GoalStep
from my_agent.modes import BUILD_MODE, PLAN_MODE
from my_agent.permissions import PermissionPolicy
from my_agent.protocol import (
    AgentState,
    ApprovalResolvedEvent,
    ClearPermissionGrantsCommand,
    ClearGoalCommand,
    ExecutePlanCommand,
    GoalClearedEvent,
    GoalUpdatedEvent,
    ItemCompletedEvent,
    ItemDeltaEvent,
    RenameSessionCommand,
    ResolveApprovalCommand,
    SandboxChangedEvent,
    SessionRenamedEvent,
    StartTurnCommand,
    SwitchModeCommand,
    SwitchPermissionProfileCommand,
    SwitchSandboxCommand,
    PermissionProfileChangedEvent,
    PermissionGrantsChangedEvent,
    RevokePermissionGrantCommand,
    TurnCompletedEvent,
    TurnContinuationRequiredEvent,
    command_from_dict,
    command_to_dict,
    event_from_dict,
    event_to_dict,
)
from my_agent.runtime import RuntimeEvent, RuntimeState, StartTurn
from my_agent.sandbox import DANGER_FULL_ACCESS, WORKSPACE_WRITE, SandboxPolicy
from my_agent.service import AgentService
from my_agent.event_store import EventStore
from my_agent.session import PendingTurn, ProposedPlan, SessionState


def _capture_error(errors, action) -> None:
    try:
        action()
    except Exception as exc:
        errors.append(exc)


class _Profile:
    provider_id = "test"
    model_id = "test-model"


class _Model:
    profile = _Profile()


class _Loop:
    def __init__(self, workspace: Path) -> None:
        self.model = _Model()
        self.mode = PLAN_MODE
        self.tools = object()
        self.permissions = PermissionPolicy()
        self.sandbox_policy = SandboxPolicy(WORKSPACE_WRITE, workspace)

    def switch_mode(self, mode, tools) -> None:
        self.mode = mode
        self.tools = tools

    def approve_plan(self, state: SessionState) -> ProposedPlan:
        if state.proposed_plan is None:
            raise ValueError("missing plan")
        state.proposed_plan.status = "approved"
        return state.proposed_plan


class _Runtime:
    def __init__(self, state: SessionState) -> None:
        self.session = state
        self.loop = _Loop(Path(state.workspace))
        self.state = RuntimeState.IDLE
        self.events: queue.Queue[RuntimeEvent] = queue.Queue()
        self.commands = []
        self.closed = False
        self.grants: list[ApprovalGrant] = []

    def send(self, command) -> None:
        self.commands.append(command)

    def next_event(self, timeout=None) -> RuntimeEvent:
        return self.events.get(timeout=timeout)

    def switch_mode(self, mode, tools) -> None:
        self.loop.switch_mode(mode, tools)

    def switch_permission_profile(self, profile_id: str) -> None:
        self.loop.permissions.switch_profile(profile_id)

    def switch_sandbox(self, mode: str) -> None:
        self.loop.sandbox_policy = SandboxPolicy(mode, Path(self.session.workspace))

    def permission_grants(self) -> tuple[ApprovalGrant, ...]:
        return tuple(self.grants)

    def revoke_permission_grant(self, action: str, resource: str) -> bool:
        before = len(self.grants)
        self.grants = [
            grant
            for grant in self.grants
            if (grant.action, grant.resource) != (action, resource)
        ]
        return len(self.grants) != before

    def clear_permission_grants(self) -> int:
        removed = len(self.grants)
        self.grants.clear()
        return removed

    def clear_goal(self) -> str | None:
        goal = self.session.goal
        self.session.goal = None
        return goal.goal_id if goal is not None else None

    def wait(self, timeout=5.0) -> None:
        return None

    def close(self, timeout=5.0) -> None:
        self.closed = True


class ProtocolTests(unittest.TestCase):
    def test_command_round_trip_preserves_typed_payload(self) -> None:
        command = ResolveApprovalCommand(
            session_id="session-1",
            request_id="approval-1",
            decision=ApprovalDecision.ALLOW_SESSION,
        )

        restored = command_from_dict(command_to_dict(command))

        self.assertEqual(restored, command)

    def test_session_rename_protocol_round_trip(self) -> None:
        command = RenameSessionCommand(
            session_id="session-1",
            command_id="command-1",
            title="Provider adapter work",
        )
        event = SessionRenamedEvent(
            session_id="session-1",
            turn_id=None,
            sequence=4,
            state=AgentState.IDLE,
            event_id="event-1",
            title="Provider adapter work",
        )

        self.assertEqual(command_from_dict(command_to_dict(command)), command)
        self.assertEqual(event_from_dict(event_to_dict(event)), event)

    def test_sandbox_protocol_round_trip(self) -> None:
        command = SwitchSandboxCommand(
            session_id="session-1",
            command_id="command-1",
            mode=DANGER_FULL_ACCESS,
        )
        event = SandboxChangedEvent(
            session_id="session-1",
            turn_id=None,
            sequence=5,
            state=AgentState.IDLE,
            event_id="event-1",
            mode=DANGER_FULL_ACCESS,
        )

        self.assertEqual(command_from_dict(command_to_dict(command)), command)
        self.assertEqual(event_from_dict(event_to_dict(event)), event)

    def test_permission_grant_protocol_round_trip(self) -> None:
        command = RevokePermissionGrantCommand(
            session_id="session-1",
            command_id="command-1",
            action="edit",
            resource="app.py",
        )
        event = PermissionGrantsChangedEvent(
            session_id="session-1",
            turn_id=None,
            sequence=6,
            state=AgentState.IDLE,
            durable=False,
            event_id="event-1",
            grants=[
                {
                    "action": "shell",
                    "resource": "pytest",
                    "description": "Run pytest",
                }
            ],
            reason="revoked",
            removed_count=1,
        )

        self.assertEqual(command_from_dict(command_to_dict(command)), command)
        self.assertEqual(event_from_dict(event_to_dict(event)), event)

    def test_goal_protocol_round_trip(self) -> None:
        command = ClearGoalCommand(
            session_id="session-1",
            command_id="command-1",
        )
        event = GoalUpdatedEvent(
            session_id="session-1",
            turn_id="session-1:1",
            sequence=7,
            state=AgentState.RUNNING_TOOL,
            event_id="event-1",
            goal={
                "goal_id": "goal-1",
                "objective": "Update app.py",
                "steps": [
                    {"description": "Edit app.py", "status": "in_progress"}
                ],
                "status": "active",
                "updated_at": "2026-10-24T00:00:00+00:00",
            },
        )

        self.assertEqual(command_from_dict(command_to_dict(command)), command)
        self.assertEqual(event_from_dict(event_to_dict(event)), event)

    def test_event_round_trip_preserves_typed_payload(self) -> None:
        event = ItemDeltaEvent(
            session_id="session-1",
            turn_id="session-1:1",
            sequence=3,
            state=AgentState.RUNNING_MODEL,
            durable=False,
            item_id="session-1:1:step:2:assistant_message",
            item_type="assistant_message",
            step=2,
            delta="hello",
        )

        restored = event_from_dict(event_to_dict(event))

        self.assertEqual(restored, event)

    def test_continuation_required_event_round_trip(self) -> None:
        event = TurnContinuationRequiredEvent(
            session_id="session-1",
            turn_id="session-1:1",
            sequence=4,
            state=AgentState.WAITING_USER,
            steps_completed=12,
            step_budget=12,
            next_step=13,
        )

        self.assertEqual(event_from_dict(event_to_dict(event)), event)


class AgentServiceTests(unittest.TestCase):
    @staticmethod
    def _subscribe(service: AgentService, session_id: str) -> str:
        subscription_id, _ = service.subscribe_events(session_ids=(session_id,))
        return subscription_id

    @staticmethod
    def _next_terminal(
        service: AgentService,
        subscription_id: str,
    ) -> tuple[list, TurnCompletedEvent]:
        events = []
        while True:
            event = service.next_event(subscription_id, timeout=1)
            events.append(event)
            if isinstance(event, TurnCompletedEvent):
                return events, event

    def _state(self, root: Path) -> SessionState:
        return SessionState(
            session_id="session-1",
            workspace=str(root),
            session_dir=root / ".my-agent" / "sessions" / "session-1",
            created_at="2026-08-20T00:00:00+00:00",
            updated_at="2026-08-20T00:00:00+00:00",
            provider_id="test",
            model_id="test-model",
            mode_id=PLAN_MODE.mode_id,
        )

    def test_service_emits_sequenced_typed_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp))
            runtime = _Runtime(state)
            runtime.events.put(
                RuntimeEvent(
                    kind="turn_started",
                    state=RuntimeState.RUNNING_MODEL,
                    data={"turn_id": "session-1:1", "task": "hi"},
                )
            )
            runtime.events.put(
                RuntimeEvent(
                    kind="content_delta",
                    state=RuntimeState.RUNNING_MODEL,
                    text="hello",
                    data={"step": 1},
                )
            )
            runtime.events.put(
                RuntimeEvent(
                    kind="model_completed",
                    state=RuntimeState.RUNNING_MODEL,
                    data={
                        "step": 1,
                        "content": "hello",
                        "reasoning": "",
                        "finish_reason": "stop",
                        "usage": None,
                        "metadata": {},
                    },
                )
            )
            runtime.events.put(
                RuntimeEvent(
                    kind="turn_completed",
                    state=RuntimeState.COMPLETED,
                    text="hello",
                    data={
                        "steps": 1,
                        "stopped_by": "final_answer",
                        "mode_id": "build",
                        "plan_id": None,
                    },
                    result=TurnResult(
                        answer="hello",
                        messages=[],
                        steps=1,
                        stopped_by="final_answer",
                    ),
                )
            )
            service = AgentService(tool_registry_factory=lambda mode: object())
            service.register(runtime)
            subscription_id = self._subscribe(service, state.session_id)

            service.dispatch(StartTurnCommand(session_id=state.session_id, user_input="hi"))
            events, terminal = self._next_terminal(service, subscription_id)

            delta = next(event for event in events if isinstance(event, ItemDeltaEvent))
            completed = next(
                event
                for event in events
                if isinstance(event, ItemCompletedEvent)
                and event.item_type == "assistant_message"
            )
            self.assertEqual(delta.delta, "hello")
            self.assertFalse(delta.durable)
            self.assertEqual(completed.content, "hello")
            self.assertEqual(terminal.answer, "hello")
            self.assertIsInstance(runtime.commands[0], StartTurn)
            service.close()

    def test_allow_session_emits_ephemeral_grant_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp))
            runtime = _Runtime(state)
            runtime.grants = [ApprovalGrant("apply_patch", "app.py", "update app.py")]
            runtime.events.put(
                RuntimeEvent(
                    kind="turn_started",
                    state=RuntimeState.RUNNING_MODEL,
                    data={"turn_id": "session-1:1", "task": "update app.py"},
                )
            )
            runtime.events.put(
                RuntimeEvent(
                    kind="approval_resolved",
                    state=RuntimeState.RUNNING_TOOL,
                    data={
                        "step": 1,
                        "call_id": "call-1",
                        "tool_name": "apply_patch",
                        "request_id": "approval-1",
                        "decision": ApprovalDecision.ALLOW_SESSION.value,
                        "prompted": True,
                        "source": "prompt",
                    },
                )
            )
            runtime.events.put(
                RuntimeEvent(
                    kind="turn_completed",
                    state=RuntimeState.COMPLETED,
                    text="done",
                    data={
                        "steps": 1,
                        "stopped_by": "final_answer",
                        "mode_id": "build",
                        "plan_id": None,
                    },
                    result=TurnResult(
                        answer="done",
                        messages=[],
                        steps=1,
                        stopped_by="final_answer",
                    ),
                )
            )
            service = AgentService(tool_registry_factory=lambda mode: object())
            service.register(runtime)
            subscription_id = self._subscribe(service, state.session_id)

            service.dispatch(
                StartTurnCommand(session_id=state.session_id, user_input="update app.py")
            )
            events, _ = self._next_terminal(service, subscription_id)

            resolved_index = next(
                index
                for index, event in enumerate(events)
                if isinstance(event, ApprovalResolvedEvent)
            )
            grant_index = next(
                index
                for index, event in enumerate(events)
                if isinstance(event, PermissionGrantsChangedEvent)
            )
            grant_event = events[grant_index]
            self.assertGreater(grant_index, resolved_index)
            self.assertFalse(grant_event.durable)
            self.assertEqual(grant_event.reason, "granted")
            self.assertEqual(grant_event.grants, [runtime.grants[0].to_dict()])
            service.close()

    def test_permission_profile_switch_is_persisted_and_emits_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp))
            runtime = _Runtime(state)
            saved = []
            service = AgentService(
                tool_registry_factory=lambda mode: object(),
                save_session=lambda session: saved.append(session.permission_profile_id),
            )
            service.register(runtime)
            subscription_id = self._subscribe(service, state.session_id)

            service.dispatch(
                SwitchPermissionProfileCommand(
                    session_id=state.session_id,
                    profile_id="accept_edits",
                )
            )
            event = service.next_event(subscription_id, timeout=1)

            self.assertIsInstance(event, PermissionProfileChangedEvent)
            self.assertEqual(event.profile_id, "accept_edits")
            self.assertEqual(state.permission_profile_id, "accept_edits")
            self.assertEqual(runtime.loop.permissions.profile.profile_id, "accept_edits")
            self.assertEqual(saved, ["accept_edits"])
            service.close()

    def test_sandbox_switch_is_persisted_and_emits_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp))
            runtime = _Runtime(state)
            saved = []
            service = AgentService(
                tool_registry_factory=lambda mode: object(),
                save_session=lambda session: saved.append(session.sandbox_mode),
            )
            service.register(runtime)
            subscription_id = self._subscribe(service, state.session_id)

            service.dispatch(
                SwitchSandboxCommand(
                    session_id=state.session_id,
                    mode=DANGER_FULL_ACCESS,
                )
            )
            event = service.next_event(subscription_id, timeout=1)

            self.assertIsInstance(event, SandboxChangedEvent)
            self.assertEqual(event.mode, DANGER_FULL_ACCESS)
            self.assertEqual(state.sandbox_mode, DANGER_FULL_ACCESS)
            self.assertEqual(runtime.loop.sandbox_policy.mode, DANGER_FULL_ACCESS)
            self.assertEqual(saved, [DANGER_FULL_ACCESS])
            service.close()

    def test_permission_grants_can_be_listed_revoked_and_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp))
            runtime = _Runtime(state)
            runtime.grants = [
                ApprovalGrant("edit", "app.py", "Edit app.py"),
                ApprovalGrant("shell", "pytest", "Run pytest"),
            ]
            service = AgentService(tool_registry_factory=lambda mode: object())
            service.register(runtime)
            subscription_id = self._subscribe(service, state.session_id)

            self.assertEqual(service.permission_grants(state.session_id), tuple(runtime.grants))

            service.dispatch(
                RevokePermissionGrantCommand(
                    session_id=state.session_id,
                    action="edit",
                    resource="app.py",
                )
            )
            revoked = service.next_event(subscription_id, timeout=1)
            self.assertIsInstance(revoked, PermissionGrantsChangedEvent)
            self.assertFalse(revoked.durable)
            self.assertEqual(revoked.reason, "revoked")
            self.assertEqual(revoked.removed_count, 1)
            self.assertEqual(revoked.grants, [runtime.grants[0].to_dict()])

            service.dispatch(ClearPermissionGrantsCommand(session_id=state.session_id))
            cleared = service.next_event(subscription_id, timeout=1)
            self.assertIsInstance(cleared, PermissionGrantsChangedEvent)
            self.assertEqual(cleared.reason, "cleared")
            self.assertEqual(cleared.removed_count, 1)
            self.assertEqual(cleared.grants, [])
            service.close()

    def test_goal_update_is_persisted_and_can_be_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp))
            state.goal = GoalState(
                goal_id="goal-1",
                objective="Update app.py",
                steps=(GoalStep("Edit app.py", "in_progress"),),
                status="active",
                updated_at="2026-10-24T00:00:00+00:00",
            )
            runtime = _Runtime(state)
            runtime.events.put(
                RuntimeEvent(
                    kind="turn_started",
                    state=RuntimeState.RUNNING_MODEL,
                    data={"turn_id": "session-1:1", "task": "Update app.py"},
                )
            )
            runtime.events.put(
                RuntimeEvent(
                    kind="goal_updated",
                    state=RuntimeState.RUNNING_TOOL,
                    data={
                        "step": 1,
                        "call_id": "call-goal",
                        "goal": state.goal.to_dict(),
                    },
                )
            )
            runtime.events.put(
                RuntimeEvent(
                    kind="turn_completed",
                    state=RuntimeState.COMPLETED,
                    text="done",
                    data={
                        "steps": 1,
                        "stopped_by": "final_answer",
                        "mode_id": "build",
                        "plan_id": None,
                    },
                    result=TurnResult(
                        answer="done",
                        messages=[],
                        steps=1,
                        stopped_by="final_answer",
                    ),
                )
            )
            saved = []
            service = AgentService(
                tool_registry_factory=lambda mode: object(),
                save_session=lambda session: saved.append(session.goal),
            )
            service.register(runtime)
            subscription_id = self._subscribe(service, state.session_id)

            service.dispatch(
                StartTurnCommand(session_id=state.session_id, user_input="Update app.py")
            )
            events, _ = self._next_terminal(service, subscription_id)
            updated = next(
                event for event in events if isinstance(event, GoalUpdatedEvent)
            )
            self.assertEqual(updated.goal, state.goal.to_dict())
            self.assertTrue(updated.durable)
            self.assertIn(state.goal, saved)
            self.assertEqual(service.current_goal(state.session_id), state.goal)

            service.dispatch(ClearGoalCommand(session_id=state.session_id))
            cleared = service.next_event(subscription_id, timeout=1)
            self.assertIsInstance(cleared, GoalClearedEvent)
            self.assertEqual(cleared.goal_id, "goal-1")
            self.assertIsNone(state.goal)
            service.close()

    def test_session_rename_is_persisted_and_emits_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp))
            runtime = _Runtime(state)
            saved = []
            service = AgentService(
                tool_registry_factory=lambda mode: object(),
                save_session=lambda session: saved.append(session.title),
            )
            service.register(runtime)
            subscription_id = self._subscribe(service, state.session_id)

            service.dispatch(
                RenameSessionCommand(
                    session_id=state.session_id,
                    title="  Context window audit  ",
                )
            )
            event = service.next_event(subscription_id, timeout=1)

            self.assertIsInstance(event, SessionRenamedEvent)
            self.assertEqual(event.title, "Context window audit")
            self.assertEqual(state.title, "Context window audit")
            self.assertEqual(saved, ["Context window audit"])
            service.close()

    def test_terminal_state_is_saved_before_terminal_event_is_published(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp))
            runtime = _Runtime(state)
            runtime.events.put(
                RuntimeEvent(
                    kind="turn_completed",
                    state=RuntimeState.COMPLETED,
                    text="done",
                    data={
                        "steps": 1,
                        "stopped_by": "final_answer",
                        "mode_id": "build",
                        "plan_id": None,
                    },
                )
            )
            save_started = threading.Event()
            release_save = threading.Event()

            def save_session(session: SessionState) -> None:
                save_started.set()
                release_save.wait(1)

            service = AgentService(
                tool_registry_factory=lambda mode: object(),
                save_session=save_session,
            )
            service.register(runtime)
            subscription_id = self._subscribe(service, state.session_id)
            service.dispatch(StartTurnCommand(session_id=state.session_id, user_input="hi"))

            self.assertTrue(save_started.wait(1))
            with self.assertRaises(queue.Empty):
                service.next_event(subscription_id, timeout=0.05)
            release_save.set()
            completed = service.next_event(subscription_id, timeout=1)

            self.assertIsInstance(completed, TurnCompletedEvent)
            service.close()

    def test_continuation_required_is_a_persisted_non_completion_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp))
            state.pending_turn = PendingTurn(
                turn_id="session-1:1",
                task="Inspect",
                message_start=0,
                step=13,
                pause_reason="step_budget",
                pause_details={"steps_completed": 12, "step_budget": 12, "next_step": 13},
            )
            runtime = _Runtime(state)
            runtime.events.put(
                RuntimeEvent(
                    kind="turn_continuation_required",
                    state=RuntimeState.WAITING_USER,
                    data={
                        "steps_completed": 12,
                        "step_budget": 12,
                        "next_step": 13,
                    },
                )
            )
            saved = []
            service = AgentService(
                tool_registry_factory=lambda mode: object(),
                save_session=lambda session: saved.append(session.pending_turn.step),
            )
            service.register(runtime)
            subscription_id = self._subscribe(service, state.session_id)

            service.dispatch(
                StartTurnCommand(session_id=state.session_id, retry=True)
            )
            event = service.next_event(subscription_id, timeout=1)
            service.wait(state.session_id)

            self.assertIsInstance(event, TurnContinuationRequiredEvent)
            self.assertEqual(event.next_step, 13)
            self.assertEqual(event.state, AgentState.WAITING_USER)
            self.assertEqual(saved, [13])
            self.assertIsNotNone(state.pending_turn)
            service.close()

    def test_service_registers_and_unregisters_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp))
            runtime = _Runtime(state)
            saved = []
            service = AgentService(
                tool_registry_factory=lambda mode: object(),
                save_session=lambda session: saved.append(session.session_id),
            )

            service.register(runtime)

            self.assertTrue(service.has_session(state.session_id))
            self.assertEqual(service.registered_session_ids(), (state.session_id,))
            service.unregister(state.session_id)
            self.assertFalse(service.has_session(state.session_id))
            self.assertTrue(runtime.closed)
            self.assertEqual(saved, [state.session_id])

    def test_unregister_waits_for_in_flight_session_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp))
            command_started = threading.Event()
            release_command = threading.Event()

            class BlockingRuntime(_Runtime):
                def switch_mode(self, mode, tools) -> None:
                    command_started.set()
                    release_command.wait(1)
                    super().switch_mode(mode, tools)

            runtime = BlockingRuntime(state)
            service = AgentService(tool_registry_factory=lambda mode: object())
            service.register(runtime)
            errors = []
            dispatch = threading.Thread(
                target=lambda: _capture_error(
                    errors,
                    lambda: service.dispatch(
                        SwitchModeCommand(session_id=state.session_id, mode_id="build")
                    ),
                )
            )
            unregister = threading.Thread(
                target=lambda: _capture_error(
                    errors,
                    lambda: service.unregister(state.session_id),
                )
            )

            dispatch.start()
            self.assertTrue(command_started.wait(1))
            unregister.start()
            self.assertTrue(unregister.is_alive())
            release_command.set()
            dispatch.join(1)
            unregister.join(1)

            self.assertFalse(dispatch.is_alive())
            self.assertFalse(unregister.is_alive())
            self.assertEqual(errors, [])
            self.assertFalse(service.has_session(state.session_id))
            self.assertTrue(runtime.closed)

    def test_execute_plan_is_owned_by_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp))
            state.proposed_plan = ProposedPlan(
                plan_id="plan-1",
                source_turn_id="session-1:1",
                content="Update app.py",
            )
            runtime = _Runtime(state)
            runtime.events.put(
                RuntimeEvent(
                    kind="turn_completed",
                    state=RuntimeState.COMPLETED,
                    text="done",
                    data={
                        "steps": 1,
                        "stopped_by": "final_answer",
                        "mode_id": "build",
                        "plan_id": None,
                    },
                    result=TurnResult(
                        answer="done",
                        messages=[],
                        steps=1,
                        stopped_by="final_answer",
                    ),
                )
            )
            saved = []
            service = AgentService(
                tool_registry_factory=lambda mode: f"tools:{mode.mode_id}",
                save_session=lambda session: saved.append(session.mode_id),
            )
            service.register(runtime)
            subscription_id = self._subscribe(service, state.session_id)

            service.dispatch(ExecutePlanCommand(session_id=state.session_id))
            approved = service.next_event(subscription_id, timeout=1)
            completed = service.next_event(subscription_id, timeout=1)

            self.assertEqual(approved.plan_id, "plan-1")
            self.assertEqual(approved.mode_id, BUILD_MODE.mode_id)
            self.assertIsInstance(completed, TurnCompletedEvent)
            self.assertEqual(state.proposed_plan.status, "approved")
            self.assertEqual(state.mode_id, BUILD_MODE.mode_id)
            self.assertEqual(runtime.loop.mode, BUILD_MODE)
            self.assertEqual(runtime.loop.tools, "tools:build")
            self.assertEqual(len(runtime.commands), 1)
            self.assertIsInstance(runtime.commands[0], StartTurn)
            self.assertEqual(runtime.commands[0].approved_plan_id, "plan-1")
            self.assertTrue(saved)
            service.close()

    def test_service_fans_out_and_replays_durable_item_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self._state(root)
            runtime = _Runtime(state)
            runtime.events.put(
                RuntimeEvent(
                    kind="turn_started",
                    state=RuntimeState.RUNNING_MODEL,
                    data={"turn_id": "session-1:1", "task": "hi"},
                )
            )
            runtime.events.put(
                RuntimeEvent(
                    kind="model_completed",
                    state=RuntimeState.RUNNING_MODEL,
                    data={
                        "step": 1,
                        "content": "hello",
                        "reasoning": "",
                        "finish_reason": "stop",
                        "usage": None,
                        "metadata": {},
                    },
                )
            )
            runtime.events.put(
                RuntimeEvent(
                    kind="content_delta",
                    state=RuntimeState.RUNNING_MODEL,
                    text="hello",
                    data={"step": 1},
                )
            )
            runtime.events.put(
                RuntimeEvent(
                    kind="turn_completed",
                    state=RuntimeState.COMPLETED,
                    text="hello",
                    data={
                        "steps": 1,
                        "stopped_by": "final_answer",
                        "mode_id": "build",
                        "plan_id": None,
                    },
                    result=TurnResult(
                        answer="hello",
                        messages=[],
                        steps=1,
                        stopped_by="final_answer",
                    ),
                )
            )
            event_store = EventStore(root / ".my-agent" / "sessions")
            service = AgentService(
                tool_registry_factory=lambda mode: object(),
                event_store=event_store,
            )
            service.register(runtime)
            first_subscription = self._subscribe(service, state.session_id)
            second_subscription = self._subscribe(service, state.session_id)
            service.dispatch(StartTurnCommand(session_id=state.session_id, user_input="hi"))
            first_events, _ = self._next_terminal(service, first_subscription)
            second_events, _ = self._next_terminal(service, second_subscription)

            self.assertEqual(
                [event.event_id for event in first_events],
                [event.event_id for event in second_events],
            )

            replay_subscription, _ = service.subscribe_events(
                session_ids=(state.session_id,),
                after_sequences={state.session_id: 0},
            )
            replayed_item = service.next_event(replay_subscription, timeout=1)
            while not (
                isinstance(replayed_item, ItemCompletedEvent)
                and replayed_item.item_type == "assistant_message"
            ):
                replayed_item = service.next_event(replay_subscription, timeout=1)
            replayed_terminal = service.next_event(replay_subscription, timeout=1)

            self.assertEqual(replayed_item.content, "hello")
            self.assertTrue(replayed_item.durable)
            self.assertIsInstance(replayed_terminal, TurnCompletedEvent)
            persisted = event_store.replay(state.session_id)
            self.assertFalse(any(isinstance(event, ItemDeltaEvent) for event in persisted))
            self.assertTrue(
                any(
                    isinstance(event, ItemCompletedEvent)
                    and event.item_type == "assistant_message"
                    and event.content == "hello"
                    for event in persisted
                )
            )
            projection = service.projection(state.session_id)
            self.assertEqual(projection["turns"][-1]["status"], "completed")
            service.close()


if __name__ == "__main__":
    unittest.main()
