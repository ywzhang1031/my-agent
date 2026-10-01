from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from pathlib import Path

from my_agent.agent_loop import TurnResult
from my_agent.control import ApprovalDecision
from my_agent.modes import BUILD_MODE, PLAN_MODE
from my_agent.protocol import (
    AgentState,
    ContentDeltaEvent,
    ExecutePlanCommand,
    ReplayCompletedEvent,
    ReplayEventsCommand,
    ResolveApprovalCommand,
    StartTurnCommand,
    SwitchModeCommand,
    TurnCompletedEvent,
    command_from_dict,
    command_to_dict,
    event_from_dict,
    event_to_dict,
)
from my_agent.runtime import RuntimeEvent, RuntimeState, StartTurn
from my_agent.service import AgentService
from my_agent.event_store import EventStore
from my_agent.session import ProposedPlan, SessionState


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
    def __init__(self) -> None:
        self.model = _Model()
        self.mode = PLAN_MODE
        self.tools = object()

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
        self.loop = _Loop()
        self.state = RuntimeState.IDLE
        self.events: queue.Queue[RuntimeEvent] = queue.Queue()
        self.commands = []
        self.closed = False

    def send(self, command) -> None:
        self.commands.append(command)

    def next_event(self, timeout=None) -> RuntimeEvent:
        return self.events.get(timeout=timeout)

    def switch_mode(self, mode, tools) -> None:
        self.loop.switch_mode(mode, tools)

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

    def test_event_round_trip_preserves_typed_payload(self) -> None:
        event = ContentDeltaEvent(
            session_id="session-1",
            sequence=3,
            state=AgentState.RUNNING_MODEL,
            step=2,
            text="hello",
        )

        restored = event_from_dict(event_to_dict(event))

        self.assertEqual(restored, event)


class AgentServiceTests(unittest.TestCase):
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
            service = AgentService(tool_registry_factory=lambda mode: object())
            service.register(runtime)

            service.dispatch(StartTurnCommand(session_id=state.session_id, user_input="hi"))
            first = service.next_event(timeout=1)
            second = service.next_event(timeout=1)

            self.assertIsInstance(first, ContentDeltaEvent)
            self.assertEqual(first.sequence, 1)
            self.assertEqual(first.session_id, state.session_id)
            self.assertIsInstance(second, TurnCompletedEvent)
            self.assertEqual(second.sequence, 2)
            self.assertEqual(second.answer, "hello")
            self.assertIsInstance(runtime.commands[0], StartTurn)
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

            service.dispatch(ExecutePlanCommand(session_id=state.session_id))
            approved = service.next_event(timeout=1)
            completed = service.next_event(timeout=1)

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

    def test_service_replays_persisted_events_after_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self._state(root)
            runtime = _Runtime(state)
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
            service = AgentService(
                tool_registry_factory=lambda mode: object(),
                event_store=EventStore(root / ".my-agent" / "sessions"),
            )
            service.register(runtime)
            service.dispatch(StartTurnCommand(session_id=state.session_id, user_input="hi"))
            service.next_event(timeout=1)
            service.next_event(timeout=1)

            service.dispatch(
                ReplayEventsCommand(session_id=state.session_id, after_sequence=1)
            )
            replayed = service.next_event(timeout=1)
            completed = service.next_event(timeout=1)

            self.assertIsInstance(replayed, TurnCompletedEvent)
            self.assertEqual(replayed.sequence, 2)
            self.assertIsInstance(completed, ReplayCompletedEvent)
            self.assertEqual(completed.replayed_events, 1)
            self.assertEqual(completed.latest_sequence, 2)
            self.assertEqual(completed.sequence, 3)
            service.close()


if __name__ == "__main__":
    unittest.main()
