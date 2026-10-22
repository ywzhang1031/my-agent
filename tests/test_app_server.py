from __future__ import annotations

import queue
import tempfile
import unittest
from pathlib import Path

from my_agent.agent_loop import TurnResult
from my_agent.app_server import AppServer
from my_agent.config import ConfiguredModel
from my_agent.event_store import EventStore
from my_agent.modes import BUILD_MODE, COLLABORATION_MODES, get_mode
from my_agent.rpc import (
    RequestEnvelope,
    ResponseEnvelope,
    request_from_dict,
    request_to_dict,
    response_from_dict,
    response_to_dict,
)
from my_agent.runtime import RuntimeEvent, RuntimeState, StartTurn
from my_agent.service import AgentService
from my_agent.session import SessionState, SessionStore


class _Profile:
    provider_id = "test"
    model_id = "test-model"


class _Model:
    profile = _Profile()


class _Loop:
    def __init__(self, mode_id: str) -> None:
        self.model = _Model()
        self.mode = get_mode(mode_id)
        self.tools = _Tools(self.mode.tool_names)

    def inspect_context(self, state):
        return _ContextSnapshot()


class _ContextSnapshot:
    def to_dict(self):
        return {
            "estimated_tokens": 125,
            "context_window_tokens": 10_000,
            "input_budget": 9_000,
            "reserve_output_tokens": 1_000,
            "active_messages": 2,
            "summarized_messages": 0,
            "compacted_messages": 0,
            "breakdown": {
                "system_prompt_tokens": 25,
                "tool_definition_tokens": 50,
                "summary_tokens": 0,
                "conversation_tokens": 34,
                "protocol_tokens": 16,
                "total_tokens": 125,
            },
        }


class _Tools:
    def __init__(self, names) -> None:
        self._names = list(names)

    def names(self):
        return self._names


class _Runtime:
    def __init__(self, state: SessionState, mode_id: str) -> None:
        self.session = state
        self.loop = _Loop(mode_id)
        self.state = RuntimeState.IDLE
        self.events: queue.Queue[RuntimeEvent] = queue.Queue()
        self.commands = []
        self.closed = False
        self.active = False

    def send(self, command) -> None:
        self.commands.append(command)
        if isinstance(command, StartTurn):
            self.active = True

    def next_event(self, timeout=None):
        event = self.events.get(timeout=timeout)
        if event.kind in {"turn_completed", "turn_failed", "turn_interrupted"}:
            self.active = False
        return event

    def close(self, timeout=5.0) -> None:
        self.closed = True
        if self.active:
            self.events.put(
                RuntimeEvent(
                    kind="turn_interrupted",
                    state=RuntimeState.INTERRUPTED,
                    text="server closed session",
                )
            )

    def wait(self, timeout=5.0) -> None:
        return None


class AppServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name, "workspace")
        self.workspace.mkdir()
        self.store = SessionStore(self.workspace / ".my-agent" / "sessions")
        self.service = AgentService(
            tool_registry_factory=lambda mode: _Tools(mode.tool_names),
            save_session=self.store.save,
            event_store=EventStore(self.store.root),
        )
        self.runtimes: dict[str, _Runtime] = {}
        self.saved_credentials: list[tuple[str, str]] = []
        self.credentials_path = self.workspace / "credentials.json"

        def runtime_factory(
            state: SessionState,
            model_ref: str | None,
            mode_id: str | None,
        ) -> _Runtime:
            if model_ref is not None and model_ref != "test/test-model":
                raise ValueError(f"unknown test model: {model_ref}")
            selected_mode = mode_id or state.mode_id or BUILD_MODE.mode_id
            get_mode(selected_mode)
            state.provider_id = "test"
            state.model_id = "test-model"
            state.mode_id = selected_mode
            runtime = _Runtime(state, selected_mode)
            self.runtimes[state.session_id] = runtime
            return runtime

        self.app = AppServer(
            workspace=self.workspace,
            session_store=self.store,
            agent_service=self.service,
            runtime_factory=runtime_factory,
            models=(
                ConfiguredModel(
                    provider_id="test",
                    model_id="test-model",
                    description="Test Model",
                    context_window_tokens=10_000,
                    max_output_tokens=1_000,
                ),
            ),
            modes=COLLABORATION_MODES,
            credentials_path=self.credentials_path,
            credential_setter=lambda provider_id, api_key: self.saved_credentials.append(
                (provider_id, api_key)
            ),
        )
        self.subscription_id = self.app.subscribe_events()

    def tearDown(self) -> None:
        self.app.close()
        self.tempdir.cleanup()

    def rpc(self, method: str, **params) -> ResponseEnvelope:
        return self.app.handle(RequestEnvelope(method=method, params=params))

    def create_session(self, **params) -> str:
        response = self.rpc("session.create", **params)
        self.assertTrue(response.ok, response.error)
        return response.result["session"]["session_id"]

    def complete_turn(self, session_id: str, answer: str) -> None:
        self.runtimes[session_id].events.put(
            RuntimeEvent(
                kind="turn_completed",
                state=RuntimeState.COMPLETED,
                text=answer,
                data={
                    "steps": 1,
                    "stopped_by": "final_answer",
                    "mode_id": "build",
                    "plan_id": None,
                },
                result=TurnResult(
                    answer=answer,
                    messages=[],
                    steps=1,
                    stopped_by="final_answer",
                ),
            )
        )

    def test_rpc_envelopes_round_trip_success_and_error(self) -> None:
        request = RequestEnvelope(
            request_id="request-1",
            method="session.list",
            params={},
        )
        self.assertEqual(request_from_dict(request_to_dict(request)), request)

        success = ResponseEnvelope.success("request-1", {"sessions": []})
        failure = ResponseEnvelope.failure(
            "request-2",
            code="not_found",
            message="missing",
            data={"session_id": "missing"},
        )
        self.assertEqual(response_from_dict(response_to_dict(success)), success)
        self.assertEqual(response_from_dict(response_to_dict(failure)), failure)

    def test_session_lifecycle_and_catalogs(self) -> None:
        session_id = self.create_session(
            model_ref="test/test-model",
            mode_id="plan",
        )

        listed = self.rpc("session.list")
        read = self.rpc("session.read", session_id=session_id)
        models = self.rpc("model.list")
        modes = self.rpc("mode.list")
        context = self.rpc("context.inspect", session_id=session_id)

        self.assertEqual(listed.result["sessions"][0]["session_id"], session_id)
        self.assertTrue(listed.result["sessions"][0]["active"])
        self.assertEqual(read.result["session"]["mode_id"], "plan")
        self.assertEqual(read.result["session"]["runtime_state"], "idle")
        self.assertEqual(read.result["messages"], [])
        self.assertEqual(read.result["projection"]["turns"], [])
        self.assertEqual(models.result["models"][0]["ref"], "test/test-model")
        self.assertEqual(
            [item["mode_id"] for item in modes.result["modes"]],
            ["build", "plan"],
        )
        self.assertEqual(context.result["context"]["estimated_tokens"], 125)

        closed = self.rpc("session.close", session_id=session_id)
        closed_again = self.rpc("session.close", session_id=session_id)
        resumed = self.rpc("session.resume", session_id=session_id)

        self.assertTrue(closed.result["closed"])
        self.assertFalse(closed_again.result["closed"])
        self.assertTrue(resumed.result["session"]["active"])
        self.assertEqual(resumed.result["session"]["mode_id"], "plan")

    def test_two_sessions_run_concurrently_but_one_session_rejects_overlap(self) -> None:
        first = self.create_session()
        second = self.create_session()

        first_start = self.rpc("turn.start", session_id=first, user_input="first")
        second_start = self.rpc("turn.start", session_id=second, user_input="second")
        overlap = self.rpc("turn.start", session_id=first, user_input="overlap")

        self.assertTrue(first_start.ok)
        self.assertTrue(second_start.ok)
        self.assertFalse(overlap.ok)
        self.assertEqual(overlap.error.code, "conflict")
        self.assertIsInstance(self.runtimes[first].commands[0], StartTurn)
        self.assertIsInstance(self.runtimes[second].commands[0], StartTurn)

        self.complete_turn(first, "first done")
        self.complete_turn(second, "second done")
        events = [
            self.app.next_event(self.subscription_id, timeout=1),
            self.app.next_event(self.subscription_id, timeout=1),
        ]

        self.assertEqual({event.session_id for event in events}, {first, second})
        self.assertEqual([event.sequence for event in events], [1, 1])
        self.service.wait(first)
        self.service.wait(second)

    def test_agent_command_requires_an_active_session(self) -> None:
        state = self.store.create(self.workspace)

        response = self.rpc(
            "turn.start",
            session_id=state.session_id,
            user_input="hello",
        )

        self.assertFalse(response.ok)
        self.assertEqual(response.error.code, "session_not_active")

    def test_provider_credential_set_rebuilds_runtime_without_returning_secret(self) -> None:
        session_id = self.create_session()
        original_runtime = self.runtimes[session_id]

        response = self.rpc(
            "provider.credential.set",
            session_id=session_id,
            api_key="new-secret",
        )

        self.assertTrue(response.ok, response.error)
        self.assertEqual(self.saved_credentials, [("test", "new-secret")])
        self.assertTrue(original_runtime.closed)
        self.assertIsNot(self.runtimes[session_id], original_runtime)
        self.assertEqual(response.result["provider_id"], "test")
        self.assertEqual(
            response.result["credentials_path"],
            str(self.credentials_path.resolve()),
        )
        self.assertNotIn("new-secret", str(response.result))
        persisted_session = self.store.load(session_id)
        self.assertNotIn("new-secret", persisted_session.metadata_path.read_text())
        self.assertNotIn("new-secret", persisted_session.messages_path.read_text())

    def test_closing_an_active_session_interrupts_and_flushes_its_event(self) -> None:
        session_id = self.create_session()
        started = self.rpc(
            "turn.start",
            session_id=session_id,
            user_input="long task",
        )

        closed = self.rpc("session.close", session_id=session_id)
        event = self.app.next_event(self.subscription_id, timeout=1)

        self.assertTrue(started.ok)
        self.assertTrue(closed.result["closed"])
        self.assertEqual(event.session_id, session_id)
        self.assertEqual(event.sequence, 1)
        self.assertEqual(event.reason, "server closed session")
        self.assertFalse(self.service.has_session(session_id))

    def test_event_subscription_rpc_creates_independent_consumers(self) -> None:
        session_id = self.create_session()
        first = self.rpc("events.subscribe")
        second = self.rpc("events.subscribe", session_ids=[session_id])

        self.assertTrue(first.ok, first.error)
        self.assertTrue(second.ok, second.error)
        started = self.rpc(
            "turn.start",
            session_id=session_id,
            user_input="test fan-out",
        )
        self.assertTrue(started.ok, started.error)
        self.complete_turn(session_id, "done")
        first_event = self.app.next_event(first.result["subscription_id"], timeout=1)
        second_event = self.app.next_event(second.result["subscription_id"], timeout=1)

        self.assertEqual(first_event.event_id, second_event.event_id)
        unsubscribed = self.rpc(
            "events.unsubscribe",
            subscription_id=first.result["subscription_id"],
        )
        self.assertTrue(unsubscribed.result["unsubscribed"])

    def test_unknown_method_and_invalid_params_are_structured_errors(self) -> None:
        unknown = self.rpc("unknown.method")
        invalid = self.rpc("session.list", unexpected=True)
        invalid_model = self.rpc("session.create", model_ref="test/missing")

        self.assertEqual(unknown.error.code, "method_not_found")
        self.assertEqual(invalid.error.code, "invalid_params")
        self.assertEqual(invalid_model.error.code, "invalid_params")
        self.assertEqual(self.store.list_sessions(), [])


if __name__ == "__main__":
    unittest.main()
