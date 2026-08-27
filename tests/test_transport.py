from __future__ import annotations

import io
import json
import queue
import tempfile
import unittest
from pathlib import Path

from my_agent.event_store import EventStore
from my_agent.protocol import (
    AgentState,
    ContentDeltaEvent,
    StartTurnCommand,
    TurnCompletedEvent,
    command_to_dict,
)
from my_agent.transport import JsonlAgentServer


class _Service:
    def __init__(self) -> None:
        self.commands = []
        self.events = queue.Queue()
        self.closed = False

    def dispatch(self, command) -> None:
        self.commands.append(command)
        self.events.put(
            TurnCompletedEvent(
                session_id=command.session_id,
                sequence=1,
                state=AgentState.COMPLETED,
                answer="done",
                steps=1,
                stopped_by="final_answer",
                mode_id="build",
            )
        )

    def next_event(self, timeout=None):
        return self.events.get(timeout=timeout)

    def close(self, timeout=5.0) -> None:
        self.closed = True


class EventStoreTests(unittest.TestCase):
    def test_append_replay_and_latest_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            first = ContentDeltaEvent(
                session_id="session-1",
                sequence=1,
                state=AgentState.RUNNING_MODEL,
                step=1,
                text="one",
            )
            second = TurnCompletedEvent(
                session_id="session-1",
                sequence=2,
                state=AgentState.COMPLETED,
                answer="done",
                steps=1,
                stopped_by="final_answer",
                mode_id="build",
            )

            store.append(first)
            store.append(second)

            self.assertEqual(store.latest_sequence("session-1"), 2)
            self.assertEqual(store.replay("session-1", after_sequence=1), [second])

    def test_append_rejects_non_monotonic_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            event = ContentDeltaEvent(
                session_id="session-1",
                sequence=1,
                state=AgentState.RUNNING_MODEL,
                step=1,
                text="one",
            )
            store.append(event)

            with self.assertRaisesRegex(ValueError, "sequence must increase"):
                store.append(event)


class JsonlAgentServerTests(unittest.TestCase):
    def test_server_decodes_commands_and_writes_typed_events(self) -> None:
        service = _Service()
        command = StartTurnCommand(session_id="session-1", user_input="hello")
        input_stream = io.StringIO(json.dumps(command_to_dict(command)) + "\n")
        output_stream = io.StringIO()
        server = JsonlAgentServer(service, session_id="session-1")

        server.serve(input_stream, output_stream)

        messages = [json.loads(line) for line in output_stream.getvalue().splitlines()]
        self.assertEqual(messages[0]["type"], "server.ready")
        self.assertEqual(messages[0]["session_id"], "session-1")
        self.assertEqual(messages[1]["type"], "turn.completed")
        self.assertEqual(messages[1]["payload"]["answer"], "done")
        self.assertEqual(service.commands, [command])
        self.assertTrue(service.closed)

    def test_server_returns_protocol_error_for_invalid_json(self) -> None:
        service = _Service()
        output_stream = io.StringIO()
        server = JsonlAgentServer(service, session_id="session-1")

        server.serve(io.StringIO("not-json\n"), output_stream)

        messages = [json.loads(line) for line in output_stream.getvalue().splitlines()]
        self.assertEqual(messages[1]["type"], "server.error")
        self.assertEqual(messages[1]["error"]["code"], "invalid_command")


if __name__ == "__main__":
    unittest.main()
