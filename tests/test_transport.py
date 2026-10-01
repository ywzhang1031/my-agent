from __future__ import annotations

import io
import json
import queue
import tempfile
import unittest
from pathlib import Path

from my_agent.event_store import EventStore
from my_agent.protocol import AgentState, ContentDeltaEvent, TurnCompletedEvent
from my_agent.rpc import (
    RequestEnvelope,
    ResponseEnvelope,
    request_to_dict,
)
from my_agent.transport import JsonlAppServer


class _AppServer:
    def __init__(self) -> None:
        self.workspace = Path("/tmp/workspace")
        self.requests = []
        self.events = queue.Queue()
        self.closed = False

    def handle(self, request: RequestEnvelope) -> ResponseEnvelope:
        self.requests.append(request)
        self.events.put(
            TurnCompletedEvent(
                session_id=request.params["session_id"],
                sequence=1,
                state=AgentState.COMPLETED,
                answer="done",
                steps=1,
                stopped_by="final_answer",
                mode_id="build",
            )
        )
        return ResponseEnvelope.success(
            request.request_id,
            {"accepted": True},
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


class JsonlAppServerTests(unittest.TestCase):
    def test_server_writes_correlated_response_and_async_event_channels(self) -> None:
        app_server = _AppServer()
        request = RequestEnvelope(
            request_id="request-1",
            method="turn.start",
            params={"session_id": "session-1", "user_input": "hello"},
        )
        input_stream = io.StringIO(json.dumps(request_to_dict(request)) + "\n")
        output_stream = io.StringIO()

        JsonlAppServer(app_server).serve(input_stream, output_stream)

        messages = [json.loads(line) for line in output_stream.getvalue().splitlines()]
        self.assertEqual(messages[0]["channel"], "control")
        self.assertEqual(messages[0]["type"], "server.ready")
        response = next(item for item in messages if item["channel"] == "response")
        event = next(item for item in messages if item["channel"] == "event")
        self.assertEqual(response["request_id"], "request-1")
        self.assertTrue(response["ok"])
        self.assertEqual(event["type"], "turn.completed")
        self.assertEqual(event["session_id"], "session-1")
        self.assertEqual(app_server.requests, [request])
        self.assertTrue(app_server.closed)

    def test_server_returns_correlated_error_for_invalid_request(self) -> None:
        app_server = _AppServer()
        output_stream = io.StringIO()
        invalid = json.dumps(
            {
                "channel": "request",
                "rpc_version": 999,
                "request_id": "request-2",
                "method": "session.list",
                "params": {},
            }
        )

        JsonlAppServer(app_server).serve(io.StringIO(invalid + "\n"), output_stream)

        messages = [json.loads(line) for line in output_stream.getvalue().splitlines()]
        self.assertEqual(messages[1]["channel"], "response")
        self.assertEqual(messages[1]["request_id"], "request-2")
        self.assertFalse(messages[1]["ok"])
        self.assertEqual(messages[1]["error"]["code"], "invalid_request")

    def test_server_returns_null_request_id_for_invalid_json(self) -> None:
        app_server = _AppServer()
        output_stream = io.StringIO()

        JsonlAppServer(app_server).serve(io.StringIO("not-json\n"), output_stream)

        messages = [json.loads(line) for line in output_stream.getvalue().splitlines()]
        self.assertIsNone(messages[1]["request_id"])
        self.assertEqual(messages[1]["error"]["code"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
