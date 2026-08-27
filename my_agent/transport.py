from __future__ import annotations

import json
import queue
import threading
from typing import Any, TextIO

from .protocol import (
    PROTOCOL_VERSION,
    command_from_dict,
    event_to_dict,
)
from .service import AgentService


class JsonlAgentServer:
    """Bidirectional stdio transport for one registered agent session."""

    def __init__(self, service: AgentService, *, session_id: str) -> None:
        self.service = service
        self.session_id = session_id
        self._write_lock = threading.Lock()
        self._stop = threading.Event()

    def serve(self, input_stream: TextIO, output_stream: TextIO) -> None:
        self._write(
            output_stream,
            {
                "protocol_version": PROTOCOL_VERSION,
                "type": "server.ready",
                "session_id": self.session_id,
            },
        )
        writer = threading.Thread(
            target=self._write_events,
            args=(output_stream,),
            name=f"agent-jsonl-writer-{self.session_id}",
            daemon=True,
        )
        writer.start()
        try:
            for line in input_stream:
                if not line.strip():
                    continue
                try:
                    command = command_from_dict(json.loads(line))
                    if command.session_id != self.session_id:
                        raise ValueError(
                            f"command session {command.session_id!r} does not match "
                            f"server session {self.session_id!r}"
                        )
                    self.service.dispatch(command)
                except (
                    AttributeError,
                    KeyError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    json.JSONDecodeError,
                ) as exc:
                    self._write_error(output_stream, str(exc))
        finally:
            self.service.close()
            self._stop.set()
            writer.join(5.0)
            if writer.is_alive():
                raise RuntimeError("JSONL event writer did not finish")

    def _write_events(self, output_stream: TextIO) -> None:
        while not self._stop.is_set():
            try:
                event = self.service.next_event(timeout=0.05)
            except queue.Empty:
                continue
            self._write(output_stream, event_to_dict(event))
        while True:
            try:
                event = self.service.next_event(timeout=0)
            except queue.Empty:
                return
            self._write(output_stream, event_to_dict(event))

    def _write_error(self, output_stream: TextIO, message: str) -> None:
        self._write(
            output_stream,
            {
                "protocol_version": PROTOCOL_VERSION,
                "type": "server.error",
                "session_id": self.session_id,
                "error": {
                    "code": "invalid_command",
                    "message": message,
                },
            },
        )

    def _write(self, output_stream: TextIO, message: dict[str, Any]) -> None:
        with self._write_lock:
            json.dump(
                message,
                output_stream,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            output_stream.write("\n")
            output_stream.flush()
