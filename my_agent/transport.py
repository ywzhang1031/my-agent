from __future__ import annotations

import json
import queue
import threading
from typing import Any, Mapping, TextIO

from .app_server import AppServer
from .protocol import PROTOCOL_VERSION, event_to_dict
from .rpc import (
    RPC_VERSION,
    ResponseEnvelope,
    request_from_dict,
    response_to_dict,
)


class JsonlAppServer:
    """Multiplexed JSONL stdio transport for one multi-session AppServer."""

    def __init__(self, app_server: AppServer) -> None:
        self.app_server = app_server
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._subscription_id: str | None = None

    def serve(self, input_stream: TextIO, output_stream: TextIO) -> None:
        self._write(
            output_stream,
            {
                "channel": "control",
                "type": "server.ready",
                "rpc_version": RPC_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "instance_id": self.app_server.instance_id,
                "workspace": str(self.app_server.workspace),
            },
        )
        self._subscription_id = self.app_server.subscribe_events()
        writer = threading.Thread(
            target=self._write_events,
            args=(output_stream,),
            name="agent-jsonl-event-writer",
            daemon=True,
        )
        writer.start()
        try:
            for line in input_stream:
                if not line.strip():
                    continue
                response = self._handle_line(line)
                self._write(output_stream, response_to_dict(response))
        finally:
            self._stop.set()
            writer.join(5.0)
            if writer.is_alive():
                raise RuntimeError("JSONL event writer did not finish")
            if self._subscription_id is not None:
                self.app_server.unsubscribe_events(self._subscription_id)
                self._subscription_id = None
            self.app_server.close()

    def _handle_line(self, line: str) -> ResponseEnvelope:
        request_id: str | None = None
        try:
            data = json.loads(line)
            if not isinstance(data, Mapping):
                raise ValueError("RPC request must be a JSON object")
            raw_request_id = data.get("request_id")
            if isinstance(raw_request_id, str):
                request_id = raw_request_id
            request = request_from_dict(data)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return ResponseEnvelope.failure(
                request_id,
                code="invalid_request",
                message=str(exc),
            )
        return self.app_server.handle(request)

    def _write_events(self, output_stream: TextIO) -> None:
        if self._subscription_id is None:
            raise RuntimeError("JSONL event subscription is not initialized")
        while not self._stop.is_set():
            try:
                event = self.app_server.next_event(
                    self._subscription_id,
                    timeout=0.05,
                )
            except queue.Empty:
                continue
            self._write_event(output_stream, event)
        while True:
            try:
                event = self.app_server.next_event(
                    self._subscription_id,
                    timeout=0,
                )
            except queue.Empty:
                return
            self._write_event(output_stream, event)

    def _write_event(self, output_stream: TextIO, event) -> None:
        message = event_to_dict(event)
        message["channel"] = "event"
        self._write(output_stream, message)

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
