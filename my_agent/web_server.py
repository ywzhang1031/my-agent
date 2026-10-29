from __future__ import annotations

import json
import queue
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

from .app_server import AppServer
from .protocol import PROTOCOL_VERSION, event_to_dict
from .request_store import MUTATING_METHODS
from .rpc import (
    RPC_VERSION,
    ResponseEnvelope,
    request_from_dict,
    response_to_dict,
)


MAX_REQUEST_BYTES = 1_048_576
MAX_EVENT_WAIT_SECONDS = 25.0
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


class _ThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class LocalWebServer:
    """Loopback-only HTTP bridge for the AppServer RPC and event protocols."""

    def __init__(
        self,
        app_server: AppServer,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        token: str | None = None,
        static_root: str | Path | None = None,
    ) -> None:
        if host not in LOOPBACK_HOSTS:
            raise ValueError("web UI host must be loopback-only")
        if port < 0 or port > 65_535:
            raise ValueError("web UI port must be between 0 and 65535")
        self.app_server = app_server
        self.host = host
        self.token = token or secrets.token_urlsafe(32)
        self.static_root = (
            Path(static_root).expanduser().resolve()
            if static_root is not None
            else Path(__file__).with_name("web").resolve()
        )
        self._httpd = _ThreadingHTTPServer(
            (host, port),
            self._handler_class(),
        )

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        display_host = "127.0.0.1" if self.host == "localhost" else self.host
        return f"http://{display_host}:{self.port}/"

    def serve_forever(self) -> None:
        try:
            self._httpd.serve_forever(poll_interval=0.1)
        finally:
            self._httpd.server_close()
            self.app_server.close()

    def shutdown(self) -> None:
        self._httpd.shutdown()

    def _handler_class(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "MyAgent"
            sys_version = ""

            def do_GET(self) -> None:
                if not self._host_allowed():
                    self._send_json(
                        HTTPStatus.FORBIDDEN,
                        {"error": "invalid host"},
                    )
                    return
                parsed = urlsplit(self.path)
                if parsed.path == "/api/events":
                    self._get_event(parsed.query)
                    return
                if parsed.path == "/api/health":
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "workspace": str(owner.app_server.workspace),
                            "instance_id": owner.app_server.instance_id,
                            "rpc_version": RPC_VERSION,
                            "protocol_version": PROTOCOL_VERSION,
                        },
                    )
                    return
                if parsed.path == "/bootstrap.js":
                    bootstrap = json.dumps(
                        {
                            "token": owner.token,
                            "workspace": str(owner.app_server.workspace),
                            "instanceID": owner.app_server.instance_id,
                            "rpcVersion": RPC_VERSION,
                            "protocolVersion": PROTOCOL_VERSION,
                            "deduplicatedMethods": sorted(MUTATING_METHODS),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    self._send_bytes(
                        HTTPStatus.OK,
                        f"window.__MY_AGENT__={bootstrap};\n".encode("utf-8"),
                        "application/javascript; charset=utf-8",
                    )
                    return
                static_files = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
                    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                }
                target = static_files.get(parsed.path)
                if target is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                filename, content_type = target
                try:
                    body = (owner.static_root / filename).read_bytes()
                except FileNotFoundError:
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": f"missing UI asset: {filename}"},
                    )
                    return
                self._send_bytes(HTTPStatus.OK, body, content_type)

            def do_POST(self) -> None:
                if not self._host_allowed():
                    self._send_json(
                        HTTPStatus.FORBIDDEN,
                        {"error": "invalid host"},
                    )
                    return
                if urlsplit(self.path).path != "/api/rpc":
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                if not self._authorized():
                    self._send_json(
                        HTTPStatus.UNAUTHORIZED,
                        {"error": "invalid UI token"},
                    )
                    return
                response = self._read_rpc_request()
                self._send_json(HTTPStatus.OK, response_to_dict(response))

            def do_OPTIONS(self) -> None:
                self._send_json(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    {"error": "cross-origin requests are not supported"},
                )

            def _get_event(self, query: str) -> None:
                if not self._authorized():
                    self._send_json(
                        HTTPStatus.UNAUTHORIZED,
                        {"error": "invalid UI token"},
                    )
                    return
                query_params = parse_qs(query)
                subscription_id = query_params.get("subscription_id", [""])[0]
                if not subscription_id:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"error": "subscription_id is required"},
                    )
                    return
                raw_timeout = query_params.get("timeout", ["20"])[0]
                try:
                    timeout = float(raw_timeout)
                except ValueError:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"error": "timeout must be a number"},
                    )
                    return
                timeout = max(0.0, min(timeout, MAX_EVENT_WAIT_SECONDS))
                try:
                    event = owner.app_server.next_event(
                        subscription_id,
                        timeout=timeout,
                    )
                except queue.Empty:
                    self._send_bytes(HTTPStatus.NO_CONTENT, b"", "application/json")
                    return
                except ValueError as exc:
                    self._send_json(
                        HTTPStatus.GONE,
                        {"error": str(exc)},
                    )
                    return
                message = event_to_dict(event)
                message["channel"] = "event"
                self._send_json(HTTPStatus.OK, message)

            def _read_rpc_request(self) -> ResponseEnvelope:
                request_id: str | None = None
                try:
                    raw_length = self.headers.get("Content-Length")
                    if raw_length is None:
                        raise ValueError("Content-Length is required")
                    content_length = int(raw_length)
                    if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                        raise ValueError("RPC request body has an invalid size")
                    data = json.loads(self.rfile.read(content_length))
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
                return owner.app_server.handle(request)

            def _authorized(self) -> bool:
                supplied = self.headers.get("X-My-Agent-Token", "")
                return secrets.compare_digest(supplied, owner.token)

            def _host_allowed(self) -> bool:
                raw_host = self.headers.get("Host", "")
                hostname = raw_host
                if raw_host.startswith("["):
                    closing = raw_host.find("]")
                    hostname = raw_host[: closing + 1] if closing >= 0 else raw_host
                elif ":" in raw_host:
                    hostname = raw_host.rsplit(":", 1)[0]
                return hostname.lower() in LOOPBACK_HOSTS

            def _send_json(self, status: HTTPStatus, value: Any) -> None:
                self._send_bytes(
                    status,
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                    "application/json; charset=utf-8",
                )

            def _send_bytes(
                self,
                status: HTTPStatus,
                body: bytes,
                content_type: str,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-My-Agent-Instance", owner.app_server.instance_id)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
                )
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return Handler
