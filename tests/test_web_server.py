from __future__ import annotations

import http.client
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from my_agent.protocol import AgentState, ItemDeltaEvent
from my_agent.rpc import RequestEnvelope, ResponseEnvelope, request_to_dict
from my_agent.web_server import LocalWebServer


class _AppServer:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.requests = []
        self.events = queue.Queue()
        self.closed = False

    def handle(self, request: RequestEnvelope) -> ResponseEnvelope:
        self.requests.append(request)
        return ResponseEnvelope.success(
            request.request_id,
            {"method": request.method},
        )

    def next_event(self, subscription_id, timeout=None):
        if subscription_id != "subscription-1":
            raise ValueError("unknown subscription")
        return self.events.get(timeout=timeout)

    def close(self, timeout=5.0) -> None:
        self.closed = True


class LocalWebServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        static_root = root / "web"
        static_root.mkdir()
        (static_root / "index.html").write_text("<main>My Agent</main>", encoding="utf-8")
        (static_root / "app.js").write_text("console.log('app')", encoding="utf-8")
        (static_root / "styles.css").write_text("main { color: black; }", encoding="utf-8")
        self.app = _AppServer(root)
        self.server = LocalWebServer(
            self.app,
            port=0,
            token="test-token",
            static_root=static_root,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(2)
        self.tempdir.cleanup()

    def request(self, path: str, *, token: bool = False):
        headers = {"X-My-Agent-Token": "test-token"} if token else {}
        return urlopen(Request(self.server.url + path.lstrip("/"), headers=headers))

    def test_serves_static_assets_and_bootstrap_without_exposing_cross_origin_access(self) -> None:
        with self.request("/") as response:
            self.assertEqual(response.status, 200)
            self.assertIn("My Agent", response.read().decode("utf-8"))
            self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])
        with self.request("/bootstrap.js") as response:
            self.assertIn("test-token", response.read().decode("utf-8"))

        connection = http.client.HTTPConnection("127.0.0.1", self.server.port)
        connection.request("GET", "/api/health", headers={"Host": "example.com"})
        response = connection.getresponse()
        response.read()
        connection.close()

        self.assertEqual(response.status, 403)

    def test_bundled_ui_contains_secure_credential_recovery_panel(self) -> None:
        bundled_root = Path(__file__).parents[1] / "my_agent" / "web"
        html = (bundled_root / "index.html").read_text(encoding="utf-8")
        javascript = (bundled_root / "app.js").read_text(encoding="utf-8")

        self.assertIn('type="password"', html)
        self.assertIn('id="credential-panel"', html)
        self.assertIn('id="permission-menu"', html)
        self.assertIn('id="model-menu"', html)
        self.assertIn('id="session-rename-form"', html)
        self.assertNotIn("<select", html)
        self.assertIn('rpc("provider.credential.set"', javascript)
        self.assertIn('rpc("permission.switch"', javascript)
        self.assertIn('rpc("session.rename"', javascript)
        self.assertNotIn("localStorage", javascript)

    def test_rpc_endpoint_requires_token_and_preserves_request_id(self) -> None:
        request = RequestEnvelope(
            request_id="request-1",
            method="session.list",
            params={},
        )
        body = json.dumps(request_to_dict(request)).encode("utf-8")
        authorized = Request(
            self.server.url + "api/rpc",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-My-Agent-Token": "test-token",
            },
            method="POST",
        )

        with urlopen(authorized) as response:
            message = json.loads(response.read())

        self.assertEqual(message["request_id"], "request-1")
        self.assertTrue(message["ok"])
        self.assertEqual(message["result"]["method"], "session.list")
        self.assertEqual(self.app.requests, [request])

        unauthorized = Request(
            self.server.url + "api/rpc",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as context:
            urlopen(unauthorized)
        self.assertEqual(context.exception.code, 401)
        context.exception.close()

    def test_event_endpoint_long_polls_typed_events(self) -> None:
        self.app.events.put(
            ItemDeltaEvent(
                session_id="session-1",
                turn_id="session-1:1",
                sequence=4,
                state=AgentState.RUNNING_MODEL,
                durable=False,
                item_id="assistant-1",
                item_type="assistant_message",
                step=1,
                delta="hello",
            )
        )

        with self.request(
            "/api/events?timeout=0&subscription_id=subscription-1",
            token=True,
        ) as response:
            message = json.loads(response.read())
        with self.request(
            "/api/events?timeout=0&subscription_id=subscription-1",
            token=True,
        ) as response:
            empty_status = response.status

        self.assertEqual(message["channel"], "event")
        self.assertEqual(message["type"], "item.delta")
        self.assertEqual(message["payload"]["delta"], "hello")
        self.assertEqual(empty_status, 204)

    def test_shutdown_closes_app_server(self) -> None:
        self.server.shutdown()
        self.thread.join(2)

        self.assertFalse(self.thread.is_alive())
        self.assertTrue(self.app.closed)

    def test_packaged_ui_assets_are_present(self) -> None:
        root = Path(__file__).parents[1] / "my_agent" / "web"

        self.assertIn('id="session-list"', (root / "index.html").read_text())
        self.assertIn("/api/events", (root / "app.js").read_text())
        self.assertIn("grid-template-columns", (root / "styles.css").read_text())


if __name__ == "__main__":
    unittest.main()
