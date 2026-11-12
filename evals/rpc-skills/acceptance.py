"""Trusted offline grader. Executed only after editing, never in model context."""
from __future__ import annotations

from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from my_agent.app_server import AppServer
from my_agent.config import ConfiguredModel
from my_agent.database import Database, StorageError
from my_agent.event_store import EventStore
from my_agent.modes import COLLABORATION_MODES, get_mode
from my_agent.rpc import RequestEnvelope, response_to_dict
from my_agent.service import AgentService
from my_agent.session import SessionState, SessionStore, session_to_dict

# These helpers come from the immutable pre-fix regression suite, not the candidate.
from test_app_server import _Runtime, _Tools


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = SessionStore(root / "sessions")
        self.runtimes = {}
        self.credentials = []
        service = AgentService(tool_registry_factory=lambda mode: _Tools(mode.tool_names),
                               save_session=self.store.save, event_store=EventStore(self.store.root))

        def runtime(state, model_ref, mode_id, permission_profile_id, sandbox_mode):
            if model_ref is not None and model_ref != "test/test-model":
                raise ValueError("unknown test model")
            state.provider_id, state.model_id = "test", "test-model"
            state.mode_id = mode_id or state.mode_id or "build"
            get_mode(state.mode_id)
            if permission_profile_id is not None:
                state.permission_profile_id = permission_profile_id
            if sandbox_mode is not None:
                state.sandbox_mode = sandbox_mode
            item = _Runtime(state, state.mode_id)
            self.runtimes[state.session_id] = item
            return item

        self.app = AppServer(workspace=root, session_store=self.store, agent_service=service,
            runtime_factory=runtime, modes=COLLABORATION_MODES,
            models=(ConfiguredModel("test", "test-model", "Offline", 10000, 1000),),
            credentials_path=root / "never-written-credentials.json",
            credential_setter=lambda provider, key: self.credentials.append((provider, key)))

    def close(self):
        self.app.close()

    def request(self, method="session.create", request_id="once", **params):
        return self.app.handle(RequestEnvelope(method=method, request_id=request_id, params=params))


def child(root: Path, operation: str) -> subprocess.CompletedProcess[str]:
    bootstrap = (
        "import runpy,sys; sys.path[:0] = " + repr(sys.path[:3]) + "; "
        "sys.argv = " + repr([__file__, operation, str(root)]) + "; "
        "runpy.run_path(" + repr(__file__) + ", run_name='__main__')"
    )
    return subprocess.run([sys.executable, "-I", "-S", "-c", bootstrap],
                          capture_output=True, text=True, timeout=20)


class RpcAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = Fixture(self.root)

    def tearDown(self):
        self.fixture.close()
        self.temp.cleanup()

    def require_ok(self, response):
        self.assertTrue(response.ok, response.error)
        return response

    def require_error(self, response, code):
        self.assertFalse(response.ok, "request unexpectedly succeeded")
        self.assertEqual(response.error.code, code)

    def test_duplicate_create(self):
        first = self.require_ok(self.fixture.request())
        second = self.fixture.request()
        self.assertEqual(len(self.fixture.store.list_sessions()), 1, "duplicate session created")
        self.assertEqual(first, second)

    def test_duplicate_turn(self):
        first = self.require_ok(self.fixture.request())
        sid = first.result["session"]["session_id"]
        request = RequestEnvelope("turn.start", {"session_id": sid, "user_input": "offline"}, "start")
        accepted = self.require_ok(self.fixture.app.handle(request))
        self.assertEqual(self.fixture.app.handle(request), accepted)
        self.assertEqual(len(self.fixture.runtimes[sid].commands), 1)

    def test_conflicting_params(self):
        self.require_ok(self.fixture.request(mode_id="build"))
        self.require_error(self.fixture.request(mode_id="plan"), "idempotency_conflict")
        self.assertEqual(len(self.fixture.store.list_sessions()), 1)

    def test_conflicting_method(self):
        sid = self.require_ok(self.fixture.request()).result["session"]["session_id"]
        response = self.fixture.request("session.rename", session_id=sid, title="wrong mutation")
        self.require_error(response, "idempotency_conflict")
        self.assertNotEqual(self.fixture.store.load(sid).title, "wrong mutation")

    def test_parameter_order(self):
        first = self.require_ok(self.fixture.request(mode_id="build", model_ref="test/test-model"))
        second = self.fixture.request(model_ref="test/test-model", mode_id="build")
        self.assertEqual(first, second)
        self.assertEqual(len(self.fixture.store.list_sessions()), 1)

    def test_restart_replays_response(self):
        first = response_to_dict(self.require_ok(self.fixture.request()))
        self.fixture.close()
        result = child(self.root, "request")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), first)
        self.assertEqual(len(SessionStore(self.root / "sessions").list_sessions()), 1)

    def crash_boundary(self, phase, count):
        self.fixture.close()
        result = child(self.root, phase)
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertEqual((self.root / "crash-marker").read_text(), phase)
        self.fixture = Fixture(self.root)
        with patch.object(self.fixture.app, "_dispatch", wraps=self.fixture.app._dispatch) as dispatch:
            response = self.fixture.request()
        self.require_error(response, "request_outcome_unknown")
        self.assertEqual(dispatch.call_count, 0, "unknown request was dispatched again")
        self.assertEqual(len(self.fixture.store.list_sessions()), count)

    def test_crash_before_dispatch(self):
        self.crash_boundary("crash-before", 0)

    def test_crash_after_side_effect(self):
        self.crash_boundary("crash-after", 1)

    def test_failed_admission_does_not_dispatch(self):
        @contextmanager
        def broken_transaction(_database):
            raise StorageError("injected disk failure")
            yield  # pragma: no cover
        with patch.object(Database, "transaction", broken_transaction), \
             patch.object(self.fixture.app, "_dispatch", wraps=self.fixture.app._dispatch) as dispatch:
            response = self.fixture.request()
        self.require_error(response, "storage_error")
        self.assertEqual(dispatch.call_count, 0)
        self.assertEqual(self.fixture.store.list_sessions(), [])

    def test_failed_response_persistence_stays_unknown(self):
        original_dispatch = self.fixture.app._dispatch
        original_transaction = Database.transaction

        @contextmanager
        def broken_transaction(_database):
            raise StorageError("injected response persistence failure")
            yield  # pragma: no cover

        def dispatch(request):
            result = original_dispatch(request)
            Database.transaction = broken_transaction
            return result

        try:
            with patch.object(self.fixture.app, "_dispatch", side_effect=dispatch):
                response = self.fixture.request()
        finally:
            Database.transaction = original_transaction
        self.require_error(response, "storage_error")
        self.require_error(self.fixture.request(), "request_outcome_unknown")
        self.assertEqual(len(self.fixture.store.list_sessions()), 1)

    def test_completed_errors_are_replayed(self):
        with patch.object(self.fixture.app, "_dispatch", wraps=self.fixture.app._dispatch) as dispatch:
            first = self.fixture.request(model_ref="test/missing")
            second = self.fixture.request(model_ref="test/missing")
        self.require_error(first, "invalid_params")
        self.assertEqual(first, second)
        self.assertEqual(dispatch.call_count, 1)

    def test_reads_remain_fresh_and_new_ids_are_independent(self):
        self.assertEqual(self.fixture.request("session.list", "list").result["sessions"], [])
        self.require_ok(self.fixture.request(request_id="first"))
        self.require_ok(self.fixture.request(request_id="second"))
        response = self.require_ok(self.fixture.request("session.list", "list"))
        self.assertEqual(len(response.result["sessions"]), 2)

    def test_duplicate_credentials(self):
        sid = self.require_ok(self.fixture.request()).result["session"]["session_id"]
        first = self.require_ok(self.fixture.request("provider.credential.set", "credential",
                                                     session_id=sid, api_key="fake-repeated-key"))
        self.assertEqual(first, self.fixture.request("provider.credential.set", "credential",
                                                     api_key="fake-repeated-key", session_id=sid))
        self.assertEqual(len(self.fixture.credentials), 1)

    def test_secret_not_persisted_or_returned(self):
        sid = self.require_ok(self.fixture.request()).result["session"]["session_id"]
        secret = "RPC-EVAL-FAKE-CREDENTIAL-DO-NOT-PERSIST"
        first = self.require_ok(self.fixture.request("provider.credential.set", "credential",
                                                     session_id=sid, api_key=secret))
        self.assertNotIn(secret, json.dumps(response_to_dict(first)))
        for file in self.root.rglob("*"):
            if file.is_file():
                self.assertNotIn(secret.encode(), file.read_bytes(), str(file))

    def test_schema3_session_is_preserved(self):
        self.fixture.close()
        legacy = self.root / "legacy"
        database_root = legacy / "sessions"
        database_root.mkdir(parents=True)
        state = SessionState("historical", str(legacy), database_root / "historical", "old", "old")
        state.title = "Keep this session"
        # Minimal genuine schema-3 sessions table; Database creates other base tables as needed.
        with sqlite3.connect(database_root / "agent.sqlite3") as connection:
            connection.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, data TEXT NOT NULL)")
            connection.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                               (state.session_id, 1, json.dumps(session_to_dict(state))))
            connection.execute("PRAGMA user_version=3")
        self.fixture = Fixture(legacy)
        self.assertEqual(self.fixture.store.load("historical").title, "Keep this session")
        self.require_ok(self.fixture.request())
        self.assertEqual(len(self.fixture.store.list_sessions()), 2)


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"test": test.id(), "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({"test": test.id(), "status": "failed", "detail": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({"test": test.id(), "status": "error", "detail": self._exc_info_to_string(err, test)})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.records.append({"test": test.id(), "status": "skipped", "detail": reason})


def main():
    if len(sys.argv) > 1 and sys.argv[1] != "grade":
        operation, root = sys.argv[1], Path(sys.argv[2])
        fixture = Fixture(root)
        if operation.startswith("crash-"):
            original = fixture.app._dispatch
            def crash(request):
                if operation == "crash-after":
                    original(request)
                (root / "crash-marker").write_text(operation)
                os._exit(78)
            fixture.app._dispatch = crash
        try:
            print(json.dumps(response_to_dict(fixture.request())))
        finally:
            fixture.close()
        return
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(RpcAcceptance)
    # Frozen pre-fix tests, not the candidate's possibly modified tests.
    for module in ("test_app_server", "test_persistence", "test_recovery", "test_service"):
        suite.addTests(loader.loadTestsFromName(module))
    result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=RecordingResult).run(suite)
    print(json.dumps({"tests": result.records, "run": result.testsRun,
                      "successful": result.wasSuccessful() and not result.skipped}))
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)


if __name__ == "__main__":
    main()
