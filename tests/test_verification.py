import json
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from my_agent.attempt_store import AttemptStore
from my_agent.cli import _print_verification, _report_verification
from my_agent.database import Database
from my_agent.session import SessionStore
from my_agent.verification import read_verification


class VerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = SessionStore(Path(temporary.name) / "sessions")
        self.state = self.store.create(temporary.name)
        self.state.turn_counter = 1
        self.store.save(self.state)
        self.turn = self.state.session_id + ":1"
        self.attempts = AttemptStore(self.store.database)
        self.counter = 0

    def record(self, name, result=None, arguments=None, turn=None):
        self.counter += 1
        data = {"attempt_id": str(self.counter), "call_id": "same-provider-id", "name": name,
                "arguments": arguments or {"argv": ["pytest"], "cwd": "."}}
        self.attempts.start(self.state.session_id, turn or self.turn, data)
        if result is not None:
            self.attempts.finish(self.state.session_id, turn or self.turn, {**data, "result": result})

    def patch(self, path="main.py"):
        self.record("apply_patch", {"ok": True, "metadata": {
            "applied": True, "changed_files": [path], "operation": "update"}}, {"patch": "not displayed"})

    def check(self, code=0, argv=None, **metadata):
        self.record("exec_command", {"ok": code == 0, "exit_code": code, "stdout": "private output",
                    "metadata": {"status": "success" if code == 0 else "nonzero_exit", "category": "test", **metadata}},
                    {"argv": argv or ["pytest"], "cwd": "."})

    def view(self):
        return read_verification(self.store.database, self.state.session_id)

    def test_no_tool_evidence_is_not_a_verified_result(self):
        self.assertEqual(self.view()["status"], "no_evidence")

    def test_cli_does_not_report_task_acceptance(self):
        self.patch()
        self.check()
        output = io.StringIO()
        _print_verification(self.view(), file=output)
        self.assertIn("checks_passed", output.getvalue())
        self.assertIn("not task acceptance", output.getvalue())
        self.assertIn("main.py", output.getvalue())
        self.assertIn("exit=0", output.getvalue())

    def test_reporting_failure_does_not_replace_the_turn_result(self):
        output = io.StringIO()
        with patch("my_agent.cli.read_verification", side_effect=OSError("disk unavailable")):
            _report_verification(self.store, self.state.session_id, file=output)
        self.assertIn("verification unavailable", output.getvalue())

    def test_edit_requires_validation_and_later_edit_invalidates_evidence(self):
        self.patch()
        self.assertEqual(self.view()["status"], "unverified")
        self.check()
        self.assertEqual(self.view()["status"], "checks_passed")
        self.patch()
        view = self.view()
        self.assertEqual(view["status"], "stale")
        self.assertEqual(view["checks"][0]["freshness"], "stale")
        self.assertEqual(len(view["changed_files"]), 1)
        self.assertEqual(view["external_changes"], "not_observed")
        self.assertNotIn("private output", json.dumps(view))
        self.assertNotIn("not displayed", json.dumps(view))

    def test_passed_unrelated_check_does_not_hide_failure(self):
        self.patch()
        self.check(1)
        self.check(argv=["pytest", "other.py"])
        self.assertEqual(self.view()["status"], "failed")
        self.check()
        self.assertEqual(self.view()["status"], "checks_passed")
        self.assertEqual(len(self.view()["checks"]), 3)

    def test_blocked_check_does_not_count_as_passed_or_executed(self):
        self.check()
        self.record("exec_command", {"ok": False, "metadata": {"status": "denied"}}, {"argv": ["pytest", "x.py"]})
        self.assertEqual(self.view()["status"], "unverified")
        self.assertEqual(self.view()["checks"][-1]["status"], "not_run")

    def test_unknown_tool_result_and_interruption_remain_unknown(self):
        self.patch()
        self.record("apply_patch")
        self.check()
        self.assertEqual(self.view()["status"], "unknown")
        self.assertEqual(len(self.view()["unknown_attempts"]), 1)
        self.assertEqual(self.view()["unknown_attempts"][0]["reason"], "result_not_committed")

    def test_zero_exit_with_timeout_is_not_passed(self):
        self.check(timed_out=True)
        self.assertEqual(self.view()["status"], "unknown")

    def test_reopen_and_turn_isolation(self):
        self.patch()
        self.check()
        before = self.view()
        reopened = Database(self.store.root)
        self.assertEqual(read_verification(reopened, self.state.session_id), before)
        self.record("apply_patch", turn=self.state.session_id + ":2")
        self.assertEqual(self.view(), before)
        with self.assertRaises(ValueError):
            read_verification(reopened, self.state.session_id, "another:1")
        self.assertIsNotNone(before["checks"][0]["finished_at"])

    def test_rolled_back_attempt_cannot_become_evidence(self):
        with self.assertRaises(OSError):
            with self.store.database.transaction():
                self.patch()
                self.check()
                raise OSError("disk failure")
        self.assertEqual(self.view()["status"], "no_evidence")

    def test_v4_database_adds_nullable_timestamps_without_fabrication(self):
        root = self.store.root.parent / "v4"
        root.mkdir()
        with sqlite3.connect(root / "agent.sqlite3") as connection:
            connection.execute("CREATE TABLE tool_attempts (attempt_id TEXT PRIMARY KEY,session_id TEXT,turn_id TEXT,call_id TEXT,tool_name TEXT,arguments TEXT,status TEXT,result TEXT)")
            connection.execute("INSERT INTO tool_attempts VALUES ('old','session','session:1','call','read_file','{}','completed','{}')")
            connection.execute("PRAGMA user_version=4")
        database = Database(root)
        with database.transaction() as connection:
            self.assertEqual(connection.execute("SELECT attempt_id,started_at,finished_at FROM tool_attempts").fetchone(), ("old", None, None))
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 5)
