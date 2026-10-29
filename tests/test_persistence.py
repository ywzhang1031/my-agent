import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from my_agent.database import DATABASE_NAME, RevisionConflict, StorageError
from my_agent.event_store import EventStore
from my_agent.messages import UserMessage
from my_agent.migration import migrate_sessions
from my_agent.protocol import AgentState, SessionRenamedEvent
from my_agent.server_instance import ServerInstanceLease, ServerInstanceConflictError
from my_agent.session import SessionStore, session_to_dict


class PersistenceTests(unittest.TestCase):
    def test_migration_cli_prints_json_result(self):
        destination = self.root / "cli-sessions"
        result = subprocess.run(
            [sys.executable, "cli.py", "migrate", "--sessions-dir", str(destination)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "migrated")
        self.assertTrue((destination / DATABASE_NAME).is_file())

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = SessionStore(self.root / "sessions")
        self.state = self.store.create(self.root)

    def test_session_and_event_commit_or_rollback_together(self):
        events = EventStore(self.store.root)
        revision = self.state.revision
        self.state.title = "Uncommitted"
        self.state.messages.append(UserMessage("New message"))
        with self.assertRaisesRegex(OSError, "disk failure"):
            with self.store.database.transaction():
                self.store.save(self.state)
                events.append(SessionRenamedEvent(
                    session_id=self.state.session_id, turn_id=None, sequence=1,
                    state=AgentState.IDLE, title=self.state.title,
                ))
                raise OSError("disk failure")
        restored = self.store.load(self.state.session_id)
        self.assertEqual(restored.title, "")
        self.assertEqual(restored.messages, [])
        self.assertEqual(events.replay(self.state.session_id), [])
        self.assertEqual(self.state.revision, revision)
        with self.store.database.transaction():
            self.store.save(self.state)
            events.append(SessionRenamedEvent(
                session_id=self.state.session_id, turn_id=None, sequence=1,
                state=AgentState.IDLE, title=self.state.title,
            ))
        self.assertEqual(self.store.load(self.state.session_id).title, "Uncommitted")
        self.assertEqual(len(events.replay(self.state.session_id)), 1)

    def test_stale_writer_cannot_overwrite_newer_state(self):
        stale = self.store.load(self.state.session_id)
        self.state.title = "Newer"
        self.store.save(self.state)
        stale.title = "Older"
        with self.assertRaises(RevisionConflict):
            self.store.save(stale)
        self.assertEqual(self.store.load(stale.session_id).title, "Newer")

    def test_process_death_rolls_back_uncommitted_checkpoint(self):
        script = """
import os, sys
from my_agent.session import SessionStore
store = SessionStore(sys.argv[1])
state = store.load(sys.argv[2])
with store.database.transaction():
    state.title = 'Must not survive'
    store.save(state)
    os._exit(23)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.store.root), self.state.session_id],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(self.store.load(self.state.session_id).title, "")
        with self.store.database.transaction() as connection:
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.assertEqual(self.store.database.path.stat().st_mode & 0o777, 0o600)

    def legacy_fixture(self):
        root = self.root / "legacy"
        directory = root / self.state.session_id
        directory.mkdir(parents=True)
        data = session_to_dict(self.state)
        data.pop("messages")
        data["message_count"] = 1
        (directory / "session.json").write_text(json.dumps(data), encoding="utf-8")
        (directory / "messages.jsonl").write_text(
            json.dumps({"role": "user", "content": "Legacy question"}) + "\n", encoding="utf-8"
        )
        return root, directory

    def test_migration_is_lossless_explicit_and_repeatable(self):
        root, directory = self.legacy_fixture()
        original = (directory / "messages.jsonl").read_bytes()
        with self.assertRaisesRegex(StorageError, "require migration"):
            SessionStore(root)
        result = migrate_sessions(root)
        loaded = SessionStore(root).load(self.state.session_id)
        self.assertEqual(loaded.messages[0].content, "Legacy question")
        self.assertEqual(loaded.created_at, self.state.created_at)
        self.assertEqual(loaded.updated_at, self.state.updated_at)
        self.assertEqual((directory / "messages.jsonl").read_bytes(), original)
        self.assertEqual((Path(result["backup"]) / self.state.session_id / "messages.jsonl").read_bytes(), original)
        self.assertEqual(migrate_sessions(root)["status"], "already_migrated")

    def test_bad_legacy_tail_never_installs_partial_database(self):
        root, directory = self.legacy_fixture()
        (directory / "events-v2.jsonl").write_text('{"event":', encoding="utf-8")
        with self.assertRaisesRegex(StorageError, "invalid legacy log"):
            migrate_sessions(root)
        self.assertFalse((root / DATABASE_NAME).exists())
        self.assertEqual((directory / "events-v2.jsonl").read_text(), '{"event":')

    def test_migration_refuses_live_server(self):
        root, _ = self.legacy_fixture()
        lease = ServerInstanceLease.acquire(root, instance_id="test", workspace=self.root)
        try:
            with self.assertRaises(ServerInstanceConflictError):
                migrate_sessions(root)
        finally:
            lease.release()
