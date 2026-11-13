from contextlib import closing
import hashlib
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from my_agent.database import Database, SCHEMA_VERSION
from my_agent.tool_output import MAX_STREAM_BYTES, OutputQuotaError, ToolOutputStore


class ToolOutputStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.store = ToolOutputStore(self.root, "session-a")

    def save(self, stdout=b"hello", stderr=b"error"):
        return self.store.save(io.BytesIO(stdout), io.BytesIO(stderr), turn_id="session-a:1", call_id="call-1")

    def test_output_survives_reopen_and_preserves_raw_bytes(self):
        data = b"abc\x00\xff" + "中文".encode()
        artifact = self.save(data)
        reopened = ToolOutputStore(self.root, "session-a")
        page = reopened.read(artifact["artifact_id"], "stdout")
        self.assertEqual(page["text"], data.decode("utf-8", errors="replace"))
        self.assertEqual(page["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(page["next_offset"], len(data))
        self.assertTrue(page["eof"])
        self.assertEqual(reopened.read(artifact["artifact_id"], "stderr")["text"], "error")
        self.assertEqual(reopened.database.path.stat().st_mode & 0o777, 0o600)

    def test_unicode_pages_reassemble_without_character_loss(self):
        text = "abc中文\n测试🙂end"
        artifact = self.save(text.encode())
        offset, parts = 0, []
        while True:
            page = self.store.read(artifact["artifact_id"], "stdout", offset, 4)
            parts.append(page["text"])
            self.assertGreater(page["next_offset"], offset)
            offset = page["next_offset"]
            if page["eof"]:
                break
        self.assertEqual("".join(parts), text)

    def test_retention_cap_is_explicit_and_hash_covers_only_retained_bytes(self):
        artifact = self.save(b"x" * (MAX_STREAM_BYTES + 17))
        info = artifact["streams"]["stdout"]
        self.assertEqual(info["total_bytes"], MAX_STREAM_BYTES + 17)
        self.assertEqual(info["stored_bytes"], MAX_STREAM_BYTES)
        self.assertTrue(info["storage_truncated"])
        self.assertEqual(info["sha256"], hashlib.sha256(b"x" * MAX_STREAM_BYTES).hexdigest())
        page = self.store.read(artifact["artifact_id"], "stdout", MAX_STREAM_BYTES - 4, 4)
        self.assertTrue(page["eof"])
        self.assertTrue(page["storage_truncated"])

    def test_cross_session_and_unknown_ids_return_the_same_error(self):
        artifact = self.save()
        other = ToolOutputStore(self.root, "session-b")
        for artifact_id in (artifact["artifact_id"], "0" * 32):
            with self.assertRaisesRegex(FileNotFoundError, "not found in this session"):
                other.read(artifact_id, "stdout")

    def test_paths_invalid_ranges_and_boolean_offsets_are_rejected(self):
        artifact_id = self.save()["artifact_id"]
        for args in (("../secret", "stdout", 0, 4), (artifact_id, "secret", 0, 4),
                     (artifact_id, "stdout", True, 4), (artifact_id, "stdout", -1, 4),
                     (artifact_id, "stdout", 0, 8193), (artifact_id, "stdout", 0, 0),
                     (artifact_id, "stdout", 6, 4)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.store.read(*args)

    def test_empty_output_and_end_of_file_are_unambiguous(self):
        artifact = self.save(b"", b"")
        page = self.store.read(artifact["artifact_id"], "stdout")
        self.assertEqual(page["text"], "")
        self.assertEqual(page["next_offset"], 0)
        self.assertTrue(page["eof"])
        self.assertFalse(page["storage_truncated"])

    def test_quotas_reject_new_output_without_deleting_previous_evidence(self):
        artifact = self.save()
        for constant in ("MAX_SESSION_BYTES", "MAX_DATABASE_BYTES", "MAX_SESSION_ARTIFACTS", "MAX_DATABASE_ARTIFACTS"):
            with self.subTest(constant=constant), patch("my_agent.tool_output." + constant, 1):
                with self.assertRaises(OutputQuotaError):
                    self.save()
            self.assertEqual(self.store.read(artifact["artifact_id"], "stdout")["text"], "hello")
        with self.store.database.transaction() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM tool_outputs").fetchone()[0], 1)

    def test_outer_transaction_rollback_leaves_no_artifact(self):
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            with self.store.database.transaction():
                artifact = self.save()
                raise RuntimeError("rollback")
        with self.assertRaises(FileNotFoundError):
            self.store.read(artifact["artifact_id"], "stdout")

    def test_schema_five_upgrade_preserves_existing_session_rows(self):
        database = Database(self.root)
        with closing(sqlite3.connect(database.path)) as connection:
            connection.execute("DROP TABLE tool_outputs")
            connection.execute("INSERT INTO sessions VALUES ('existing', 1, '{}')")
            connection.execute("PRAGMA user_version=5")
            connection.commit()
        Database(self.root)
        with closing(sqlite3.connect(database.path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(connection.execute("SELECT data FROM sessions WHERE session_id='existing'").fetchone()[0], "{}")
            self.assertEqual(connection.execute("SELECT count(*) FROM tool_outputs").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
