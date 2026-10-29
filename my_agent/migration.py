from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from .database import DATABASE_NAME, Database, StorageError
from .event_store import EventStore
from .protocol import event_from_dict
from .server_instance import ServerInstanceLease
from .session import SessionStore, session_from_dict


def migrate_sessions(root: str | Path) -> dict[str, object]:
    root = Path(root).expanduser().resolve()
    lease = ServerInstanceLease.acquire(root, instance_id="migration", workspace=root)
    try:
        if (root / DATABASE_NAME).exists():
            Database(root)
            return {"status": "already_migrated", "database": str(root / DATABASE_NAME)}
        sources = sorted(root.glob("*/session.json"))
        backup = root / f".legacy-backup-{uuid4().hex}"
        backup.mkdir(mode=0o700)
        manifest = {}
        for source in sources:
            if source.parent.is_symlink():
                raise StorageError(f"refusing symlink session: {source.parent}")
            destination = backup / source.parent.name
            destination.mkdir(mode=0o700)
            for name in ("session.json", "messages.jsonl", "events-v2.jsonl", "trace.jsonl", "trajectory.json"):
                path = source.parent / name
                if path.is_symlink():
                    raise StorageError(f"refusing symlink data: {path}")
                if path.is_file():
                    copied = destination / name
                    shutil.copyfile(path, copied)
                    os.chmod(copied, 0o600)
                    with copied.open("rb") as stream:
                        os.fsync(stream.fileno())
                    manifest[str(path.relative_to(root))] = hashlib.sha256(copied.read_bytes()).hexdigest()
        manifest_path = backup / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.chmod(manifest_path, 0o600)
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())

        with tempfile.TemporaryDirectory(prefix=".migration-", dir=root) as staging:
            store = SessionStore(staging)
            events = EventStore(staging)
            with store.database.transaction() as connection:
                for source in sources:
                    session_id = source.parent.name
                    data = json.loads((backup / session_id / "session.json").read_text(encoding="utf-8"))
                    if data.get("session_id") != session_id:
                        raise StorageError(f"session id mismatch: {source}")
                    data["messages"] = _jsonl(backup / session_id / "messages.jsonl")
                    if data.get("message_count", len(data["messages"])) != len(data["messages"]):
                        raise StorageError(f"message count mismatch: {source}")
                    state = session_from_dict(data, store._session_dir(session_id))
                    for item in _jsonl(backup / session_id / "events-v2.jsonl"):
                        event = event_from_dict(item)
                        if event.session_id != session_id or not event.durable:
                            raise StorageError(f"invalid legacy event: {session_id}")
                        events.append(event)
                        if event.turn_id and event.turn_id.startswith(session_id + ":"):
                            suffix = event.turn_id[len(session_id) + 1:]
                            if suffix.isdigit():
                                state.turn_counter = max(state.turn_counter, int(suffix))
                    if state.pending_turn is not None:
                        state.pending_turn.pause_reason = "recovery_required"
                        state.pending_turn.last_error = "Imported incomplete turn; inspect workspace before continuing."
                    store.save(state)
                    store.load(session_id)
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise StorageError("migrated database failed integrity check")
            os.replace(store.database.path, root / DATABASE_NAME)
            descriptor = os.open(root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return {"status": "migrated", "sessions": len(sources), "backup": str(backup),
                "database": str(root / DATABASE_NAME)}
    finally:
        lease.release()


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("expected an object")
        except ValueError as exc:
            raise StorageError(f"invalid legacy log {path}:{number}: {exc}") from exc
        records.append(item)
    return records
