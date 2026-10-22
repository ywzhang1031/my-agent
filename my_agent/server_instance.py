from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


LOCK_FILENAME = ".app-server.lock"


class ServerInstanceConflictError(RuntimeError):
    pass


class ServerInstanceLease:
    """An OS-backed exclusive lease for one AppServer session store."""

    def __init__(self, path: Path, stream: TextIO) -> None:
        self.path = path
        self._stream: TextIO | None = stream

    @classmethod
    def acquire(
        cls,
        sessions_root: str | Path,
        *,
        instance_id: str,
        workspace: str | Path,
    ) -> ServerInstanceLease:
        root = Path(sessions_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = root / LOCK_FILENAME
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "r+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner = cls._read_owner(stream)
            stream.close()
            details = cls._owner_description(owner)
            raise ServerInstanceConflictError(
                f"session store is already in use by another AppServer{details}: "
                f"{root}. Reuse the running server or choose a different --sessions-dir."
            ) from exc
        except Exception:
            stream.close()
            raise

        metadata = {
            "instance_id": instance_id,
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "workspace": str(Path(workspace).expanduser().resolve()),
            "sessions_root": str(root),
        }
        stream.seek(0)
        stream.truncate()
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        return cls(path, stream)

    def release(self) -> None:
        stream = self._stream
        if stream is None:
            return
        self._stream = None
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    @staticmethod
    def _read_owner(stream: TextIO) -> dict[str, object]:
        try:
            stream.seek(0)
            value = json.load(stream)
        except (json.JSONDecodeError, OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _owner_description(owner: dict[str, object]) -> str:
        details = []
        pid = owner.get("pid")
        instance_id = owner.get("instance_id")
        started_at = owner.get("started_at")
        if isinstance(pid, int):
            details.append(f"pid={pid}")
        if isinstance(instance_id, str) and instance_id:
            details.append(f"instance={instance_id}")
        if isinstance(started_at, str) and started_at:
            details.append(f"started_at={started_at}")
        return f" ({', '.join(details)})" if details else ""
