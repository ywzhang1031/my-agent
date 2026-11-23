from __future__ import annotations

import codecs
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import BinaryIO
from uuid import uuid4

from .database import Database, StorageError
from .messages import ToolResultMessage


MAX_STREAM_BYTES = 1024 * 1024
MAX_PAGE_BYTES = 8192
MAX_SESSION_BYTES = 32 * 1024 * 1024
MAX_DATABASE_BYTES = 256 * 1024 * 1024
MAX_SESSION_ARTIFACTS = 256
MAX_DATABASE_ARTIFACTS = 4096


class OutputQuotaError(StorageError):
    pass


class ToolOutputStore:
    """Bounded raw output, scoped by the runtime's session, never by model-supplied paths."""

    def __init__(self, root: Path, session_id: str) -> None:
        self.root = root
        self.session_id = session_id
        self._database: Database | None = None

    @property
    def database(self) -> Database:
        if self._database is None:
            self._database = Database(self.root)
        return self._database

    def save(self, stdout: BinaryIO, stderr: BinaryIO, *, turn_id: str, call_id: str) -> dict:
        data, streams = {}, {}
        for name, source in (("stdout", stdout), ("stderr", stderr)):
            source.seek(0, 2)
            total = source.tell()
            source.seek(0)
            data[name] = source.read(MAX_STREAM_BYTES)
            streams[name] = {
                "total_bytes": total, "stored_bytes": len(data[name]),
                "storage_truncated": total > len(data[name]),
                "sha256": hashlib.sha256(data[name]).hexdigest(),
            }
        metadata = {"artifact_id": uuid4().hex, "turn_id": turn_id, "call_id": call_id,
                    "created_at": datetime.now(timezone.utc).isoformat(), "streams": streams}
        size = sum(len(value) for value in data.values())
        with self.database.transaction() as connection:
            # BEGIN IMMEDIATE serializes quota admission with insertion, including across processes.
            for session, max_bytes, max_items in (
                (self.session_id, MAX_SESSION_BYTES, MAX_SESSION_ARTIFACTS),
                (None, MAX_DATABASE_BYTES, MAX_DATABASE_ARTIFACTS),
            ):
                sql = "SELECT count(*),coalesce(sum(length(stdout)+length(stderr)),0) FROM tool_outputs"
                count, used = connection.execute(sql + (" WHERE session_id=?" if session else ""),
                                                 (session,) if session else ()).fetchone()
                if count >= max_items or used + size > max_bytes:
                    raise OutputQuotaError("tool output retention quota reached; existing artifacts were not evicted")
            connection.execute("INSERT INTO tool_outputs VALUES (?, ?, ?, ?, ?)",
                (metadata["artifact_id"], self.session_id, json.dumps(metadata), data["stdout"], data["stderr"]))
        return metadata

    def confirms_observation(self, turn_id: str, message: ToolResultMessage, artifact_id: str) -> bool:
        # A stored artifact alone does not prove the tool-result checkpoint committed.
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT metadata FROM tool_outputs WHERE session_id=? AND artifact_id=? AND EXISTS "
                "(SELECT 1 FROM tool_attempts WHERE session_id=? AND turn_id=? AND call_id=? "
                "AND tool_name=? AND status='completed' AND result=?)",
                (self.session_id, artifact_id, self.session_id, turn_id, message.tool_call_id,
                 message.tool_name, message.content),
            ).fetchone()
        if not row:
            return False
        artifact = json.loads(row[0])
        metadata = json.loads(message.content)["metadata"]
        if message.tool_name == "exec_command":
            return metadata["output_artifact"] == {"status": "stored", **artifact}
        stream = artifact["streams"].get(metadata.get("stream"))
        return bool(stream and all(metadata.get(k) == v for k, v in stream.items())
                    and all(metadata.get(k) == artifact[k] for k in ("call_id", "turn_id"))
                    and type(metadata.get("offset")) is int and type(metadata.get("next_offset")) is int
                    and 0 <= metadata["offset"] <= metadata["next_offset"] <= stream["stored_bytes"])

    def read(self, artifact_id: str, stream: str, offset: int = 0, limit: int = 4096) -> dict:
        if not isinstance(artifact_id, str) or not re.fullmatch(r"[0-9a-f]{32}", artifact_id):
            raise ValueError("artifact_id must be a 32-character lowercase hex ID, not a path")
        if stream not in ("stdout", "stderr"):
            raise ValueError("stream must be stdout or stderr")
        if type(offset) is not int or not 0 <= offset <= MAX_STREAM_BYTES:
            raise ValueError(f"offset must be an integer between 0 and {MAX_STREAM_BYTES}")
        if type(limit) is not int or not 4 <= limit <= MAX_PAGE_BYTES:
            raise ValueError(f"limit must be an integer between 4 and {MAX_PAGE_BYTES}")
        with self.database.transaction() as connection:
            row = connection.execute(
                f"SELECT metadata,substr({stream},?,?) FROM tool_outputs WHERE artifact_id=? AND session_id=?",
                (offset + 1, limit, artifact_id, self.session_id),
            ).fetchone()
        if row is None:
            raise FileNotFoundError("tool output not found in this session")
        metadata, raw = json.loads(row[0]), row[1] or b""
        details = metadata["streams"][stream]
        if offset > details["stored_bytes"]:
            raise ValueError("offset exceeds retained output; use stored_bytes from the artifact metadata")
        # Defer an incomplete UTF-8 suffix to the next page so normal paging never splits a character.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        text = decoder.decode(raw, final=offset + len(raw) >= details["stored_bytes"])
        next_offset = offset + len(raw) - len(decoder.getstate()[0])
        return {"text": text, "artifact_id": artifact_id, "stream": stream, "offset": offset,
                "next_offset": next_offset, "eof": next_offset == details["stored_bytes"],
                **details, "turn_id": metadata["turn_id"], "call_id": metadata["call_id"]}
