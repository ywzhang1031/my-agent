"""Versioned request projection. Original observations remain the recovery journal."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

from .messages import AssistantMessage, Message, ToolResultMessage, UserMessage


@dataclass(frozen=True)
class OutputCheckpoint:
    index: int
    fingerprint: str
    artifact_id: str
    version: int = 1


def older_observations(messages: list[Message], start: int) -> range:
    """Only consider complete call/result batches; protect the entire newest one."""
    if not 0 <= start < len(messages) or not isinstance(messages[start], UserMessage):
        return range(0)
    i = newest = start + 1
    while i < len(messages):
        assistant = messages[i]
        if not isinstance(assistant, AssistantMessage) or not assistant.tool_calls:
            return range(0)
        newest = i
        i += 1
        for call in assistant.tool_calls:
            obs = messages[i] if i < len(messages) else None
            if not isinstance(obs, ToolResultMessage) or (obs.tool_call_id, obs.tool_name) != (call.call_id, call.name):
                return range(0)
            i += 1
    return range(start + 1, newest)


def fingerprint(message: ToolResultMessage) -> str:
    identity = [message.tool_name, message.tool_call_id, message.content]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


def compact_v1(message: ToolResultMessage, turn_id: str) -> tuple[str, ToolResultMessage] | None:
    if message.tool_name not in {"exec_command", "read_tool_output"}:
        return None
    try:
        result = json.loads(message.content)
        meta = result["metadata"]
        if message.tool_name == "exec_command":
            if meta["status"] not in {"success", "nonzero_exit"} or type(result["exit_code"]) is not int:
                return None
            ref = meta["output_artifact"]
            if (ref["status"] != "stored" or ref["turn_id"] != turn_id or ref["call_id"] != message.tool_call_id
                    or any(ref["streams"][s]["storage_truncated"] is not False for s in ("stdout", "stderr"))):
                return None
        else:
            if meta["status"] != "output_read" or not result["ok"] or meta["storage_truncated"] is not False:
                return None
            ref = meta
        artifact_id = ref["artifact_id"]
        if not isinstance(artifact_id, str):
            return None
        # Leave short diagnostics intact; mark clipped previews explicitly.
        clipped = False
        for stream in ("stdout", "stderr"):
            value = result[stream]
            if not isinstance(value, str):
                return None
            if len(value) > 192:
                result[stream] = value[:64] + "\n[... use read_tool_output ...]\n" + value[-128:]
                clipped = True
        if not clipped:
            return None
        result["truncated"] = True
        meta["context_compacted"] = True
        return artifact_id, replace(message, content=json.dumps(result, ensure_ascii=False))
    except (KeyError, TypeError, ValueError):
        return None
