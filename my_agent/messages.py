from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserMessage:
    content: str


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class AssistantMessage:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str = ""


@dataclass
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: str


Message = UserMessage | AssistantMessage | ToolResultMessage
