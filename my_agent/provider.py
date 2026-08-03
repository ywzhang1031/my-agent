from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .messages import Message, ToolCall
from .tools import ToolSpec


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class ProviderResponse:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: TokenUsage | None = None
    raw_response: dict[str, Any] | None = None


class Provider(Protocol):
    def send(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
    ) -> ProviderResponse:
        ...


class ScriptedProvider:
    def __init__(self, replies: list[ProviderResponse]) -> None:
        self._replies = list(replies)
        self.requests: list[list[Message]] = []

    def send(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
    ) -> ProviderResponse:
        self.requests.append(list(messages))
        if not self._replies:
            return ProviderResponse(content="No scripted replies remain.")
        return self._replies.pop(0)
