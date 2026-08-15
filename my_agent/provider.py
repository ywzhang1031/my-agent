from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

from .control import raise_if_cancelled
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


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


@dataclass(frozen=True)
class ProviderEvent:
    kind: str
    text: str = ""
    response: ProviderResponse | None = None
    data: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    context_window_tokens: int
    max_output_tokens: int

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[ProviderEvent]:
        ...


class ScriptedProvider:
    def __init__(
        self,
        replies: list[ProviderResponse | Exception],
        *,
        context_window_tokens: int = 1_000_000,
        max_output_tokens: int = 64_000,
    ) -> None:
        self._replies = list(replies)
        self.context_window_tokens = context_window_tokens
        self.max_output_tokens = max_output_tokens
        self.requests: list[list[Message]] = []
        self.system_prompts: list[str] = []

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[ProviderEvent]:
        raise_if_cancelled(cancel_event)
        self.requests.append(list(messages))
        self.system_prompts.append(system_prompt)
        if not self._replies:
            reply: ProviderResponse | Exception = ProviderResponse(
                content="No scripted replies remain."
            )
        else:
            reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if reply.reasoning_content:
            raise_if_cancelled(cancel_event)
            yield ProviderEvent(kind="reasoning_delta", text=reply.reasoning_content)
        if reply.content:
            raise_if_cancelled(cancel_event)
            yield ProviderEvent(kind="content_delta", text=reply.content)
        raise_if_cancelled(cancel_event)
        yield ProviderEvent(kind="completed", response=reply)
