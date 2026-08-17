from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Protocol

from .control import raise_if_cancelled
from .messages import Message, ToolCall
from .tools import ToolSpec


class ModelErrorKind(str, Enum):
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_OVERFLOW = "context_overflow"
    CONNECTION = "connection"
    SERVER = "server"
    STREAM = "stream"
    PROTOCOL = "protocol"
    UNKNOWN = "unknown"


class ModelErrorPhase(str, Enum):
    CONFIGURATION = "configuration"
    REQUEST = "request"
    STREAM = "stream"
    DECODING = "decoding"


class StopReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


class ModelError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        kind: ModelErrorKind = ModelErrorKind.UNKNOWN,
        phase: ModelErrorPhase = ModelErrorPhase.REQUEST,
        retryable: bool = False,
        provider_id: str | None = None,
        model_id: str | None = None,
        status_code: int | None = None,
        provider_code: str | None = None,
        request_id: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.phase = phase
        self.retryable = retryable
        self.provider_id = provider_id
        self.model_id = model_id
        self.status_code = status_code
        self.provider_code = provider_code
        self.request_id = request_id
        self.retry_after_seconds = retry_after_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "phase": self.phase.value,
            "retryable": self.retryable,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "status_code": self.status_code,
            "provider_code": self.provider_code,
            "request_id": self.request_id,
            "retry_after_seconds": self.retry_after_seconds,
        }


@dataclass(frozen=True)
class ProviderCapabilities:
    streaming: bool = True
    tool_calling: bool = True
    reasoning: bool = False
    parallel_tool_calls: bool = True


@dataclass(frozen=True)
class ModelProfile:
    provider_id: str
    model_id: str
    context_window_tokens: int
    max_output_tokens: int
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)

    def __post_init__(self) -> None:
        if self.context_window_tokens <= 0 or self.max_output_tokens <= 0:
            raise ValueError("model token limits must be positive")
        if self.max_output_tokens >= self.context_window_tokens:
            raise ValueError("max output tokens must be smaller than the context window")


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    system_prompt: str = ""


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


@dataclass(frozen=True)
class ModelMetadata:
    provider_id: str
    model_id: str
    response_model_id: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    native: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextOutput:
    text: str


@dataclass(frozen=True)
class ReasoningOutput:
    text: str


@dataclass(frozen=True)
class ToolCallOutput:
    call: ToolCall


ModelOutputItem = TextOutput | ReasoningOutput | ToolCallOutput


@dataclass(frozen=True)
class ModelResponse:
    output: tuple[ModelOutputItem, ...]
    stop_reason: StopReason
    usage: ModelUsage | None
    metadata: ModelMetadata

    @property
    def text(self) -> str:
        return "".join(item.text for item in self.output if isinstance(item, TextOutput))

    @property
    def reasoning(self) -> str:
        return "".join(item.text for item in self.output if isinstance(item, ReasoningOutput))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [item.call for item in self.output if isinstance(item, ToolCallOutput)]

    @classmethod
    def from_parts(
        cls,
        *,
        text: str = "",
        reasoning: str = "",
        tool_calls: list[ToolCall] | tuple[ToolCall, ...] = (),
        stop_reason: StopReason = StopReason.STOP,
        usage: ModelUsage | None = None,
        metadata: ModelMetadata,
    ) -> ModelResponse:
        output: list[ModelOutputItem] = []
        if reasoning:
            output.append(ReasoningOutput(reasoning))
        if text:
            output.append(TextOutput(text))
        output.extend(ToolCallOutput(call) for call in tool_calls)
        return cls(
            output=tuple(output),
            stop_reason=stop_reason,
            usage=usage,
            metadata=metadata,
        )


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    call_id_delta: str = ""
    name_delta: str = ""
    arguments_delta: str = ""


@dataclass(frozen=True)
class ModelCompleted:
    response: ModelResponse


ModelEvent = TextDelta | ReasoningDelta | ToolCallDelta | ModelCompleted


class ProviderAdapter(Protocol):
    profile: ModelProfile

    def stream_once(
        self,
        request: ModelRequest,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[ModelEvent]:
        """Perform exactly one provider API attempt."""
        ...


class ScriptedProviderAdapter:
    def __init__(
        self,
        replies: list[ModelResponse | Exception],
        *,
        context_window_tokens: int = 1_000_000,
        max_output_tokens: int = 64_000,
    ) -> None:
        self._replies = list(replies)
        self.profile = ModelProfile(
            provider_id="scripted",
            model_id="scripted-model",
            context_window_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
            capabilities=ProviderCapabilities(reasoning=True),
        )
        self.requests: list[ModelRequest] = []

    def stream_once(
        self,
        request: ModelRequest,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[ModelEvent]:
        raise_if_cancelled(cancel_event)
        self.requests.append(request)
        if not self._replies:
            reply: ModelResponse | Exception = ModelResponse.from_parts(
                text="No scripted replies remain.",
                metadata=ModelMetadata(
                    provider_id=self.profile.provider_id,
                    model_id=self.profile.model_id,
                ),
            )
        else:
            reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        tool_index = 0
        for item in reply.output:
            raise_if_cancelled(cancel_event)
            if isinstance(item, ReasoningOutput):
                yield ReasoningDelta(item.text)
            elif isinstance(item, TextOutput):
                yield TextDelta(item.text)
            elif isinstance(item, ToolCallOutput):
                yield ToolCallDelta(
                    index=tool_index,
                    call_id_delta=item.call.call_id,
                    name_delta=item.call.name,
                    arguments_delta=json.dumps(item.call.arguments, ensure_ascii=False),
                )
                tool_index += 1
        raise_if_cancelled(cancel_event)
        yield ModelCompleted(reply)
