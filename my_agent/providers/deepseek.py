from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..control import raise_if_cancelled
from ..messages import AssistantMessage, Message, ToolCall, ToolResultMessage, UserMessage
from ..provider import (
    ModelCompleted,
    ModelError,
    ModelErrorKind,
    ModelErrorPhase,
    ModelEvent,
    ModelMetadata,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ProviderCapabilities,
    ReasoningDelta,
    ReasoningOutput,
    StopReason,
    TextDelta,
    TextOutput,
    ToolCallDelta,
    ToolCallOutput,
)
from ..tools import ToolSpec


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
CONTEXT_ERROR_MARKERS = (
    "context length",
    "context window",
    "too many tokens",
    "maximum prompt length",
    "prompt is too long",
)


class DeepSeekAdapter:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        thinking: str | None = None,
        context_window_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        model_id = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.base_url = (
            base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        ).rstrip("/")
        self.thinking = thinking or os.environ.get("DEEPSEEK_THINKING", "enabled")
        if self.thinking not in {"enabled", "disabled"}:
            raise ValueError("DeepSeek thinking must be 'enabled' or 'disabled'")
        self.profile = ModelProfile(
            provider_id="deepseek",
            model_id=model_id,
            context_window_tokens=context_window_tokens
            or int(os.environ.get("DEEPSEEK_CONTEXT_WINDOW_TOKENS", "1000000")),
            max_output_tokens=max_output_tokens
            or int(os.environ.get("DEEPSEEK_MAX_OUTPUT_TOKENS", "64000")),
            capabilities=ProviderCapabilities(reasoning=True),
        )

    def stream_once(
        self,
        request: ModelRequest,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[ModelEvent]:
        raise_if_cancelled(cancel_event)
        if not self.api_key:
            raise ModelError(
                "DEEPSEEK_API_KEY is required for the DeepSeek provider",
                kind=ModelErrorKind.CONFIGURATION,
                phase=ModelErrorPhase.CONFIGURATION,
                provider_id=self.profile.provider_id,
                model_id=self.profile.model_id,
            )

        payload = self.build_payload(request)
        accumulator = _StreamAccumulator(
            provider_id=self.profile.provider_id,
            requested_model_id=self.profile.model_id,
        )
        for chunk in self._stream_once(payload, cancel_event):
            for event in accumulator.consume(chunk):
                raise_if_cancelled(cancel_event)
                yield event
        yield ModelCompleted(accumulator.finish())

    def _stream_once(
        self,
        payload: dict[str, Any],
        cancel_event: threading.Event | None,
    ) -> Iterator[dict[str, Any]]:
        raise_if_cancelled(cancel_event)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                saw_done = False
                for raw_line in response:
                    raise_if_cancelled(cancel_event)
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise self._error(
                            f"DeepSeek stream returned invalid JSON: {exc}",
                            kind=ModelErrorKind.PROTOCOL,
                            phase=ModelErrorPhase.DECODING,
                        ) from exc
                    if not isinstance(chunk, dict):
                        raise self._error(
                            "DeepSeek stream chunk is not a JSON object",
                            kind=ModelErrorKind.PROTOCOL,
                            phase=ModelErrorPhase.DECODING,
                        )
                    yield chunk
                if not saw_done:
                    raise self._error(
                        "DeepSeek stream closed before [DONE]",
                        kind=ModelErrorKind.STREAM,
                        phase=ModelErrorPhase.STREAM,
                        retryable=True,
                    )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message, provider_code = _parse_error_body(body)
            request_id = _header(exc.headers, "x-request-id")
            raise self._error(
                f"DeepSeek API error {exc.code}: {message}",
                kind=_http_error_kind(exc.code, message),
                phase=ModelErrorPhase.REQUEST,
                retryable=exc.code in RETRYABLE_STATUS_CODES,
                status_code=exc.code,
                provider_code=provider_code,
                request_id=request_id,
                retry_after_seconds=_retry_after_seconds(exc.headers),
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise self._error(
                f"DeepSeek connection error: {exc}",
                kind=ModelErrorKind.CONNECTION,
                phase=ModelErrorPhase.REQUEST,
                retryable=True,
            ) from exc

    def build_payload(self, request: ModelRequest) -> dict[str, Any]:
        api_messages: list[dict[str, Any]] = []
        if request.system_prompt:
            api_messages.append({"role": "system", "content": request.system_prompt})
        api_messages.extend(self._convert_messages(request.messages))

        payload: dict[str, Any] = {
            "model": self.profile.model_id,
            "messages": api_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "thinking": {"type": self.thinking},
            "max_tokens": self.profile.max_output_tokens,
        }
        if request.tools:
            payload["tools"] = [self._convert_tool(tool) for tool in request.tools]
            payload["tool_choice"] = "auto"
        return payload

    def _convert_messages(self, messages: tuple[Message, ...]) -> list[dict[str, Any]]:
        api_messages: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, UserMessage):
                api_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AssistantMessage):
                item: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content,
                }
                if message.reasoning_content:
                    item["reasoning_content"] = message.reasoning_content
                if message.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in message.tool_calls
                    ]
                api_messages.append(item)
            elif isinstance(message, ToolResultMessage):
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content,
                    }
                )
        return api_messages

    def _convert_tool(self, tool: ToolSpec) -> dict[str, Any]:
        function: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if tool.strict:
            function["strict"] = True
        return {"type": "function", "function": function}

    def _error(self, message: str, **kwargs: Any) -> ModelError:
        return ModelError(
            message,
            provider_id=self.profile.provider_id,
            model_id=self.profile.model_id,
            **kwargs,
        )


@dataclass
class _ToolCallParts:
    call_id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _StreamAccumulator:
    provider_id: str
    requested_model_id: str
    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    tool_calls: dict[int, _ToolCallParts] = field(default_factory=dict)
    stop_reason: str = "unknown"
    usage: ModelUsage | None = None
    response_id: str | None = None
    response_model_id: str | None = None
    chunks: int = 0
    choices_seen: int = 0

    def consume(self, chunk: dict[str, Any]) -> list[ModelEvent]:
        self.chunks += 1
        self.response_id = _non_empty_string(chunk.get("id")) or self.response_id
        self.response_model_id = _non_empty_string(chunk.get("model")) or self.response_model_id
        self._consume_usage(chunk.get("usage"))
        choices = chunk.get("choices") or []
        if not choices:
            return []
        if not isinstance(choices, list) or not isinstance(choices[0], dict):
            raise self._protocol_error("DeepSeek choices must be a list of objects")

        self.choices_seen += 1
        choice = choices[0]
        stop_reason = choice.get("finish_reason")
        if isinstance(stop_reason, str) and stop_reason:
            self.stop_reason = stop_reason
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise self._protocol_error("DeepSeek delta must be a JSON object")
        events: list[ModelEvent] = []

        reasoning = delta.get("reasoning_content") or ""
        if reasoning:
            if not isinstance(reasoning, str):
                raise self._protocol_error("DeepSeek reasoning_content must be a string")
            self.reasoning_parts.append(reasoning)
            events.append(ReasoningDelta(reasoning))

        content = delta.get("content") or ""
        if content:
            if not isinstance(content, str):
                raise self._protocol_error("DeepSeek content must be a string")
            self.content_parts.append(content)
            events.append(TextDelta(content))

        raw_calls = delta.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise self._protocol_error("DeepSeek tool_calls must be a list")
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                raise self._protocol_error("DeepSeek tool call delta must be an object")
            try:
                index = int(raw_call.get("index") or 0)
            except (TypeError, ValueError) as exc:
                raise self._protocol_error("DeepSeek tool call index must be an integer") from exc
            parts = self.tool_calls.setdefault(index, _ToolCallParts())
            call_id_delta = _non_empty_string(raw_call.get("id")) or ""
            parts.call_id += call_id_delta
            function = raw_call.get("function") or {}
            if not isinstance(function, dict):
                raise self._protocol_error("DeepSeek tool call function must be an object")
            name_delta = _non_empty_string(function.get("name")) or ""
            arguments_delta = _non_empty_string(function.get("arguments")) or ""
            parts.name += name_delta
            parts.arguments += arguments_delta
            events.append(
                ToolCallDelta(
                    index=index,
                    call_id_delta=call_id_delta,
                    name_delta=name_delta,
                    arguments_delta=arguments_delta,
                )
            )
        return events

    def finish(self) -> ModelResponse:
        if self.choices_seen == 0 or self.stop_reason == "unknown":
            raise ModelError(
                "DeepSeek stream ended without a completed choice",
                kind=ModelErrorKind.STREAM,
                phase=ModelErrorPhase.STREAM,
                retryable=True,
                provider_id=self.provider_id,
                model_id=self.requested_model_id,
            )
        if self.stop_reason == "insufficient_system_resource":
            raise ModelError(
                "DeepSeek stopped because inference resources were insufficient",
                kind=ModelErrorKind.SERVER,
                phase=ModelErrorPhase.STREAM,
                retryable=True,
                provider_id=self.provider_id,
                model_id=self.requested_model_id,
            )

        output: list[TextOutput | ReasoningOutput | ToolCallOutput] = []
        reasoning = "".join(self.reasoning_parts)
        if reasoning:
            output.append(ReasoningOutput(reasoning))
        content = "".join(self.content_parts)
        if content:
            output.append(TextOutput(content))
        for index in sorted(self.tool_calls):
            parts = self.tool_calls[index]
            if not parts.call_id or not parts.name:
                raise self._protocol_error(
                    f"DeepSeek tool call {index} is missing an id or function name"
                )
            try:
                arguments = json.loads(parts.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise self._protocol_error(
                    f"DeepSeek tool call {parts.name} returned invalid JSON arguments"
                ) from exc
            if not isinstance(arguments, dict):
                raise self._protocol_error(
                    f"DeepSeek tool call {parts.name} arguments must be a JSON object"
                )
            output.append(
                ToolCallOutput(
                    ToolCall(
                        call_id=parts.call_id,
                        name=parts.name,
                        arguments=arguments,
                    )
                )
            )
        return ModelResponse(
            output=tuple(output),
            stop_reason=_normalize_stop_reason(self.stop_reason),
            usage=self.usage,
            metadata=ModelMetadata(
                provider_id=self.provider_id,
                model_id=self.requested_model_id,
                response_model_id=self.response_model_id,
                response_id=self.response_id,
                native={
                    "stream": True,
                    "chunks": self.chunks,
                    "finish_reason": self.stop_reason,
                },
            ),
        )

    def _consume_usage(self, raw_usage: Any) -> None:
        if not isinstance(raw_usage, dict):
            return
        details = raw_usage.get("completion_tokens_details") or {}
        if not isinstance(details, dict):
            details = {}
        self.usage = ModelUsage(
            input_tokens=_optional_int(raw_usage.get("prompt_tokens")),
            output_tokens=_optional_int(raw_usage.get("completion_tokens")),
            total_tokens=_optional_int(raw_usage.get("total_tokens")),
            reasoning_tokens=_optional_int(details.get("reasoning_tokens")),
            cache_read_tokens=_optional_int(raw_usage.get("prompt_cache_hit_tokens")),
        )

    def _protocol_error(self, message: str) -> ModelError:
        return ModelError(
            message,
            kind=ModelErrorKind.PROTOCOL,
            phase=ModelErrorPhase.DECODING,
            provider_id=self.provider_id,
            model_id=self.requested_model_id,
        )


def _parse_error_body(body: str) -> tuple[str, str | None]:
    body = body.strip()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return (body[:4000] or "empty response body"), None
    if not isinstance(payload, dict):
        return str(payload)[:4000], None
    error = payload.get("error")
    if not isinstance(error, dict):
        return json.dumps(payload, ensure_ascii=False)[:4000], None
    message = str(error.get("message") or body or "unknown provider error")
    code = error.get("code")
    return message[:4000], str(code) if code is not None else None


def _normalize_stop_reason(value: str) -> StopReason:
    try:
        return StopReason(value)
    except ValueError:
        return StopReason.UNKNOWN


def _http_error_kind(status_code: int, message: str) -> ModelErrorKind:
    lowered = message.lower()
    if status_code in {400, 413} and any(marker in lowered for marker in CONTEXT_ERROR_MARKERS):
        return ModelErrorKind.CONTEXT_OVERFLOW
    if status_code in {401, 403}:
        return ModelErrorKind.AUTHENTICATION
    if status_code == 408:
        return ModelErrorKind.CONNECTION
    if status_code == 429:
        return ModelErrorKind.RATE_LIMIT
    if status_code == 409 or status_code >= 500:
        return ModelErrorKind.SERVER
    if 400 <= status_code < 500:
        return ModelErrorKind.INVALID_REQUEST
    return ModelErrorKind.UNKNOWN


def _retry_after_seconds(headers: Any) -> float | None:
    value = _header(headers, "retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    value = headers.get(name)
    return str(value) if value is not None else None


def _non_empty_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
