from __future__ import annotations

import json
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
    "maximum input tokens",
    "prompt is too long",
)
TERMINAL_EVENT_TYPES = {
    "response.completed",
    "response.failed",
    "response.incomplete",
}


class OpenAIAdapter:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        context_window_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key
        model_id = model or "gpt-5.6"
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.reasoning_effort = reasoning_effort or "medium"
        if self.reasoning_effort not in {
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("unsupported OpenAI reasoning effort")
        self.reasoning_summary = reasoning_summary or "auto"
        if self.reasoning_summary not in {"auto", "concise", "detailed"}:
            raise ValueError("unsupported OpenAI reasoning summary")
        self.profile = ModelProfile(
            provider_id="openai",
            model_id=model_id,
            context_window_tokens=context_window_tokens or 1_050_000,
            max_output_tokens=max_output_tokens or 128_000,
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
                "OpenAI API key is required (OPENAI_API_KEY or credentials.json)",
                kind=ModelErrorKind.CONFIGURATION,
                phase=ModelErrorPhase.CONFIGURATION,
                provider_id=self.profile.provider_id,
                model_id=self.profile.model_id,
            )

        accumulator = _ResponsesAccumulator(
            provider_id=self.profile.provider_id,
            requested_model_id=self.profile.model_id,
        )
        for event_payload in self._stream_once(self.build_payload(request), cancel_event):
            for event in accumulator.consume(event_payload):
                raise_if_cancelled(cancel_event)
                yield event
        yield ModelCompleted(accumulator.finish())

    def build_payload(self, request: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.profile.model_id,
            "input": self._convert_messages(request.messages),
            "stream": True,
            "store": False,
            "max_output_tokens": self.profile.max_output_tokens,
            "parallel_tool_calls": True,
        }
        if request.system_prompt:
            payload["instructions"] = request.system_prompt
        if request.tools:
            payload["tools"] = [self._convert_tool(tool) for tool in request.tools]
            payload["tool_choice"] = "auto"
        if self.reasoning_effort != "none":
            payload["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": self.reasoning_summary,
            }
            payload["include"] = ["reasoning.encrypted_content"]
        return payload

    def _convert_messages(self, messages: tuple[Message, ...]) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, UserMessage):
                input_items.append({"role": "user", "content": message.content})
                continue
            if isinstance(message, ToolResultMessage):
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content,
                    }
                )
                continue

            native = message.provider_data.get("openai", {})
            native_items = native.get("input_items") if isinstance(native, dict) else None
            if isinstance(native_items, list) and all(
                isinstance(item, dict) for item in native_items
            ):
                input_items.extend(native_items)
                continue
            if message.content:
                input_items.append({"role": "assistant", "content": message.content})
            input_items.extend(
                {
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                }
                for call in message.tool_calls
            )
        return input_items

    def _convert_tool(self, tool: ToolSpec) -> dict[str, Any]:
        converted: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if tool.strict:
            converted["strict"] = True
        return converted

    def _stream_once(
        self,
        payload: dict[str, Any],
        cancel_event: threading.Event | None,
    ) -> Iterator[dict[str, Any]]:
        raise_if_cancelled(cancel_event)
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                saw_terminal = False
                for raw_line in response:
                    raise_if_cancelled(cancel_event)
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise self._error(
                            f"OpenAI stream returned invalid JSON: {exc}",
                            kind=ModelErrorKind.PROTOCOL,
                            phase=ModelErrorPhase.DECODING,
                        ) from exc
                    if not isinstance(event, dict):
                        raise self._error(
                            "OpenAI stream event is not a JSON object",
                            kind=ModelErrorKind.PROTOCOL,
                            phase=ModelErrorPhase.DECODING,
                        )
                    yield event
                    if event.get("type") in TERMINAL_EVENT_TYPES:
                        saw_terminal = True
                        break
                if not saw_terminal:
                    raise self._error(
                        "OpenAI stream closed before a terminal response event",
                        kind=ModelErrorKind.STREAM,
                        phase=ModelErrorPhase.STREAM,
                        retryable=True,
                    )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message, provider_code = _parse_error_body(body)
            raise self._error(
                f"OpenAI API error {exc.code}: {message}",
                kind=_http_error_kind(exc.code, message),
                phase=ModelErrorPhase.REQUEST,
                retryable=exc.code in RETRYABLE_STATUS_CODES,
                status_code=exc.code,
                provider_code=provider_code,
                request_id=_header(exc.headers, "x-request-id"),
                retry_after_seconds=_retry_after_seconds(exc.headers),
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise self._error(
                f"OpenAI connection error: {exc}",
                kind=ModelErrorKind.CONNECTION,
                phase=ModelErrorPhase.REQUEST,
                retryable=True,
            ) from exc

    def _error(self, message: str, **kwargs: Any) -> ModelError:
        return ModelError(
            message,
            provider_id=self.profile.provider_id,
            model_id=self.profile.model_id,
            **kwargs,
        )


@dataclass
class _FunctionCallParts:
    call_id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _ResponsesAccumulator:
    provider_id: str
    requested_model_id: str
    streamed_text: list[str] = field(default_factory=list)
    streamed_reasoning: list[str] = field(default_factory=list)
    streamed_calls: dict[int, _FunctionCallParts] = field(default_factory=dict)
    response: dict[str, Any] | None = None
    terminal_type: str | None = None
    events_seen: int = 0

    def consume(self, event: dict[str, Any]) -> list[ModelEvent]:
        self.events_seen += 1
        event_type = event.get("type")
        if not isinstance(event_type, str):
            raise self._protocol_error("OpenAI event is missing a string type")
        if event_type == "error":
            error = event.get("error") if isinstance(event.get("error"), dict) else event
            message = str(error.get("message") or "OpenAI stream error")
            raise ModelError(
                message,
                kind=ModelErrorKind.STREAM,
                phase=ModelErrorPhase.STREAM,
                retryable=False,
                provider_id=self.provider_id,
                model_id=self.requested_model_id,
                provider_code=_optional_string(error.get("code")),
            )
        if event_type == "response.output_text.delta":
            delta = _required_event_string(event, "delta", event_type)
            self.streamed_text.append(delta)
            return [TextDelta(delta)]
        if event_type == "response.reasoning_summary_text.delta":
            delta = _required_event_string(event, "delta", event_type)
            self.streamed_reasoning.append(delta)
            return [ReasoningDelta(delta)]
        if event_type == "response.output_item.added":
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") != "function_call":
                return []
            index = _required_event_index(event, event_type)
            parts = self.streamed_calls.setdefault(index, _FunctionCallParts())
            call_id = _optional_string(item.get("call_id")) or ""
            name = _optional_string(item.get("name")) or ""
            arguments = _optional_string(item.get("arguments")) or ""
            parts.call_id = call_id
            parts.name = name
            parts.arguments = arguments
            return [
                ToolCallDelta(
                    index=index,
                    call_id_delta=call_id,
                    name_delta=name,
                    arguments_delta=arguments,
                )
            ]
        if event_type == "response.function_call_arguments.delta":
            index = _required_event_index(event, event_type)
            delta = _required_event_string(event, "delta", event_type)
            parts = self.streamed_calls.setdefault(index, _FunctionCallParts())
            parts.arguments += delta
            return [ToolCallDelta(index=index, arguments_delta=delta)]
        if event_type in TERMINAL_EVENT_TYPES:
            response = event.get("response")
            if not isinstance(response, dict):
                raise self._protocol_error(f"{event_type} is missing response")
            self.response = response
            self.terminal_type = event_type
            if event_type == "response.failed":
                error = response.get("error")
                error = error if isinstance(error, dict) else {}
                raise ModelError(
                    str(error.get("message") or "OpenAI response failed"),
                    kind=ModelErrorKind.SERVER,
                    phase=ModelErrorPhase.STREAM,
                    retryable=True,
                    provider_id=self.provider_id,
                    model_id=self.requested_model_id,
                    provider_code=_optional_string(error.get("code")),
                )
        return []

    def finish(self) -> ModelResponse:
        if self.response is None or self.terminal_type is None:
            raise ModelError(
                "OpenAI stream ended without a terminal response",
                kind=ModelErrorKind.STREAM,
                phase=ModelErrorPhase.STREAM,
                retryable=True,
                provider_id=self.provider_id,
                model_id=self.requested_model_id,
            )
        raw_output = self.response.get("output")
        if not isinstance(raw_output, list):
            raise self._protocol_error("OpenAI response output must be a list")
        output: list[TextOutput | ReasoningOutput | ToolCallOutput] = []
        tool_calls: list[ToolCall] = []
        for item in raw_output:
            if not isinstance(item, dict):
                raise self._protocol_error("OpenAI response output item must be an object")
            item_type = item.get("type")
            if item_type == "reasoning":
                reasoning = _reasoning_summary(item)
                if reasoning:
                    output.append(ReasoningOutput(reasoning))
            elif item_type == "message":
                text = _message_text(item)
                if text:
                    output.append(TextOutput(text))
            elif item_type == "function_call":
                call = _function_call(item, self._protocol_error)
                tool_calls.append(call)
                output.append(ToolCallOutput(call))

        usage = _usage(self.response.get("usage"))
        response_model_id = _optional_string(self.response.get("model"))
        response_id = _optional_string(self.response.get("id"))
        return ModelResponse(
            output=tuple(output),
            stop_reason=_stop_reason(self.terminal_type, self.response, tool_calls),
            usage=usage,
            metadata=ModelMetadata(
                provider_id=self.provider_id,
                model_id=self.requested_model_id,
                response_model_id=response_model_id,
                response_id=response_id,
                native={
                    "stream": True,
                    "events": self.events_seen,
                    "status": self.response.get("status"),
                },
            ),
            provider_data={"openai": {"input_items": raw_output}},
        )

    def _protocol_error(self, message: str) -> ModelError:
        return ModelError(
            message,
            kind=ModelErrorKind.PROTOCOL,
            phase=ModelErrorPhase.DECODING,
            provider_id=self.provider_id,
            model_id=self.requested_model_id,
        )


def _reasoning_summary(item: dict[str, Any]) -> str:
    summary = item.get("summary")
    if not isinstance(summary, list):
        return ""
    return "".join(
        str(part.get("text") or "")
        for part in summary
        if isinstance(part, dict) and part.get("type") == "summary_text"
    )


def _message_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "output_text"
    )


def _function_call(
    item: dict[str, Any],
    protocol_error: Any,
) -> ToolCall:
    call_id = _optional_string(item.get("call_id"))
    name = _optional_string(item.get("name"))
    arguments_text = _optional_string(item.get("arguments")) or "{}"
    if not call_id or not name:
        raise protocol_error("OpenAI function call is missing call_id or name")
    try:
        arguments = json.loads(arguments_text)
    except json.JSONDecodeError as exc:
        raise protocol_error(f"OpenAI function {name} returned invalid JSON arguments") from exc
    if not isinstance(arguments, dict):
        raise protocol_error(f"OpenAI function {name} arguments must be a JSON object")
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _usage(raw: Any) -> ModelUsage | None:
    if not isinstance(raw, dict):
        return None
    input_details = raw.get("input_tokens_details")
    output_details = raw.get("output_tokens_details")
    input_details = input_details if isinstance(input_details, dict) else {}
    output_details = output_details if isinstance(output_details, dict) else {}
    return ModelUsage(
        input_tokens=_optional_int(raw.get("input_tokens")),
        output_tokens=_optional_int(raw.get("output_tokens")),
        total_tokens=_optional_int(raw.get("total_tokens")),
        reasoning_tokens=_optional_int(output_details.get("reasoning_tokens")),
        cache_read_tokens=_optional_int(input_details.get("cached_tokens")),
    )


def _stop_reason(
    terminal_type: str,
    response: dict[str, Any],
    tool_calls: list[ToolCall],
) -> StopReason:
    if tool_calls:
        return StopReason.TOOL_CALLS
    if terminal_type == "response.incomplete":
        details = response.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        if reason == "max_output_tokens":
            return StopReason.LENGTH
        if reason == "content_filter":
            return StopReason.CONTENT_FILTER
        return StopReason.UNKNOWN
    return StopReason.STOP


def _required_event_string(event: dict[str, Any], field_name: str, event_type: str) -> str:
    value = event.get(field_name)
    if not isinstance(value, str):
        raise ModelError(
            f"OpenAI {event_type} field {field_name} must be a string",
            kind=ModelErrorKind.PROTOCOL,
            phase=ModelErrorPhase.DECODING,
            provider_id="openai",
        )
    return value


def _required_event_index(event: dict[str, Any], event_type: str) -> int:
    value = event.get("output_index")
    if isinstance(value, bool):
        value = None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ModelError(
            f"OpenAI {event_type} output_index must be an integer",
            kind=ModelErrorKind.PROTOCOL,
            phase=ModelErrorPhase.DECODING,
            provider_id="openai",
        ) from exc


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


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
