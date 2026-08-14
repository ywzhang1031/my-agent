from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..messages import AssistantMessage, Message, ToolCall, ToolResultMessage, UserMessage
from ..provider import ProviderError, ProviderEvent, ProviderResponse, TokenUsage
from ..tools import ToolSpec


RETRYABLE_STATUS_CODES = {429, 500, 503}


class DeepSeekProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        thinking: str | None = None,
        context_window_tokens: int | None = None,
        max_output_tokens: int | None = None,
        max_retries: int = 2,
        retry_base_seconds: float = 0.75,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.base_url = (
            base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        ).rstrip("/")
        self.thinking = thinking or os.environ.get("DEEPSEEK_THINKING", "enabled")
        self.context_window_tokens = context_window_tokens or int(
            os.environ.get("DEEPSEEK_CONTEXT_WINDOW_TOKENS", "1000000")
        )
        self.max_output_tokens = max_output_tokens or int(
            os.environ.get("DEEPSEEK_MAX_OUTPUT_TOKENS", "64000")
        )
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        if self.context_window_tokens <= 0 or self.max_output_tokens <= 0:
            raise ValueError("DeepSeek token limits must be positive")
        if self.max_output_tokens >= self.context_window_tokens:
            raise ValueError("max output tokens must be smaller than the context window")

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
    ) -> Iterator[ProviderEvent]:
        if not self.api_key:
            raise ProviderError("DEEPSEEK_API_KEY is required for the DeepSeek provider")

        payload = self.build_payload(messages, tools, system_prompt)
        for attempt in range(1, self.max_retries + 2):
            accumulator = _StreamAccumulator()
            emitted_delta = False
            try:
                for chunk in self._stream_once(payload):
                    for event in accumulator.consume(chunk):
                        emitted_delta = True
                        yield event
                response = accumulator.finish()
                yield ProviderEvent(kind="completed", response=response)
                return
            except ProviderError as exc:
                can_retry = (
                    exc.retryable
                    and not emitted_delta
                    and attempt <= self.max_retries
                )
                if not can_retry:
                    raise
                delay = self.retry_base_seconds * (2 ** (attempt - 1))
                yield ProviderEvent(
                    kind="retry",
                    text=str(exc),
                    data={
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                        "status_code": exc.status_code,
                    },
                )
                if delay > 0:
                    time.sleep(delay)

        raise ProviderError("DeepSeek stream exhausted retries")

    def _stream_once(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
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
                        raise ProviderError(
                            f"DeepSeek stream returned invalid JSON: {exc}"
                        ) from exc
                    if not isinstance(chunk, dict):
                        raise ProviderError("DeepSeek stream chunk is not a JSON object")
                    yield chunk
                if not saw_done:
                    raise ProviderError(
                        "DeepSeek stream closed before [DONE]",
                        retryable=True,
                    )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(
                f"DeepSeek API error {exc.code}: {body}",
                retryable=exc.code in RETRYABLE_STATUS_CODES,
                status_code=exc.code,
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(
                f"DeepSeek connection error: {exc}",
                retryable=True,
            ) from exc

    def build_payload(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
    ) -> dict[str, Any]:
        api_messages: list[dict[str, Any]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(self._convert_messages(messages))

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "thinking": {"type": self.thinking},
            "max_tokens": self.max_output_tokens,
        }
        if tools:
            payload["tools"] = [self._convert_tool(tool) for tool in tools]
            payload["tool_choice"] = "auto"
        return payload

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
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


@dataclass
class _ToolCallParts:
    call_id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _StreamAccumulator:
    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    tool_calls: dict[int, _ToolCallParts] = field(default_factory=dict)
    finish_reason: str = "unknown"
    usage: TokenUsage = field(default_factory=TokenUsage)
    chunks: int = 0
    choices_seen: int = 0

    def consume(self, chunk: dict[str, Any]) -> list[ProviderEvent]:
        self.chunks += 1
        self._consume_usage(chunk.get("usage"))
        choices = chunk.get("choices") or []
        if not choices:
            return []

        self.choices_seen += 1
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            self.finish_reason = finish_reason
        delta = choice.get("delta") or {}
        events: list[ProviderEvent] = []

        reasoning = delta.get("reasoning_content") or ""
        if reasoning:
            self.reasoning_parts.append(reasoning)
            events.append(ProviderEvent(kind="reasoning_delta", text=reasoning))

        content = delta.get("content") or ""
        if content:
            self.content_parts.append(content)
            events.append(ProviderEvent(kind="content_delta", text=content))

        for raw_call in delta.get("tool_calls") or []:
            index = int(raw_call.get("index") or 0)
            parts = self.tool_calls.setdefault(index, _ToolCallParts())
            if raw_call.get("id"):
                parts.call_id += raw_call["id"]
            function = raw_call.get("function") or {}
            if function.get("name"):
                parts.name += function["name"]
            if function.get("arguments"):
                parts.arguments += function["arguments"]
        return events

    def finish(self) -> ProviderResponse:
        if self.choices_seen == 0 or self.finish_reason == "unknown":
            raise ProviderError(
                "DeepSeek stream ended without a completed choice",
                retryable=True,
            )
        if self.finish_reason == "insufficient_system_resource":
            raise ProviderError(
                "DeepSeek stopped because inference resources were insufficient",
                retryable=True,
            )
        calls: list[ToolCall] = []
        for index in sorted(self.tool_calls):
            parts = self.tool_calls[index]
            try:
                arguments = json.loads(parts.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw_arguments": parts.arguments}
            calls.append(
                ToolCall(
                    call_id=parts.call_id,
                    name=parts.name,
                    arguments=arguments,
                )
            )
        return ProviderResponse(
            content="".join(self.content_parts),
            reasoning_content="".join(self.reasoning_parts),
            tool_calls=calls,
            finish_reason=self.finish_reason,
            usage=self.usage,
            raw_response={"stream": True, "chunks": self.chunks},
        )

    def _consume_usage(self, raw_usage: Any) -> None:
        if not isinstance(raw_usage, dict):
            return
        self.usage = TokenUsage(
            input_tokens=int(raw_usage.get("prompt_tokens") or 0),
            output_tokens=int(raw_usage.get("completion_tokens") or 0),
            cache_read_tokens=int(raw_usage.get("prompt_cache_hit_tokens") or 0),
        )
