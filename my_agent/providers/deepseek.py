from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from ..messages import AssistantMessage, Message, ToolCall, ToolResultMessage, UserMessage
from ..provider import ProviderResponse, TokenUsage
from ..tools import ToolSpec


class DeepSeekProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        thinking: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.thinking = thinking or os.environ.get("DEEPSEEK_THINKING", "enabled")

    def send(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
    ) -> ProviderResponse:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for the DeepSeek provider")

        payload = self.build_payload(messages, tools, system_prompt)
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
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API error {exc.code}: {body}") from exc

        return self.parse_response(data)

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
            "stream": False,
            "thinking": {"type": self.thinking},
        }
        if tools:
            payload["tools"] = [self._convert_tool(tool) for tool in tools]
            payload["tool_choice"] = "auto"
        return payload

    def parse_response(self, data: dict[str, Any]) -> ProviderResponse:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek API returned no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        tool_calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw_arguments": function.get("arguments", "")}
            tool_calls.append(
                ToolCall(
                    call_id=raw_call.get("id", ""),
                    name=function.get("name", ""),
                    arguments=arguments,
                )
            )

        usage = data.get("usage") or {}
        return ProviderResponse(
            content=message.get("content") or "",
            reasoning_content=message.get("reasoning_content") or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "unknown"),
            usage=TokenUsage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
            ),
            raw_response=data,
        )

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
