from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .messages import AssistantMessage, Message, ToolResultMessage, UserMessage
from .session import SessionState
from .tools import ToolSpec


class ContextWindowError(RuntimeError):
    pass


@dataclass(frozen=True)
class ContextBreakdown:
    system_prompt_tokens: int
    tool_definition_tokens: int
    summary_tokens: int
    conversation_tokens: int
    protocol_tokens: int

    @property
    def total_tokens(self) -> int:
        return (
            self.system_prompt_tokens
            + self.tool_definition_tokens
            + self.summary_tokens
            + self.conversation_tokens
            + self.protocol_tokens
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "system_prompt_tokens": self.system_prompt_tokens,
            "tool_definition_tokens": self.tool_definition_tokens,
            "summary_tokens": self.summary_tokens,
            "conversation_tokens": self.conversation_tokens,
            "protocol_tokens": self.protocol_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ContextSnapshot:
    messages: list[Message]
    system_prompt: str
    estimated_tokens: int
    context_window_tokens: int
    input_budget: int
    reserve_output_tokens: int
    active_messages: int
    summarized_messages: int
    breakdown: ContextBreakdown
    compacted_messages: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_tokens": self.estimated_tokens,
            "context_window_tokens": self.context_window_tokens,
            "input_budget": self.input_budget,
            "reserve_output_tokens": self.reserve_output_tokens,
            "active_messages": self.active_messages,
            "summarized_messages": self.summarized_messages,
            "compacted_messages": self.compacted_messages,
            "breakdown": self.breakdown.to_dict(),
        }


class ContextManager:
    def __init__(
        self,
        *,
        context_window_tokens: int,
        reserve_output_tokens: int,
        compact_threshold: float = 0.85,
        compact_target: float = 0.7,
        summary_tokens: int = 4_000,
    ) -> None:
        if context_window_tokens <= 0 or reserve_output_tokens <= 0:
            raise ValueError("context token limits must be positive")
        if reserve_output_tokens >= context_window_tokens:
            raise ValueError("reserved output must be smaller than the context window")
        if not 0 < compact_target < compact_threshold < 1:
            raise ValueError("context ratios must satisfy 0 < target < threshold < 1")
        if summary_tokens <= 0:
            raise ValueError("summary token budget must be positive")
        self.context_window_tokens = context_window_tokens
        self.reserve_output_tokens = reserve_output_tokens
        self.compact_threshold = compact_threshold
        self.compact_target = compact_target
        self.summary_tokens = summary_tokens

    @property
    def input_budget(self) -> int:
        return self.context_window_tokens - self.reserve_output_tokens

    def inspect(
        self,
        *,
        state: SessionState,
        tools: list[ToolSpec],
        system_prompt: str,
    ) -> ContextSnapshot:
        return self._snapshot(state, tools, system_prompt, compacted_messages=0)

    def prepare(
        self,
        *,
        state: SessionState,
        tools: list[ToolSpec],
        system_prompt: str,
    ) -> ContextSnapshot:
        snapshot = self.inspect(state=state, tools=tools, system_prompt=system_prompt)
        threshold = int(self.input_budget * self.compact_threshold)
        if snapshot.estimated_tokens <= threshold:
            return snapshot

        start = state.summarized_message_count
        candidates = [
            index
            for index in range(start + 1, len(state.messages))
            if isinstance(state.messages[index], UserMessage)
        ]
        if not candidates:
            self._ensure_fits(snapshot)
            return snapshot

        original_start = start
        summary = state.context_summary
        target = int(self.input_budget * self.compact_target)
        for cutoff in candidates:
            summary = _merge_summary(
                summary,
                state.messages[start:cutoff],
                max_tokens=self.summary_tokens,
            )
            start = cutoff
            state.context_summary = summary
            state.summarized_message_count = cutoff
            snapshot = self._snapshot(
                state,
                tools,
                system_prompt,
                compacted_messages=cutoff - original_start,
            )
            if snapshot.estimated_tokens <= target:
                break

        self._ensure_fits(snapshot)
        return snapshot

    def _snapshot(
        self,
        state: SessionState,
        tools: list[ToolSpec],
        system_prompt: str,
        compacted_messages: int,
    ) -> ContextSnapshot:
        start = state.summarized_message_count
        if start < 0 or start > len(state.messages):
            raise ContextWindowError("session summarized_message_count is invalid")
        active_messages = list(state.messages[start:])
        effective_prompt = _with_summary(system_prompt, state.context_summary)
        breakdown = estimate_request_breakdown(
            system_prompt=system_prompt,
            conversation_summary=state.context_summary,
            messages=active_messages,
            tools=tools,
        )
        return ContextSnapshot(
            messages=active_messages,
            system_prompt=effective_prompt,
            estimated_tokens=breakdown.total_tokens,
            context_window_tokens=self.context_window_tokens,
            input_budget=self.input_budget,
            reserve_output_tokens=self.reserve_output_tokens,
            active_messages=len(active_messages),
            summarized_messages=start,
            breakdown=breakdown,
            compacted_messages=compacted_messages,
        )

    def _ensure_fits(self, snapshot: ContextSnapshot) -> None:
        if snapshot.estimated_tokens > self.input_budget:
            raise ContextWindowError(
                "active turn exceeds the context input budget: "
                f"estimated={snapshot.estimated_tokens} budget={self.input_budget}"
            )


def estimate_request_tokens(
    system_prompt: str,
    messages: list[Message],
    tools: list[ToolSpec],
) -> int:
    return estimate_request_breakdown(
        system_prompt=system_prompt,
        conversation_summary="",
        messages=messages,
        tools=tools,
    ).total_tokens


def estimate_request_breakdown(
    *,
    system_prompt: str,
    conversation_summary: str,
    messages: list[Message],
    tools: list[ToolSpec],
) -> ContextBreakdown:
    tool_payload = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tools
    ]
    effective_prompt = _with_summary(system_prompt, conversation_summary)
    effective_system_prompt = (
        system_prompt.rstrip() if conversation_summary else system_prompt
    )
    system_prompt_tokens = estimate_text_tokens(effective_system_prompt)
    summary_tokens = 0
    if conversation_summary:
        effective_prompt_tokens = estimate_text_tokens(effective_prompt)
        summary_tokens = max(0, effective_prompt_tokens - system_prompt_tokens)
    return ContextBreakdown(
        system_prompt_tokens=system_prompt_tokens,
        tool_definition_tokens=(
            estimate_text_tokens(json.dumps(tool_payload, ensure_ascii=False))
            if tool_payload
            else 0
        ),
        summary_tokens=summary_tokens,
        conversation_tokens=sum(
            estimate_text_tokens(_message_payload(message)) for message in messages
        ),
        protocol_tokens=16,
    )


def estimate_text_tokens(text: str) -> int:
    ascii_chars = sum(1 for character in text if ord(character) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, math.ceil(ascii_chars / 4) + non_ascii_chars)


def _message_payload(message: Message) -> str:
    if isinstance(message, UserMessage):
        payload = {"role": "user", "content": message.content}
    elif isinstance(message, AssistantMessage):
        payload = {
            "role": "assistant",
            "content": message.content,
            "reasoning_content": message.reasoning_content,
            "tool_calls": [call.to_dict() for call in message.tool_calls],
        }
    else:
        payload = {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "content": message.content,
        }
    return json.dumps(payload, ensure_ascii=False)


def _with_summary(system_prompt: str, summary: str) -> str:
    if not summary:
        return system_prompt
    return (
        f"{system_prompt.rstrip()}\n\n"
        "<conversation_summary>\n"
        f"{summary}\n"
        "</conversation_summary>\n"
        "Treat this as a lossy summary of older completed turns. Recent messages are authoritative."
    )


def _merge_summary(previous: str, messages: list[Message], max_tokens: int) -> str:
    lines = previous.splitlines() if previous else []
    lines.extend(_summary_line(message) for message in messages)
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        tokens = estimate_text_tokens(line)
        if kept and used + tokens > max_tokens:
            break
        if not kept and tokens > max_tokens:
            line = _clip(line, max_tokens * 4)
            tokens = estimate_text_tokens(line)
        kept.append(line)
        used += tokens
    kept.reverse()
    if len(kept) < len(lines):
        kept.insert(0, "[earlier compacted context omitted]")
    return "\n".join(kept)


def _summary_line(message: Message) -> str:
    if isinstance(message, UserMessage):
        return f"user: {_clip(message.content, 600)}"
    if isinstance(message, AssistantMessage):
        tools = ", ".join(call.name for call in message.tool_calls)
        suffix = f"; tool_calls={tools}" if tools else ""
        return f"assistant: {_clip(message.content, 800)}{suffix}"
    try:
        result = json.loads(message.content)
    except json.JSONDecodeError:
        result = None
    if isinstance(result, dict):
        status = result.get("metadata", {}).get("status") or (
            "success" if result.get("ok") else "failed"
        )
        detail = result.get("stderr") or result.get("stdout") or ""
        return f"tool {message.tool_name}: {status}; {_clip(str(detail), 500)}"
    return f"tool {message.tool_name}: {_clip(message.content, 500)}"


def _clip(value: str, max_chars: int) -> str:
    text = " ".join(value.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"
