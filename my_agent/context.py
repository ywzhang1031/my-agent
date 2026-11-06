from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any

from .messages import AssistantMessage, Message, ToolResultMessage, UserMessage
from .session import SessionState
from .tools import ToolSpec, skill_load_result


class ContextWindowError(RuntimeError):
    pass


@dataclass(frozen=True)
class ContextBreakdown:
    system_prompt_tokens: int
    tool_definition_tokens: int
    summary_tokens: int
    conversation_tokens: int
    protocol_tokens: int
    project_rules_tokens: int = 0
    skill_catalog_tokens: int = 0
    skill_body_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.system_prompt_tokens
            + self.tool_definition_tokens
            + self.summary_tokens
            + self.conversation_tokens
            + self.protocol_tokens
            + self.project_rules_tokens
            + self.skill_catalog_tokens
            + self.skill_body_tokens
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "system_prompt_tokens": self.system_prompt_tokens,
            "tool_definition_tokens": self.tool_definition_tokens,
            "summary_tokens": self.summary_tokens,
            "conversation_tokens": self.conversation_tokens,
            "protocol_tokens": self.protocol_tokens,
            "project_rules_tokens": self.project_rules_tokens,
            "skill_catalog_tokens": self.skill_catalog_tokens,
            "skill_body_tokens": self.skill_body_tokens,
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
    project_rules: dict[str, Any] | None = None
    skills: dict[str, Any] | None = None

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
            "project_rules": self.project_rules,
            "skills": self.skills,
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
        if snapshot.project_rules and snapshot.project_rules["status"] == "error":
            raise ContextWindowError(
                f"{snapshot.project_rules['error']}. Fix the file, then abandon this turn and start a new one."
            )
        if snapshot.skills and snapshot.skills.get("error"):
            raise ContextWindowError(f"{snapshot.skills['error']}. Abandon this turn and start a new one after fixing the skill.")
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
        active_messages = _project_messages(state, start)
        rules = state.current_project_rules()
        rules_prompt = rules.prompt() if rules is not None else ""
        rule_messages = [UserMessage(content=rules_prompt)] if rules_prompt else []
        skills = state.current_skills()
        catalog_prompt = skills.catalog_prompt() if skills else ""
        body_prompt = skills.manual_prompt() if skills else ""
        catalog_messages = [UserMessage(content=catalog_prompt)] if catalog_prompt else []
        body_messages = [UserMessage(content=body_prompt)] if body_prompt else []
        effective_prompt = _with_summary(system_prompt, state.context_summary)
        breakdown = estimate_request_breakdown(
            system_prompt=system_prompt,
            conversation_summary=state.context_summary,
            messages=active_messages,
            tools=tools,
            project_rules_messages=rule_messages,
            skill_catalog_messages=catalog_messages,
            skill_body_messages=body_messages,
        )
        return ContextSnapshot(
            messages=rule_messages + catalog_messages + body_messages + active_messages,
            system_prompt=effective_prompt,
            estimated_tokens=breakdown.total_tokens,
            context_window_tokens=self.context_window_tokens,
            input_budget=self.input_budget,
            reserve_output_tokens=self.reserve_output_tokens,
            active_messages=len(active_messages),
            summarized_messages=start,
            breakdown=breakdown,
            compacted_messages=compacted_messages,
            project_rules=rules.manifest() if rules is not None else None,
            skills=skills.manifest() if skills is not None else None,
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


def _skill_result_payload(message: Message) -> dict[str, Any] | None:
    if not isinstance(message, ToolResultMessage) or message.tool_name != "load_skill":
        return None
    try:
        payload = json.loads(message.content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return None
    metadata = payload.get("metadata")
    if (not isinstance(metadata, dict) or metadata.get("status") != "skill_loaded"
            or not isinstance(metadata.get("skill"), dict)):
        return None
    return payload


def _project_messages(state: SessionState, start: int) -> list[Message]:
    current_start = next((i for i in range(len(state.messages) - 1, -1, -1)
                          if isinstance(state.messages[i], UserMessage)), -1)
    skills = state.current_skills()
    loaded = {item.skill.name: item for item in skills.loaded} if skills else {}
    seen: set[str] = set()
    projected: list[Message] = []
    for index, message in enumerate(state.messages[start:], start):
        if isinstance(message, UserMessage):
            projected.append(UserMessage(content=message.content))
            continue
        payload = _skill_result_payload(message)
        if payload is not None:
            manifest = payload["metadata"]["skill"]
            skill = loaded.get(manifest.get("name"))
            if index >= current_start and skill is not None and manifest == skill.manifest():
                include_body = skill.reason == "model" and skill.skill.name not in seen
                payload = skill_load_result(skill, include_body=include_body).to_dict()
                seen.add(skill.skill.name)
            else:
                # Keep call/result pairing and provenance; never rewrite stored history.
                payload = {**payload,
                           "stdout": "Skill guidance is not active for this turn. Load the skill again if relevant.",
                           "metadata": {**payload["metadata"], "body_included": False}}
            message = replace(message, content=json.dumps(payload, ensure_ascii=False))
        projected.append(message)
    return projected


def estimate_request_breakdown(
    *,
    system_prompt: str,
    conversation_summary: str,
    messages: list[Message],
    tools: list[ToolSpec],
    project_rules_messages: list[UserMessage] | None = None,
    skill_catalog_messages: list[UserMessage] | None = None,
    skill_body_messages: list[UserMessage] | None = None,
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
    conversation_tokens = 0
    skill_tool_tokens = 0
    for message in messages:
        tokens = estimate_text_tokens(_message_payload(message))
        payload = _skill_result_payload(message)
        if payload is not None and payload["metadata"].get("body_included") is True:
            skill_tool_tokens += tokens
        else:
            conversation_tokens += tokens
    return ContextBreakdown(
        system_prompt_tokens=system_prompt_tokens,
        tool_definition_tokens=(
            estimate_text_tokens(json.dumps(tool_payload, ensure_ascii=False))
            if tool_payload
            else 0
        ),
        summary_tokens=summary_tokens,
        conversation_tokens=conversation_tokens,
        protocol_tokens=16,
        project_rules_tokens=sum(
            estimate_text_tokens(_message_payload(message))
            for message in project_rules_messages or []
        ),
        skill_catalog_tokens=sum(estimate_text_tokens(_message_payload(message)) for message in skill_catalog_messages or []),
        skill_body_tokens=skill_tool_tokens + sum(estimate_text_tokens(_message_payload(message)) for message in skill_body_messages or []),
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
            "provider_data": message.provider_data,
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
    skill_result = _skill_result_payload(message)
    if skill_result is not None:
        source = skill_result["metadata"]["skill"]
        return f"tool load_skill: skill_loaded; name={source.get('name')}; source={source.get('path')}; guidance expired"
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
