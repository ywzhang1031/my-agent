from __future__ import annotations

import json
from dataclasses import dataclass

from .messages import AssistantMessage, Message, ToolResultMessage, UserMessage
from .permissions import PermissionPolicy
from .provider import Provider, ProviderResponse
from .session import SessionState
from .tools import ToolContext, ToolRegistry
from .trace import TraceRecorder
from .workspace import Workspace


DEFAULT_INSTRUCTIONS = """You are a coding agent operating inside one workspace.
You may inspect files, search the repository, edit files with apply_patch, and run test commands.
Each apply_patch call must change exactly one workspace-relative file.
Do not modify .git, .my-agent, or paths outside the workspace.
After editing, inspect the actual changes with git_diff and run the narrowest relevant tests.
Use read_file when you need the contents of an untracked file reported by git_diff.
Return a concise result only after checking the changes and test evidence.
"""


@dataclass
class TurnResult:
    answer: str
    messages: list[Message]
    steps: int
    stopped_by: str


class AgentLoop:
    def __init__(
        self,
        workspace: Workspace,
        provider: Provider,
        tools: ToolRegistry,
        permissions: PermissionPolicy,
        trace: TraceRecorder,
        max_steps: int = 12,
        instructions: str = DEFAULT_INSTRUCTIONS,
    ) -> None:
        self.workspace = workspace
        self.provider = provider
        self.tools = tools
        self.permissions = permissions
        self.trace = trace
        self.max_steps = max_steps
        self.instructions = instructions

    def run_turn(self, state: SessionState, user_input: str) -> TurnResult:
        if state.workspace != str(self.workspace.root):
            raise ValueError(
                f"session workspace {state.workspace!r} does not match loop workspace "
                f"{str(self.workspace.root)!r}"
            )
        turn_number = sum(isinstance(message, UserMessage) for message in state.messages) + 1
        messages = state.messages
        messages.append(UserMessage(content=user_input))
        seen_calls: dict[str, int] = {}
        ctx = ToolContext(workspace=self.workspace, permissions=self.permissions)
        event_context = {
            "session_id": state.session_id,
            "turn_id": f"{state.session_id}:{turn_number}",
        }

        self.trace.write(
            "turn_started",
            {
                **event_context,
                "task": user_input,
                "workspace": str(self.workspace.root),
                "max_steps": self.max_steps,
                "messages_before_request": len(messages),
            },
        )

        for step in range(1, self.max_steps + 1):
            self.trace.write(
                "model_request",
                {
                    **event_context,
                    "step": step,
                    "messages": len(messages),
                    "tools": self.tools.names(),
                },
            )
            reply: ProviderResponse = self.provider.send(
                messages=messages,
                tools=self.tools.specs(),
                system_prompt=self.instructions,
            )
            self.trace.write(
                "model_response",
                {
                    **event_context,
                    "step": step,
                    "content": reply.content,
                    "tool_calls": [call.to_dict() for call in reply.tool_calls],
                },
            )

            if not reply.tool_calls:
                answer = reply.content.strip() or "(model returned no final content)"
                messages.append(
                    AssistantMessage(
                        content=answer,
                        reasoning_content=reply.reasoning_content,
                    )
                )
                self.trace.write(
                    "final_answer",
                    {**event_context, "step": step, "answer": answer},
                )
                return TurnResult(
                    answer=answer,
                    messages=list(messages),
                    steps=step,
                    stopped_by="final_answer",
                )

            messages.append(
                AssistantMessage(
                    content=reply.content,
                    reasoning_content=reply.reasoning_content,
                    tool_calls=reply.tool_calls,
                )
            )

            for call in reply.tool_calls:
                call_key = json.dumps(
                    {"name": call.name, "arguments": call.arguments},
                    sort_keys=True,
                    ensure_ascii=False,
                )
                seen_calls[call_key] = seen_calls.get(call_key, 0) + 1
                if seen_calls[call_key] > 3:
                    result = {
                        "ok": False,
                        "stderr": "repeated identical tool call stopped",
                    }
                    self.trace.write(
                        "tool_result",
                        {
                            **event_context,
                            "step": step,
                            "call": call.to_dict(),
                            "result": result,
                        },
                    )
                    messages.append(
                        ToolResultMessage(
                            tool_call_id=call.call_id,
                            tool_name=call.name,
                            content=json.dumps(result, ensure_ascii=False),
                        )
                    )
                    continue

                self.trace.write(
                    "tool_call",
                    {**event_context, "step": step, "call": call.to_dict()},
                )
                tool_result = self.tools.run(call, ctx)
                result_payload = tool_result.to_dict()
                self.trace.write(
                    "tool_result",
                    {
                        **event_context,
                        "step": step,
                        "call": call.to_dict(),
                        "result": result_payload,
                    },
                )
                messages.append(
                    ToolResultMessage(
                        tool_call_id=call.call_id,
                        tool_name=call.name,
                        content=json.dumps(result_payload, ensure_ascii=False),
                    )
                )

        answer = f"Stopped after reaching max_steps={self.max_steps} without a final answer."
        messages.append(AssistantMessage(content=answer))
        self.trace.write(
            "final_answer",
            {**event_context, "step": self.max_steps, "answer": answer},
        )
        return TurnResult(
            answer=answer,
            messages=list(messages),
            steps=self.max_steps,
            stopped_by="max_steps",
        )
