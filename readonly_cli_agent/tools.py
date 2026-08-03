from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any, Protocol

from .messages import ToolCall
from .permissions import ReadOnlyPermissionPolicy
from .workspace import Workspace, WorkspaceError


MAX_OUTPUT_CHARS = 12000


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str] = field(default_factory=list)
    strict: bool = False


@dataclass
class ToolResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    truncated: bool = False
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "truncated": self.truncated,
            "path": self.path,
            "metadata": self.metadata,
        }


@dataclass
class ToolContext:
    workspace: Workspace
    permissions: ReadOnlyPermissionPolicy
    timeout_seconds: int = 60


class BaseTool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str]

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ...


class ToolBase:
    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str] = []
    strict: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            required=self.required,
            strict=self.strict,
        )


class ToolRegistry:
    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(ok=False, stderr=f"unknown tool: {call.name}")
        return tool.run(call.arguments, ctx)


class ListFilesTool(ToolBase):
    name = "list_files"
    description = "List files under a workspace-relative directory."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative directory path."},
            "max_files": {"type": "integer", "description": "Maximum number of files to return."},
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    required = ["path"]

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            files, truncated = ctx.workspace.list_files(
                arguments.get("path", "."),
                max_files=int(arguments.get("max_files", 200)),
            )
            return ToolResult(
                ok=True,
                stdout="\n".join(files),
                truncated=truncated,
                metadata={"count": len(files)},
            )
        except (WorkspaceError, OSError, ValueError) as exc:
            return ToolResult(ok=False, stderr=str(exc))


class ReadFileTool(ToolBase):
    name = "read_file"
    description = "Read a UTF-8 text file inside the workspace with truncation."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "max_chars": {"type": "integer", "description": "Maximum characters to read."},
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    required = ["path"]

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            text, truncated, resolved = ctx.workspace.read_file(
                arguments.get("path", ""),
                max_chars=int(arguments.get("max_chars", MAX_OUTPUT_CHARS)),
            )
            return ToolResult(ok=True, stdout=text, truncated=truncated, path=str(resolved))
        except (WorkspaceError, OSError, UnicodeDecodeError, ValueError) as exc:
            return ToolResult(ok=False, stderr=str(exc))


class SearchTool(ToolBase):
    name = "search"
    description = "Search workspace text files for a literal string or regex pattern."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern."},
            "path": {"type": "string", "description": "Workspace-relative directory path."},
            "literal": {"type": "boolean", "description": "Treat pattern as literal text."},
            "max_results": {"type": "integer", "description": "Maximum matching lines."},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    required = ["pattern"]

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            matches, truncated = ctx.workspace.search(
                pattern=arguments.get("pattern", ""),
                path=arguments.get("path", "."),
                literal=bool(arguments.get("literal", False)),
                max_results=int(arguments.get("max_results", 100)),
            )
            return ToolResult(
                ok=True,
                stdout=json.dumps(matches, ensure_ascii=False, indent=2),
                truncated=truncated,
                metadata={"count": len(matches)},
            )
        except (WorkspaceError, OSError, ValueError) as exc:
            return ToolResult(ok=False, stderr=str(exc))


class RunTestsTool(ToolBase):
    name = "run_tests"
    description = "Run an allowlisted test command without shell expansion."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Test command to run."},
            "timeout_seconds": {"type": "integer", "description": "Timeout in seconds."},
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    required = ["command"]

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = arguments.get("command", "")
        allowed, reason, argv = ctx.permissions.allow_test_command(command)
        if not allowed:
            return ToolResult(ok=False, stderr=reason)
        timeout = int(arguments.get("timeout_seconds", ctx.timeout_seconds))
        try:
            completed = subprocess.run(
                argv,
                cwd=ctx.workspace.root,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _truncate(exc.stdout or "")
            stderr = _truncate(exc.stderr or f"command timed out after {timeout}s")
            return ToolResult(ok=False, stdout=stdout[0], stderr=stderr[0], truncated=stdout[1] or stderr[1])
        except OSError as exc:
            return ToolResult(ok=False, stderr=str(exc))

        stdout, stdout_truncated = _truncate(completed.stdout)
        stderr, stderr_truncated = _truncate(completed.stderr)
        return ToolResult(
            ok=completed.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            exit_code=completed.returncode,
            truncated=stdout_truncated or stderr_truncated,
            metadata={"command": argv},
        )


def _truncate(value: bytes | str, max_chars: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n...[truncated]", True
