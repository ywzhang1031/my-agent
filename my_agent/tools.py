from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .messages import ToolCall
from .patches import MAX_PATCH_CHARS, apply_patch
from .permissions import PermissionPolicy
from .workspace import Workspace, WorkspaceError


MAX_OUTPUT_CHARS = 12000
MAX_UNTRACKED_FILES = 200


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
    permissions: PermissionPolicy
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


class ApplyPatchTool(ToolBase):
    name = "apply_patch"
    description = "Apply one exact, workspace-relative Add, Update, or Delete File patch."
    parameters = {
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": (
                    "One-file patch delimited by *** Begin Patch and *** End Patch. "
                    "Update targets must be UTF-8 text with LF line endings."
                ),
                "maxLength": MAX_PATCH_CHARS,
            }
        },
        "required": ["patch"],
        "additionalProperties": False,
    }
    required = ["patch"]
    strict = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        patch = arguments.get("patch")
        if not isinstance(patch, str):
            return ToolResult(
                ok=False,
                stderr="patch must be a string",
                metadata={"applied": False},
            )
        try:
            applied = apply_patch(patch, ctx.workspace, ctx.permissions)
        except (WorkspaceError, OSError, ValueError) as exc:
            return ToolResult(
                ok=False,
                stderr=str(exc),
                metadata={"applied": False},
            )
        return ToolResult(
            ok=True,
            stdout=f"{applied.operation} {applied.path}",
            path=str(applied.absolute_path),
            metadata={
                "applied": True,
                "operation": applied.operation,
                "changed_files": [applied.path],
            },
        )


class GitDiffTool(ToolBase):
    name = "git_diff"
    description = "Show tracked changes from HEAD plus workspace-relative untracked file names."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Optional workspace-relative file or directory path.",
            }
        },
        "additionalProperties": False,
    }
    strict = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            resolved, pathspec = _git_pathspec(arguments.get("path", "."), ctx.workspace)
            repository = _run_git(ctx, ["rev-parse", "--is-inside-work-tree"])
            if repository.returncode != 0 or repository.stdout.strip() != "true":
                return ToolResult(
                    ok=False,
                    stderr="workspace is not a Git repository",
                    exit_code=repository.returncode,
                )
            head = _run_git(ctx, ["rev-parse", "--verify", "HEAD"])
            if head.returncode != 0:
                return ToolResult(
                    ok=False,
                    stderr="Git repository has no HEAD commit",
                    exit_code=head.returncode,
                )

            diff = _run_git(
                ctx,
                [
                    "diff",
                    "--no-color",
                    "--no-ext-diff",
                    "--no-textconv",
                    "HEAD",
                    "--",
                    pathspec,
                ],
            )
            if diff.returncode != 0:
                return ToolResult(
                    ok=False,
                    stderr=diff.stderr.strip() or "git diff failed",
                    exit_code=diff.returncode,
                )
            untracked_result = _run_git(
                ctx,
                ["ls-files", "--others", "--exclude-standard", "-z", "--", pathspec],
            )
            if untracked_result.returncode != 0:
                return ToolResult(
                    ok=False,
                    stderr=untracked_result.stderr.strip() or "git ls-files failed",
                    exit_code=untracked_result.returncode,
                )
        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=False,
                stderr=f"git command timed out after {ctx.timeout_seconds}s",
            )
        except (OSError, ValueError, WorkspaceError) as exc:
            return ToolResult(ok=False, stderr=str(exc))

        all_untracked = sorted(
            path
            for path in untracked_result.stdout.split("\0")
            if path and ".my-agent" not in {part.casefold() for part in PurePosixPath(path).parts}
        )
        untracked = all_untracked[:MAX_UNTRACKED_FILES]
        untracked_truncated = len(all_untracked) > len(untracked)
        sections: list[str] = []
        if diff.stdout.strip():
            sections.append(diff.stdout.rstrip())
        if untracked:
            listing = "[untracked files]\n" + "\n".join(untracked)
            if untracked_truncated:
                listing += "\n...[untracked files truncated]"
            sections.append(listing)
        output, output_truncated = _truncate("\n\n".join(sections) or "No changes.")
        return ToolResult(
            ok=True,
            stdout=output,
            exit_code=0,
            truncated=output_truncated or untracked_truncated,
            path=str(resolved),
            metadata={
                "base": "HEAD",
                "path": pathspec,
                "untracked_files": untracked,
                "untracked_truncated": untracked_truncated,
            },
        )


def _git_pathspec(path: Any, workspace: Workspace) -> tuple[Path, str]:
    if not isinstance(path, str) or not path:
        raise ValueError("git diff path must be a non-empty workspace-relative string")
    candidate = Path(path)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ValueError("git diff path must be workspace-relative")
    resolved = workspace.resolve(path)
    relative = resolved.relative_to(workspace.root)
    return resolved, "." if relative == Path(".") else relative.as_posix()


def _run_git(
    ctx: ToolContext,
    arguments: list[str],
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    return subprocess.run(
        ["git", "-c", "core.pager=cat", "-c", "color.ui=false", *arguments],
        cwd=ctx.workspace.root,
        env=environment,
        text=True,
        errors="replace",
        capture_output=True,
        timeout=ctx.timeout_seconds,
        check=False,
    )


def _truncate(value: bytes | str, max_chars: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n...[truncated]", True
