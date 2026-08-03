from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent_loop import AgentLoop
from .permissions import PermissionPolicy
from .providers.deepseek import DeepSeekProvider
from .session import SessionState, SessionStore
from .tools import (
    ApplyPatchTool,
    ListFilesTool,
    ReadFileTool,
    RunTestsTool,
    SearchTool,
    ToolRegistry,
)
from .trace import TraceRecorder
from .trajectory import make_trajectory, read_jsonl_trace, write_trajectory_json
from .workspace import Workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal coding agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Run one persisted conversation turn.")
    _add_runtime_options(ask)
    ask.add_argument("prompt", nargs="*", help="Task for the agent. Reads stdin if omitted.")

    chat = subparsers.add_parser("chat", help="Start a new interactive session.")
    _add_runtime_options(chat)

    resume = subparsers.add_parser("resume", help="Resume an existing interactive session.")
    resume.add_argument("session_id", help="Session identifier printed by ask or chat.")
    _add_runtime_options(resume)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "ask":
            return _run_ask(args)
        if args.command == "chat":
            return _run_new_chat(args)
        if args.command == "resume":
            return _run_resume(args)
        raise ValueError(f"unknown command: {args.command}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    _add_provider_options(parser)
    parser.add_argument("--workspace", default=".", help="Workspace root the agent may inspect.")
    parser.add_argument(
        "--sessions-dir",
        default=None,
        help="Session storage root. Defaults to <workspace>/.my-agent/sessions.",
    )
    parser.add_argument("--max-steps", type=int, default=12, help="Maximum model/tool loop steps per turn.")


def _add_provider_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default="deepseek", choices=["deepseek"], help="LLM provider.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults to DEEPSEEK_MODEL or deepseek-v4-flash.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Provider base URL. Defaults to DEEPSEEK_BASE_URL or DeepSeek API.",
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default=None,
        help="DeepSeek thinking mode. Defaults to DEEPSEEK_THINKING or enabled.",
    )


def _run_ask(args: argparse.Namespace) -> int:
    prompt = _read_prompt(args.prompt)
    workspace = Workspace(args.workspace)
    store = _session_store(args, workspace)
    state = store.create(workspace.root)
    print(f"session: {state.session_id}", file=sys.stderr)
    loop, _ = _build_runtime(args, workspace, state.trace_path)
    try:
        result = loop.run_turn(state, prompt)
    finally:
        _save_session_artifacts(store, state)
    print(result.answer)
    return 0


def _run_new_chat(args: argparse.Namespace) -> int:
    workspace = Workspace(args.workspace)
    store = _session_store(args, workspace)
    state = store.create(workspace.root)
    return _chat(args, store, state)


def _run_resume(args: argparse.Namespace) -> int:
    requested_workspace = Workspace(args.workspace)
    store = _session_store(args, requested_workspace)
    state = store.load(args.session_id)
    return _chat(args, store, state)


def _chat(args: argparse.Namespace, store: SessionStore, state: SessionState) -> int:
    workspace = Workspace(state.workspace)
    loop, registry = _build_runtime(args, workspace, state.trace_path)
    print(f"session: {state.session_id}")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _save_session_artifacts(store, state)
            return 0

        if not user_input:
            continue
        if user_input == "/exit":
            _save_session_artifacts(store, state)
            return 0
        if user_input == "/new":
            state = store.create(state.workspace)
            workspace = Workspace(state.workspace)
            loop, registry = _build_runtime(args, workspace, state.trace_path)
            print(f"session: {state.session_id}")
            continue
        if user_input == "/sessions":
            for summary in store.list_sessions():
                print(
                    f"{summary.session_id}  messages={summary.message_count}  "
                    f"updated={summary.updated_at}"
                )
            continue
        if user_input == "/summary":
            print(
                f"session={state.session_id} workspace={state.workspace} "
                f"messages={len(state.messages)}"
            )
            continue
        if user_input == "/trace":
            print(state.trace_path)
            continue
        if user_input == "/tools":
            print("\n".join(registry.names()))
            continue
        if user_input.startswith("/"):
            print(f"unknown command: {user_input}")
            continue

        try:
            result = loop.run_turn(state, user_input)
        finally:
            _save_session_artifacts(store, state)
        print(f"assistant> {result.answer}")


def _build_runtime(
    args: argparse.Namespace,
    workspace: Workspace,
    trace_path: str | Path,
) -> tuple[AgentLoop, ToolRegistry]:
    registry = build_tool_registry()
    provider = DeepSeekProvider(model=args.model, base_url=args.base_url, thinking=args.thinking)
    loop = AgentLoop(
        workspace=workspace,
        provider=provider,
        tools=registry,
        permissions=PermissionPolicy(),
        trace=TraceRecorder(trace_path),
        max_steps=args.max_steps,
    )
    return loop, registry


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ApplyPatchTool(),
            ListFilesTool(),
            ReadFileTool(),
            SearchTool(),
            RunTestsTool(),
        ]
    )


def _session_store(args: argparse.Namespace, workspace: Workspace) -> SessionStore:
    root = Path(args.sessions_dir) if args.sessions_dir else workspace.root / ".my-agent" / "sessions"
    return SessionStore(root)


def _save_session_artifacts(store: SessionStore, state: SessionState) -> None:
    store.save(state)
    if not state.trace_path.exists():
        return
    events = read_jsonl_trace(state.trace_path)
    trajectory = make_trajectory(events, source_path=state.trace_path)
    write_trajectory_json(trajectory, state.trajectory_path)


def _read_prompt(parts: list[str]) -> str:
    prompt = " ".join(parts).strip() or sys.stdin.read().strip()
    if not prompt:
        raise ValueError("prompt is required")
    return prompt
