from __future__ import annotations

import argparse
import sys
from pathlib import Path
from queue import Empty

from .agent_loop import AgentEvent, AgentLoop
from .control import ApprovalDecision
from .permissions import PermissionPolicy
from .providers.deepseek import DeepSeekProvider
from .runtime import (
    AgentRuntime,
    InterruptTurn,
    ResolveApproval,
    RuntimeEvent,
    StartTurn,
)
from .session import SessionState, SessionStore
from .terminal import (
    LineEditor,
    SlashCommand,
    StreamRenderer,
    format_context_status,
    render_context_usage,
)
from .tools import (
    ApplyPatchTool,
    ExecCommandTool,
    GitDiffTool,
    ListFilesTool,
    ReadFileTool,
    SearchTool,
    ToolRegistry,
)
from .trace import TraceRecorder
from .trajectory import make_trajectory, read_jsonl_trace, write_trajectory_json
from .workspace import Workspace


SLASH_COMMANDS = (
    SlashCommand("/new", "Start a new session"),
    SlashCommand("/sessions", "List saved sessions"),
    SlashCommand("/summary", "Show current session state"),
    SlashCommand("/context", "Show detailed context usage"),
    SlashCommand("/retry", "Resume the pending model step"),
    SlashCommand("/abort", "Discard the pending turn"),
    SlashCommand("/trace", "Show the raw trace path"),
    SlashCommand("/tools", "List available tools"),
    SlashCommand("/help", "Show slash commands"),
    SlashCommand("/exit", "Save and exit"),
)


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
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream model text and tool progress. Enabled by default.",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Display streamed reasoning_content before answer content.",
    )
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=None,
        help="Override the provider context window used by the context manager.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum model output tokens reserved from the context window.",
    )


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
    runtime = AgentRuntime(loop, state, interactive_approvals=False)
    renderer = StreamRenderer(answer_prefix="", show_thinking=args.show_thinking)
    try:
        runtime.send(StartTurn(user_input=prompt))
        terminal_event = _consume_runtime_events(args, runtime, renderer)
        if terminal_event.kind != "turn_completed" or terminal_event.result is None:
            raise RuntimeError(terminal_event.text or terminal_event.kind)
        result = terminal_event.result
    finally:
        runtime.close()
        _save_session_artifacts(store, state)
    if args.stream:
        renderer.finish(result.answer)
    else:
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
    runtime = AgentRuntime(loop, state, interactive_approvals=True)
    print(f"session: {state.session_id}")

    history_path = store.root.parent / "history"
    toolbar = []

    def current_toolbar():
        return toolbar

    try:
        with LineEditor(
            history_path,
            commands=SLASH_COMMANDS,
            status_provider=current_toolbar,
        ) as editor:
            while True:
                snapshot = loop.inspect_context(state)
                toolbar = format_context_status(
                    snapshot,
                    pending=state.pending_turn is not None,
                )
                try:
                    user_input = editor.read("you> ").strip()
                except KeyboardInterrupt:
                    print("^C")
                    continue
                except EOFError:
                    print()
                    _save_session_artifacts(store, state)
                    return 0

                if not user_input:
                    continue
                if user_input == "/exit":
                    _save_session_artifacts(store, state)
                    return 0
                if user_input == "/new":
                    runtime.close()
                    state = store.create(state.workspace)
                    workspace = Workspace(state.workspace)
                    loop, registry = _build_runtime(args, workspace, state.trace_path)
                    runtime = AgentRuntime(loop, state, interactive_approvals=True)
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
                    pending = state.pending_turn.turn_id if state.pending_turn else "none"
                    print(
                        f"session={state.session_id} workspace={state.workspace} "
                        f"messages={len(state.messages)} summarized="
                        f"{state.summarized_message_count} pending={pending}"
                    )
                    continue
                if user_input == "/context":
                    snapshot = loop.inspect_context(state)
                    print(
                        render_context_usage(
                            snapshot,
                            last_provider_input=state.last_input_tokens,
                            color=sys.stdout.isatty(),
                        )
                    )
                    continue
                if user_input == "/retry":
                    if state.pending_turn is None:
                        print("no pending turn to retry")
                        continue
                    _run_chat_turn(args, store, state, runtime, editor, retry=True)
                    continue
                if user_input == "/abort":
                    if state.pending_turn is None:
                        print("no pending turn to abort")
                        continue
                    pending = loop.abort_turn(state)
                    _save_session_artifacts(store, state)
                    warning = (
                        " Filesystem changes from completed tools were not reverted."
                        if pending.tool_calls_executed
                        else ""
                    )
                    print(f"aborted {pending.turn_id}.{warning}")
                    continue
                if user_input == "/trace":
                    print(state.trace_path)
                    continue
                if user_input == "/tools":
                    print("\n".join(registry.names()))
                    continue
                if user_input == "/help":
                    print("Commands")
                    for command in SLASH_COMMANDS:
                        print(f"  {command.name:<12}{command.description}")
                    print("\nType / to open the menu; Tab accepts or runs a command.")
                    continue
                if user_input.startswith("/"):
                    print(f"unknown command: {user_input}")
                    continue

                if state.pending_turn is not None:
                    print("pending turn exists; use /retry or /abort before sending a new message")
                    continue
                _run_chat_turn(args, store, state, runtime, editor, user_input=user_input)
    finally:
        runtime.close()


def _run_chat_turn(
    args: argparse.Namespace,
    store: SessionStore,
    state: SessionState,
    runtime: AgentRuntime,
    editor: LineEditor,
    *,
    user_input: str | None = None,
    retry: bool = False,
) -> None:
    renderer = StreamRenderer(show_thinking=args.show_thinking)
    try:
        runtime.send(StartTurn(user_input=user_input or "", retry=retry))
        terminal_event = _consume_runtime_events(args, runtime, renderer, editor)
    finally:
        _save_session_artifacts(store, state)

    if terminal_event.kind == "turn_completed" and terminal_event.result is not None:
        if args.stream:
            renderer.finish(terminal_event.result.answer)
        else:
            print(f"assistant> {terminal_event.result.answer}")
        return
    renderer.finish("")
    if terminal_event.kind == "turn_interrupted":
        print("generation interrupted; use /retry to continue or /abort to discard the turn")
    else:
        if not args.stream:
            print(f"error: {terminal_event.text}")
        print("use /retry to continue or /abort to discard the pending turn")


def _consume_runtime_events(
    args: argparse.Namespace,
    runtime: AgentRuntime,
    renderer: StreamRenderer,
    editor: LineEditor | None = None,
) -> RuntimeEvent:
    interrupt_requested = False
    while True:
        try:
            event = runtime.next_event(timeout=0.1)
        except Empty:
            continue
        except KeyboardInterrupt:
            if not interrupt_requested:
                runtime.send(InterruptTurn())
                print("interrupt requested", file=sys.stderr)
                interrupt_requested = True
            continue

        if args.stream:
            renderer.handle(AgentEvent(kind=event.kind, text=event.text, data=event.data))
        if event.kind == "approval_requested":
            if args.stream:
                renderer.finish("")
            if editor is None:
                raise RuntimeError("non-interactive runtime requested approval")
            decision = _prompt_for_approval(editor, event)
            if decision is None:
                runtime.send(InterruptTurn())
                interrupt_requested = True
            else:
                runtime.send(
                    ResolveApproval(
                        request_id=event.data["request_id"],
                        decision=decision,
                    )
                )
        if event.kind in {"turn_completed", "turn_failed", "turn_interrupted"}:
            if args.stream and event.kind != "turn_completed":
                renderer.finish("")
            runtime.wait()
            return event


def _prompt_for_approval(
    editor: LineEditor,
    event: RuntimeEvent,
) -> ApprovalDecision | None:
    details = event.data.get("details", {})
    print(f"approval> {event.data.get('description', event.data.get('action', 'tool action'))}")
    if details.get("argv"):
        print(f"  argv: {details['argv']}")
        print(f"  cwd: {details.get('cwd', '.')}")
    if details.get("patch_preview"):
        print(details["patch_preview"])
        if details.get("preview_truncated"):
            print("...[approval preview truncated]")

    choices = {
        "y": ApprovalDecision.ALLOW_ONCE,
        "once": ApprovalDecision.ALLOW_ONCE,
        "a": ApprovalDecision.ALLOW_SESSION,
        "session": ApprovalDecision.ALLOW_SESSION,
        "n": ApprovalDecision.DENY,
        "deny": ApprovalDecision.DENY,
    }
    while True:
        try:
            value = editor.read(
                "approve [y] once / [a] session / [n] deny> "
            ).strip().lower()
        except KeyboardInterrupt:
            print("^C")
            return None
        except EOFError:
            return ApprovalDecision.DENY
        decision = choices.get(value)
        if decision is not None:
            return decision
        print("enter y, a, or n")


def _build_runtime(
    args: argparse.Namespace,
    workspace: Workspace,
    trace_path: str | Path,
) -> tuple[AgentLoop, ToolRegistry]:
    registry = build_tool_registry()
    provider = DeepSeekProvider(
        model=args.model,
        base_url=args.base_url,
        thinking=args.thinking,
        context_window_tokens=args.context_window_tokens,
        max_output_tokens=args.max_output_tokens,
    )
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
            GitDiffTool(),
            ListFilesTool(),
            ReadFileTool(),
            SearchTool(),
            ExecCommandTool(),
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
