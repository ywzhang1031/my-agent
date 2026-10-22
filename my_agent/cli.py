from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from queue import Empty

from .agent_loop import AgentLoop
from .app_server import AppServer
from .config import RuntimeConfig, load_runtime_config, save_provider_credential
from .control import ApprovalDecision
from .event_store import EventStore
from .model_invoker import ModelInvoker
from .modes import BUILD_MODE, COLLABORATION_MODES, CollaborationMode, get_mode
from .permissions import PERMISSION_PROFILES, PermissionPolicy, get_permission_profile
from .providers.factory import (
    SUPPORTED_PROVIDER_IDS,
    ProviderConfig,
    create_provider_adapter,
)
from .protocol import (
    AbortTurnCommand,
    ApprovalRequestedEvent,
    ExecutePlanCommand,
    InterruptTurnCommand,
    ModeChangedEvent,
    ModelChangedEvent,
    PlanApprovedEvent,
    PlanRevisionRequestedEvent,
    PermissionProfileChangedEvent,
    ResolveApprovalCommand,
    RevisePlanCommand,
    StartTurnCommand,
    SwitchModeCommand,
    SwitchModelCommand,
    SwitchPermissionProfileCommand,
    TurnAbortedEvent,
    TurnCompletedEvent,
    TurnContinuationRequiredEvent,
    TurnFailedEvent,
    TurnInterruptedEvent,
)
from .runtime import AgentRuntime
from .sandbox import SANDBOX_MODES, SandboxPolicy
from .service import AgentService
from .session import SessionState, SessionStore
from .terminal import (
    LineEditor,
    SelectionOption,
    SlashCommand,
    StreamRenderer,
    format_context_status,
    render_context_usage,
)
from .transport import JsonlAppServer
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
from .web_server import LocalWebServer


SLASH_COMMANDS = (
    SlashCommand("/new", "Start a new session"),
    SlashCommand("/sessions", "List saved sessions"),
    SlashCommand("/summary", "Show current session state"),
    SlashCommand("/context", "Show detailed context usage"),
    SlashCommand("/model", "Choose the model"),
    SlashCommand("/mode", "Choose Build or Plan mode"),
    SlashCommand("/permissions", "Choose the permission profile"),
    SlashCommand("/approve", "Approve the proposed plan and execute it"),
    SlashCommand("/config", "Show provider configuration"),
    SlashCommand("/retry", "Resume the pending model step"),
    SlashCommand("/abort", "Discard the pending turn"),
    SlashCommand("/trace", "Show the raw trace path"),
    SlashCommand("/tools", "List available tools"),
    SlashCommand("/help", "Show slash commands"),
    SlashCommand("/exit", "Save and exit"),
)

PLAN_ACTIONS = (
    SelectionOption("Execute plan", "Approve, switch to Build, and execute now"),
    SelectionOption("Revise plan", "Stay in Plan mode and describe changes"),
)

CONTINUATION_ACTIONS = (
    SelectionOption("Continue", "Run another model/tool step batch"),
    SelectionOption("Stop turn", "Discard the pending turn and keep completed filesystem changes"),
)

TerminalEvent = (
    TurnCompletedEvent
    | TurnContinuationRequiredEvent
    | TurnFailedEvent
    | TurnInterruptedEvent
    | TurnAbortedEvent
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

    serve = subparsers.add_parser(
        "serve",
        help="Run a multi-session JSONL app server over stdin/stdout.",
    )
    _add_runtime_options(serve, terminal_ui=False)

    ui = subparsers.add_parser(
        "ui",
        help="Run the local multi-session web interface.",
    )
    _add_runtime_options(ui, terminal_ui=False)
    ui.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Loopback HTTP port. Use 0 to select an available port.",
    )
    ui.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the web interface in the default browser.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.runtime_config = load_runtime_config(args.config, args.credentials)
        if args.command == "ask":
            return _run_ask(args)
        if args.command == "chat":
            return _run_new_chat(args)
        if args.command == "resume":
            return _run_resume(args)
        if args.command == "serve":
            return _run_serve(args)
        if args.command == "ui":
            return _run_ui(args)
        raise ValueError(f"unknown command: {args.command}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _add_runtime_options(
    parser: argparse.ArgumentParser,
    *,
    terminal_ui: bool = True,
) -> None:
    _add_provider_options(parser)
    parser.add_argument("--workspace", default=".", help="Workspace root the agent may inspect.")
    parser.add_argument(
        "--sessions-dir",
        default=None,
        help="Session storage root. Defaults to <workspace>/.my-agent/sessions.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="Model/tool steps to run before asking whether to continue.",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.mode_id for mode in COLLABORATION_MODES],
        default=None,
        help="Collaboration mode. New sessions default to build.",
    )
    parser.add_argument(
        "--permission-profile",
        choices=[profile.profile_id for profile in PERMISSION_PROFILES],
        default=None,
        help="Permission profile. New sessions default to supervised.",
    )
    parser.add_argument(
        "--sandbox",
        choices=SANDBOX_MODES,
        default=None,
        help="Command sandbox. New sessions default to workspace-write.",
    )
    if terminal_ui:
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
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider ID. Uses config default when omitted.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider model ID. Uses the selected provider's default when omitted.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the selected provider's API base URL.",
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default=None,
        help="DeepSeek thinking mode. Uses provider config when omitted.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default=None,
        help="OpenAI reasoning effort. Uses provider config when omitted.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Provider config path. Defaults to ~/.config/my-agent/config.json.",
    )
    parser.add_argument(
        "--credentials",
        default=None,
        help="Credentials path. Defaults beside the provider config.",
    )


def _run_ask(args: argparse.Namespace) -> int:
    prompt = _read_prompt(args.prompt)
    workspace = Workspace(args.workspace)
    store = _session_store(args, workspace)
    state = store.create(workspace.root)
    _restore_session_mode(args, state)
    _restore_session_permission_profile(args, state)
    _restore_session_sandbox(args, state)
    print(f"session: {state.session_id}", file=sys.stderr)
    loop, _ = _build_runtime(args, workspace, state.trace_path)
    _sync_session_model(state, loop)
    _sync_session_mode(state, loop)
    _sync_session_permission_profile(state, loop)
    _sync_session_sandbox(state, loop)
    service = _build_agent_service(
        args,
        store,
        state,
        loop,
        interactive_approvals=False,
    )
    renderer = StreamRenderer(
        answer_prefix="",
        show_thinking=args.show_thinking,
        hide_plan_protocol=loop.mode.produces_plan,
    )
    try:
        service.dispatch(StartTurnCommand(session_id=state.session_id, user_input=prompt))
        terminal_event = _consume_service_events(args, service, state.session_id, renderer)
        if not isinstance(terminal_event, TurnCompletedEvent):
            raise RuntimeError(_terminal_event_message(terminal_event))
    finally:
        service.close()
        _save_session_artifacts(store, state)
    if args.stream:
        renderer.finish(terminal_event.answer)
    else:
        print(terminal_event.answer)
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


def _run_serve(args: argparse.Namespace) -> int:
    app_server = _build_app_server(args)
    JsonlAppServer(app_server).serve(sys.stdin, sys.stdout)
    return 0


def _run_ui(args: argparse.Namespace) -> int:
    app_server = _build_app_server(args)
    try:
        server = LocalWebServer(app_server, port=args.port)
    except Exception:
        app_server.close()
        raise
    print(f"My Agent UI: {server.url}", file=sys.stderr)
    if args.open_browser:
        import webbrowser

        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def _build_app_server(args: argparse.Namespace) -> AppServer:
    workspace = Workspace(args.workspace)
    store = _session_store(args, workspace)
    service = AgentService(
        tool_registry_factory=build_tool_registry,
        model_resolver=lambda model_ref: _build_model(args, model_ref=model_ref),
        save_session=lambda session: _save_session_artifacts(store, session),
        event_store=EventStore(store.root),
    )
    try:
        return AppServer(
            workspace=workspace.root,
            session_store=store,
            agent_service=service,
            runtime_factory=lambda state, model_ref, mode_id: _build_server_runtime(
                args,
                state,
                model_ref=model_ref,
                mode_id=mode_id,
            ),
            models=(
                model
                for model in args.runtime_config.models()
                if model.provider_id in SUPPORTED_PROVIDER_IDS
            ),
            modes=COLLABORATION_MODES,
            credentials_path=args.runtime_config.credentials_path,
            credential_setter=lambda provider_id, api_key: _store_provider_credential(
                args,
                provider_id,
                api_key,
            ),
        )
    except Exception:
        service.close()
        raise


def _store_provider_credential(
    args: argparse.Namespace,
    provider_id: str,
    api_key: str,
) -> dict[str, str]:
    provider = args.runtime_config.providers.get(provider_id)
    if provider is None:
        raise ValueError(f"unknown provider in config: {provider_id}")
    args.runtime_config = save_provider_credential(
        args.runtime_config,
        provider_id,
        api_key,
    )
    # Keep the running process aligned with the credential the user just selected.
    # This also supersedes a stale environment value until the process restarts.
    import os

    os.environ[provider.api_key_env] = api_key
    return {"credentials_path": str(args.runtime_config.credentials_path)}


def _chat(args: argparse.Namespace, store: SessionStore, state: SessionState) -> int:
    _restore_session_model(args, state)
    _restore_session_mode(args, state)
    _restore_session_permission_profile(args, state)
    _restore_session_sandbox(args, state)
    workspace = Workspace(state.workspace)
    loop, _ = _build_runtime(args, workspace, state.trace_path)
    _sync_session_model(state, loop)
    _sync_session_mode(state, loop)
    _sync_session_permission_profile(state, loop)
    _sync_session_sandbox(state, loop)
    _save_session_artifacts(store, state)
    service = _build_agent_service(
        args,
        store,
        state,
        loop,
        interactive_approvals=True,
    )
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
                view = service.session_view(state.session_id)
                snapshot = service.inspect_context(state.session_id)
                toolbar = format_context_status(
                    snapshot,
                    provider_id=view.provider_id,
                    model_id=view.model_id,
                    mode_id=view.mode_id,
                    permission_profile_id=view.permission_profile_id,
                    sandbox_mode=view.sandbox_mode,
                    pending=view.pending_turn_id is not None,
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
                    service.close()
                    state = store.create(state.workspace)
                    workspace = Workspace(state.workspace)
                    loop, _ = _build_runtime(args, workspace, state.trace_path)
                    service = _build_agent_service(
                        args,
                        store,
                        state,
                        loop,
                        interactive_approvals=True,
                    )
                    _sync_session_model(state, loop)
                    _sync_session_mode(state, loop)
                    _sync_session_permission_profile(state, loop)
                    _save_session_artifacts(store, state)
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
                    plan = (
                        f"{state.proposed_plan.status}:{state.proposed_plan.plan_id}"
                        if state.proposed_plan is not None
                        else "none"
                    )
                    print(
                        f"session={state.session_id} workspace={state.workspace} "
                        f"model={view.provider_id}/{view.model_id} "
                        f"mode={view.mode_id} "
                        f"permissions={view.permission_profile_id} "
                        f"sandbox={view.sandbox_mode} "
                        f"messages={len(state.messages)} summarized="
                        f"{state.summarized_message_count} pending={pending} plan={plan}"
                    )
                    continue
                if user_input == "/model":
                    if state.pending_turn is not None:
                        print("pending turn exists; use /retry or /abort before switching model")
                        continue
                    current_model_ref = f"{view.provider_id}/{view.model_id}"
                    entries = tuple(
                        model
                        for model in args.runtime_config.models()
                        if model.provider_id in SUPPORTED_PROVIDER_IDS
                    )
                    if not entries:
                        print(f"no selectable models for {view.provider_id}")
                        continue
                    if not editor.supports_interactive_selection:
                        _print_model_catalog(
                            args.runtime_config,
                            current_model_ref,
                        )
                        print("interactive model selection requires a TTY")
                        continue
                    requested_model = editor.select(
                        "model> ",
                        (
                            SelectionOption(entry.ref, entry.description)
                            for entry in entries
                        ),
                        current_value=(
                            current_model_ref
                        ),
                    )
                    if requested_model is None:
                        print("model selection cancelled")
                        continue
                    if requested_model == current_model_ref:
                        print(
                            f"already using {view.provider_id}/{view.model_id}"
                        )
                        continue
                    service.dispatch(
                        SwitchModelCommand(
                            session_id=state.session_id,
                            model_ref=requested_model,
                        )
                    )
                    changed = service.next_event(timeout=1)
                    if not isinstance(changed, ModelChangedEvent):
                        raise RuntimeError(
                            f"expected model.changed, got {changed.event_type.value}"
                        )
                    args.provider = changed.provider_id
                    args.model = changed.model_id
                    print(f"model: {changed.provider_id}/{changed.model_id}")
                    continue
                if user_input == "/mode":
                    if state.pending_turn is not None:
                        print("pending turn exists; use /retry or /abort before switching mode")
                        continue
                    if not editor.supports_interactive_selection:
                        _print_mode_catalog(view.mode_id)
                        print("interactive mode selection requires a TTY")
                        continue
                    requested_mode = editor.select(
                        "mode> ",
                        (
                            SelectionOption(mode.mode_id, mode.description)
                            for mode in COLLABORATION_MODES
                        ),
                        current_value=view.mode_id,
                    )
                    if requested_mode is None:
                        print("mode selection cancelled")
                        continue
                    if requested_mode == view.mode_id:
                        print(f"already using mode: {requested_mode}")
                        continue
                    service.dispatch(
                        SwitchModeCommand(
                            session_id=state.session_id,
                            mode_id=requested_mode,
                        )
                    )
                    changed = service.next_event(timeout=1)
                    if not isinstance(changed, ModeChangedEvent):
                        raise RuntimeError(
                            f"expected mode.changed, got {changed.event_type.value}"
                        )
                    args.mode = changed.mode_id
                    print(f"mode: {changed.mode_id}")
                    continue
                if user_input == "/permissions":
                    if state.pending_turn is not None:
                        print(
                            "pending turn exists; use /retry or /abort before switching permissions"
                        )
                        continue
                    if not editor.supports_interactive_selection:
                        _print_permission_catalog(view.permission_profile_id)
                        print("interactive permission selection requires a TTY")
                        continue
                    requested_profile = editor.select(
                        "permissions> ",
                        (
                            SelectionOption(profile.profile_id, profile.description)
                            for profile in PERMISSION_PROFILES
                        ),
                        current_value=view.permission_profile_id,
                    )
                    if requested_profile is None:
                        print("permission selection cancelled")
                        continue
                    if requested_profile == view.permission_profile_id:
                        print(f"already using permissions: {requested_profile}")
                        continue
                    service.dispatch(
                        SwitchPermissionProfileCommand(
                            session_id=state.session_id,
                            profile_id=requested_profile,
                        )
                    )
                    changed = service.next_event(timeout=1)
                    if not isinstance(changed, PermissionProfileChangedEvent):
                        raise RuntimeError(
                            f"expected permission.changed, got {changed.event_type.value}"
                        )
                    args.permission_profile = changed.profile_id
                    print(f"permissions: {changed.profile_id}")
                    continue
                if user_input == "/approve":
                    if state.pending_turn is not None:
                        print("pending turn exists; use /retry or /abort before approving a plan")
                        continue
                    try:
                        _approve_and_execute_plan(
                            args,
                            store,
                            state,
                            service,
                            editor,
                        )
                    except ValueError as exc:
                        print(str(exc))
                    continue
                if user_input == "/config":
                    _print_runtime_config(args.runtime_config)
                    continue
                if user_input == "/context":
                    snapshot = service.inspect_context(state.session_id)
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
                    terminal_event = _run_chat_turn(
                        args, store, state, service, editor, retry=True
                    )
                    terminal_event = _offer_continuation_action(
                        args, store, state, service, editor, terminal_event
                    )
                    _offer_plan_action(
                        args, store, state, service, editor, terminal_event
                    )
                    continue
                if user_input == "/abort":
                    if state.pending_turn is None:
                        print("no pending turn to abort")
                        continue
                    service.dispatch(AbortTurnCommand(session_id=state.session_id))
                    aborted = service.next_event(timeout=1)
                    if not isinstance(aborted, TurnAbortedEvent):
                        raise RuntimeError(
                            f"expected turn.aborted, got {aborted.event_type.value}"
                        )
                    warning = (
                        " Filesystem changes from completed tools were not reverted."
                        if aborted.tool_calls_executed
                        else ""
                    )
                    print(f"aborted {aborted.turn_id}.{warning}")
                    continue
                if user_input == "/trace":
                    print(state.trace_path)
                    continue
                if user_input == "/tools":
                    print("\n".join(service.tool_names(state.session_id)))
                    continue
                if user_input == "/help":
                    print("Commands")
                    for command in SLASH_COMMANDS:
                        print(f"  {command.name:<12}{command.description}")
                    print("\nType / to open the menu; use Up/Down and Enter to run.")
                    print("Tab remains available for completion and quick execution.")
                    print("/model uses Up/Down to choose, Enter to switch, and Esc to cancel.")
                    print("/mode uses the same selector.")
                    print("/permissions selects supervised, accept_edits, or trusted.")
                    print("Completed plans open Execute/Revise; /approve runs a saved proposal.")
                    continue
                if user_input.startswith("/"):
                    print(f"unknown command: {user_input}")
                    continue

                if state.pending_turn is not None:
                    print("pending turn exists; use /retry or /abort before sending a new message")
                    continue
                terminal_event = _run_chat_turn(
                    args, store, state, service, editor, user_input=user_input
                )
                terminal_event = _offer_continuation_action(
                    args, store, state, service, editor, terminal_event
                )
                _offer_plan_action(
                    args, store, state, service, editor, terminal_event
                )
    finally:
        service.close()


def _approve_and_execute_plan(
    args: argparse.Namespace,
    store: SessionStore,
    state: SessionState,
    service: AgentService,
    editor: LineEditor,
) -> TerminalEvent:
    service.dispatch(ExecutePlanCommand(session_id=state.session_id))
    approved = service.next_event(timeout=1)
    if not isinstance(approved, PlanApprovedEvent):
        raise RuntimeError(
            f"expected plan.approved, got {approved.event_type.value}"
        )
    args.mode = approved.mode_id
    print(f"plan approved: {approved.plan_id}; mode: {approved.mode_id}")
    renderer = StreamRenderer(show_thinking=args.show_thinking)
    terminal_event = _consume_service_events(
        args,
        service,
        state.session_id,
        renderer,
        editor,
    )
    _save_session_artifacts(store, state)
    _render_terminal_event(args, renderer, terminal_event)
    return _offer_continuation_action(
        args, store, state, service, editor, terminal_event
    )


def _offer_continuation_action(
    args: argparse.Namespace,
    store: SessionStore,
    state: SessionState,
    service: AgentService,
    editor: LineEditor,
    terminal_event: TerminalEvent,
) -> TerminalEvent:
    while isinstance(terminal_event, TurnContinuationRequiredEvent):
        if not editor.supports_interactive_selection:
            print("turn paused; use /retry to continue or /abort to stop")
            return terminal_event
        action = editor.select(
            "continue> ",
            CONTINUATION_ACTIONS,
            current_value="Continue",
        )
        if action == "Continue":
            terminal_event = _run_chat_turn(
                args,
                store,
                state,
                service,
                editor,
                retry=True,
            )
            continue
        service.dispatch(AbortTurnCommand(session_id=state.session_id))
        aborted = service.next_event(timeout=1)
        if not isinstance(aborted, TurnAbortedEvent):
            raise RuntimeError(
                f"expected turn.aborted, got {aborted.event_type.value}"
            )
        warning = (
            " Filesystem changes from completed tools were not reverted."
            if aborted.tool_calls_executed
            else ""
        )
        print(f"aborted {aborted.turn_id}.{warning}")
        return aborted
    return terminal_event


def _offer_plan_action(
    args: argparse.Namespace,
    store: SessionStore,
    state: SessionState,
    service: AgentService,
    editor: LineEditor,
    terminal_event: TerminalEvent,
) -> None:
    if (
        not isinstance(terminal_event, TurnCompletedEvent)
        or terminal_event.stopped_by != "proposed_plan"
    ):
        return
    if not editor.supports_interactive_selection:
        print("plan ready; use /approve to execute or continue in Plan mode to revise it")
        return

    action = editor.select(
        "plan> ",
        PLAN_ACTIONS,
        current_value="Execute plan",
    )
    if action == "Execute plan":
        try:
            _approve_and_execute_plan(args, store, state, service, editor)
        except ValueError as exc:
            print(str(exc))
        return
    if action == "Revise plan":
        try:
            service.dispatch(RevisePlanCommand(session_id=state.session_id))
            revised = service.next_event(timeout=1)
            if not isinstance(revised, PlanRevisionRequestedEvent):
                raise RuntimeError(
                    f"expected plan.revision_requested, got {revised.event_type.value}"
                )
        except ValueError as exc:
            print(str(exc))
            return
        print("plan marked for revision; describe the changes you want")
        return
    print("plan kept as proposed; use /approve to execute it later")


def _run_chat_turn(
    args: argparse.Namespace,
    store: SessionStore,
    state: SessionState,
    service: AgentService,
    editor: LineEditor,
    *,
    user_input: str | None = None,
    retry: bool = False,
) -> TerminalEvent:
    mode = get_mode(service.session_view(state.session_id).mode_id)
    renderer = StreamRenderer(
        show_thinking=args.show_thinking,
        hide_plan_protocol=mode.produces_plan,
    )
    try:
        service.dispatch(
            StartTurnCommand(
                session_id=state.session_id,
                user_input=user_input or "",
                retry=retry,
            )
        )
        terminal_event = _consume_service_events(
            args,
            service,
            state.session_id,
            renderer,
            editor,
        )
    finally:
        _save_session_artifacts(store, state)
    _render_terminal_event(args, renderer, terminal_event)
    return terminal_event


def _consume_service_events(
    args: argparse.Namespace,
    service: AgentService,
    session_id: str,
    renderer: StreamRenderer,
    editor: LineEditor | None = None,
) -> TerminalEvent:
    interrupt_requested = False
    while True:
        try:
            event = service.next_event(timeout=0.1)
        except Empty:
            continue
        except KeyboardInterrupt:
            if not interrupt_requested:
                service.dispatch(InterruptTurnCommand(session_id=session_id))
                print("interrupt requested", file=sys.stderr)
                interrupt_requested = True
            continue

        if args.stream:
            renderer.handle(event)
        if isinstance(event, ApprovalRequestedEvent):
            if args.stream:
                renderer.finish("")
            if editor is None:
                raise RuntimeError("non-interactive runtime requested approval")
            decision = _prompt_for_approval(editor, event)
            if decision is None:
                service.dispatch(InterruptTurnCommand(session_id=session_id))
                interrupt_requested = True
            else:
                service.dispatch(
                    ResolveApprovalCommand(
                        session_id=session_id,
                        request_id=event.request_id,
                        decision=decision,
                    )
                )
        if isinstance(
            event,
            (
                TurnCompletedEvent,
                TurnContinuationRequiredEvent,
                TurnFailedEvent,
                TurnInterruptedEvent,
            ),
        ):
            if args.stream and not isinstance(event, TurnCompletedEvent):
                renderer.finish("")
            service.wait(session_id)
            return event


def _render_terminal_event(
    args: argparse.Namespace,
    renderer: StreamRenderer,
    event: TerminalEvent,
) -> None:
    if isinstance(event, TurnCompletedEvent):
        if args.stream:
            renderer.finish(event.answer)
        else:
            print(f"assistant> {event.answer}")
        return
    renderer.finish("")
    if isinstance(event, TurnContinuationRequiredEvent):
        print(
            f"step budget reached after {event.steps_completed} steps; "
            f"continue to run the next {event.step_budget} steps or stop the turn"
        )
        return
    if isinstance(event, TurnAbortedEvent):
        return
    if isinstance(event, TurnInterruptedEvent):
        print("generation interrupted; use /retry to continue or /abort to discard the turn")
    else:
        if not args.stream:
            print(f"error: {event.error}")
        print("use /retry to continue or /abort to discard the pending turn")


def _terminal_event_message(event: TerminalEvent) -> str:
    if isinstance(event, TurnCompletedEvent):
        return event.answer
    if isinstance(event, TurnContinuationRequiredEvent):
        return "turn requires a user decision before continuing"
    if isinstance(event, TurnAbortedEvent):
        return "turn aborted"
    if isinstance(event, TurnInterruptedEvent):
        return event.reason
    return event.error


def _prompt_for_approval(
    editor: LineEditor,
    event: ApprovalRequestedEvent,
) -> ApprovalDecision | None:
    details = event.details
    print(f"approval> {event.description or event.action}")
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
    mode = get_mode(args.mode)
    registry = build_tool_registry(mode)
    model = _build_model(args)
    loop = AgentLoop(
        workspace=workspace,
        model=model,
        tools=registry,
        permissions=PermissionPolicy(args.permission_profile),
        trace=TraceRecorder(trace_path),
        max_steps=args.max_steps,
        mode=mode,
        sandbox_policy=SandboxPolicy(mode=args.sandbox, workspace=workspace.root),
    )
    return loop, registry


def _build_server_runtime(
    args: argparse.Namespace,
    state: SessionState,
    *,
    model_ref: str | None,
    mode_id: str | None,
) -> AgentRuntime:
    session_args = copy.copy(args)
    is_new = state.provider_id is None and state.model_id is None
    session_args.provider = args.provider if is_new else None
    session_args.model = args.model if is_new else None
    session_args.mode = args.mode if is_new else None
    if model_ref is not None:
        session_args.provider = None
        session_args.model = model_ref
    if mode_id is not None:
        session_args.mode = mode_id
    _restore_session_model(session_args, state)
    _restore_session_mode(session_args, state)
    _restore_session_permission_profile(session_args, state)
    _restore_session_sandbox(session_args, state)
    workspace = Workspace(state.workspace)
    loop, _ = _build_runtime(session_args, workspace, state.trace_path)
    _sync_session_model(state, loop)
    _sync_session_mode(state, loop)
    _sync_session_permission_profile(state, loop)
    _sync_session_sandbox(state, loop)
    return AgentRuntime(loop, state, interactive_approvals=True)


def _build_agent_service(
    args: argparse.Namespace,
    store: SessionStore,
    state: SessionState,
    loop: AgentLoop,
    *,
    interactive_approvals: bool,
) -> AgentService:
    runtime = AgentRuntime(loop, state, interactive_approvals=interactive_approvals)
    service = AgentService(
        tool_registry_factory=build_tool_registry,
        model_resolver=lambda model_ref: _build_model(args, model_ref=model_ref),
        save_session=lambda session: _save_session_artifacts(store, session),
        event_store=EventStore(store.root),
    )
    service.register(runtime)
    service.open_default_event_stream(session_ids=(state.session_id,))
    return service


def _build_model(
    args: argparse.Namespace,
    *,
    provider_id: str | None = None,
    model_id: str | None = None,
    model_ref: str | None = None,
) -> ModelInvoker:
    resolved = args.runtime_config.resolve_model(
        provider_id=provider_id if provider_id is not None else args.provider,
        model_id=model_id if model_id is not None else args.model,
        model_ref=model_ref,
    )
    selected_provider = resolved.model.provider_id
    options = dict(resolved.options)
    if selected_provider == "deepseek" and args.thinking is not None:
        options["thinking"] = args.thinking
    if selected_provider == "openai" and args.reasoning_effort is not None:
        options["reasoning_effort"] = args.reasoning_effort
    adapter = create_provider_adapter(
        ProviderConfig(
            provider_id=selected_provider,
            api_key=resolved.api_key,
            model_id=resolved.model.model_id,
            base_url=args.base_url or resolved.base_url,
            context_window_tokens=(
                args.context_window_tokens
                if args.context_window_tokens is not None
                else resolved.model.context_window_tokens
            ),
            max_output_tokens=(
                args.max_output_tokens
                if args.max_output_tokens is not None
                else resolved.model.max_output_tokens
            ),
            options=options,
        )
    )
    return ModelInvoker(adapter)


def _restore_session_model(args: argparse.Namespace, state: SessionState) -> None:
    if args.model is not None and "/" in args.model:
        return
    if args.provider is None and state.provider_id:
        args.provider = state.provider_id
    if args.model is None and state.model_id and state.provider_id == args.provider:
        args.model = state.model_id


def _restore_session_mode(args: argparse.Namespace, state: SessionState) -> None:
    required_mode = (
        state.pending_turn.mode_id if state.pending_turn is not None else state.mode_id
    )
    if state.pending_turn is not None and args.mode not in {None, required_mode}:
        raise ValueError(
            f"pending turn requires mode {required_mode}; requested mode is {args.mode}"
        )
    args.mode = args.mode or required_mode
    get_mode(args.mode)


def _sync_session_model(state: SessionState, loop: AgentLoop) -> None:
    state.provider_id = loop.model.profile.provider_id
    state.model_id = loop.model.profile.model_id


def _sync_session_mode(state: SessionState, loop: AgentLoop) -> None:
    state.mode_id = loop.mode.mode_id


def _restore_session_permission_profile(
    args: argparse.Namespace,
    state: SessionState,
) -> None:
    selected = args.permission_profile or state.permission_profile_id
    args.permission_profile = get_permission_profile(selected).profile_id


def _sync_session_permission_profile(state: SessionState, loop: AgentLoop) -> None:
    state.permission_profile_id = loop.permissions.profile.profile_id


def _restore_session_sandbox(args: argparse.Namespace, state: SessionState) -> None:
    if state.pending_turn is not None and args.sandbox not in {None, state.sandbox_mode}:
        raise ValueError(
            f"pending turn requires sandbox {state.sandbox_mode}; "
            f"requested sandbox is {args.sandbox}"
        )
    args.sandbox = args.sandbox or state.sandbox_mode
    if args.sandbox not in SANDBOX_MODES:
        raise ValueError(f"unknown sandbox mode: {args.sandbox}")


def _sync_session_sandbox(state: SessionState, loop: AgentLoop) -> None:
    state.sandbox_mode = loop.sandbox_policy.mode


def _print_model_catalog(config: RuntimeConfig, current_model_ref: str) -> None:
    print(f"Current model: {current_model_ref}")
    print("Available models")
    for model in config.models():
        if model.provider_id not in SUPPORTED_PROVIDER_IDS:
            continue
        marker = "*" if model.ref == current_model_ref else " "
        print(f"  {marker} {model.ref:<32}{model.description}")


def _print_mode_catalog(current_mode_id: str) -> None:
    print(f"Current mode: {current_mode_id}")
    print("Available modes")
    for mode in COLLABORATION_MODES:
        marker = "*" if mode.mode_id == current_mode_id else " "
        print(f"  {marker} {mode.mode_id:<12}{mode.description}")


def _print_permission_catalog(current_profile_id: str) -> None:
    print(f"Current permissions: {current_profile_id}")
    print("Available permission profiles")
    for profile in PERMISSION_PROFILES:
        marker = "*" if profile.profile_id == current_profile_id else " "
        print(f"  {marker} {profile.profile_id:<16}{profile.description}")


def _print_runtime_config(config: RuntimeConfig) -> None:
    config_status = "loaded" if config.config_path.exists() else "built-in defaults"
    credentials_status = "loaded" if config.credentials_path.exists() else "not found"
    print("Provider Configuration")
    print(f"  config:      {config.config_path} ({config_status})")
    print(f"  credentials: {config.credentials_path} ({credentials_status})")
    print(f"  default:     {config.default_model}")
    print(f"  providers:   {', '.join(config.providers)}")


def build_tool_registry(mode: CollaborationMode = BUILD_MODE) -> ToolRegistry:
    tools = {
        tool.name: tool
        for tool in (
            ApplyPatchTool(),
            GitDiffTool(),
            ListFilesTool(),
            ReadFileTool(),
            SearchTool(),
            ExecCommandTool(),
        )
    }
    return ToolRegistry([tools[name] for name in mode.tool_names])


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
