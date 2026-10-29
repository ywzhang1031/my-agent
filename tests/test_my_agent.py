import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import CompleteEvent, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput

from my_agent.agent_loop import AgentEvent, AgentLoop
from my_agent.cli import (
    _build_model,
    _restore_session_mode,
    _restore_session_model,
    _restore_session_sandbox,
    build_parser,
    build_tool_registry,
    main,
)
from my_agent.config import load_runtime_config, save_provider_credential
from my_agent.context import ContextManager, estimate_text_tokens
from my_agent.control import (
    ApprovalDecision,
    ApprovalGrant,
    DenyApprovalBroker,
    PermissionRequest,
    raise_if_cancelled,
)
from my_agent.goals import GoalState, GoalStep, update_goal
from my_agent.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from my_agent.model_invoker import ModelInvoker, RetryPolicy, RetryScheduled
from my_agent.modes import BUILD_MODE, PLAN_MODE, extract_proposed_plan, get_mode
from my_agent.permissions import PermissionEffect, PermissionPolicy
from my_agent.protocol import (
    AgentState,
    ItemCompletedEvent,
    ItemDeltaEvent,
    ItemStartedEvent,
    ModelStartedEvent,
)
from my_agent.provider import (
    ModelCompleted,
    ModelError,
    ModelErrorKind,
    ModelMetadata,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ProviderCapabilities,
    ReasoningDelta,
    ScriptedProviderAdapter,
    StopReason,
    TextDelta,
)
from my_agent.providers.deepseek import DeepSeekAdapter
from my_agent.providers.factory import ProviderConfig, create_provider_adapter
from my_agent.providers.openai import OpenAIAdapter
from my_agent.runtime import (
    AgentRuntime,
    InterruptTurn,
    ResolveApproval,
    RuntimeState,
    StartTurn,
)
from my_agent.session import PendingTurn, ProposedPlan, SessionStore
from my_agent.sandbox import DANGER_FULL_ACCESS, WORKSPACE_WRITE, SandboxPolicy
from my_agent.terminal import (
    LineEditor,
    SelectionCompleter,
    SelectionOption,
    SlashCommand,
    SlashCommandCompleter,
    StreamRenderer,
    _completion_key_bindings,
    _selection_key_bindings,
    format_context_status,
    render_context_usage,
)
from my_agent.tools import (
    ApplyPatchTool,
    ExecCommandTool,
    GitDiffTool,
    ListFilesTool,
    ReadFileTool,
    SearchTool,
    ToolContext,
    ToolRegistry,
)
from my_agent.trace import TraceRecorder
from my_agent.trajectory import make_trajectory, read_jsonl_trace
from my_agent.workspace import Workspace


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )


def _commit_workspace(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=My Agent Tests",
        "-c",
        "user.email=my-agent@example.invalid",
        "commit",
        "-qm",
        "initial",
    )


def _model_response(
    *,
    content: str = "",
    reasoning_content: str = "",
    tool_calls: list[ToolCall] | tuple[ToolCall, ...] = (),
    finish_reason: str = "stop",
    usage: ModelUsage | None = None,
) -> ModelResponse:
    return ModelResponse.from_parts(
        text=content,
        reasoning=reasoning_content,
        tool_calls=tool_calls,
        stop_reason=StopReason(finish_reason),
        usage=usage,
        metadata=ModelMetadata(provider_id="scripted", model_id="scripted-model"),
    )


def _model(adapter, *, max_attempts: int = 1) -> ModelInvoker:
    return ModelInvoker(
        adapter,
        RetryPolicy(
            max_attempts=max_attempts,
            initial_delay_seconds=0,
            max_delay_seconds=0,
        ),
    )


class MyAgentTests(unittest.TestCase):
    def test_runtime_config_loads_builtin_provider_catalog(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_runtime_config(
                Path(tmpdir, "config.json"),
                Path(tmpdir, "credentials.json"),
            )

        self.assertEqual(config.default_model, "deepseek/deepseek-v4-flash")
        self.assertEqual(
            [model.ref for model in config.models()],
            [
                "deepseek/deepseek-v4-flash",
                "deepseek/deepseek-v4-pro",
                "openai/gpt-5.6",
            ],
        )
        openai = config.resolve_model(model_ref="openai/gpt-5.6")
        self.assertEqual(openai.base_url, "https://api.openai.com/v1")
        self.assertEqual(openai.model.context_window_tokens, 1_050_000)
        self.assertEqual(
            config.resolve_model(model_id="openai/gpt-5.6").model.ref,
            "openai/gpt-5.6",
        )
        with self.assertRaisesRegex(ValueError, "conflicts with model reference"):
            config.resolve_model(
                provider_id="deepseek",
                model_id="openai/gpt-5.6",
            )

    def test_runtime_config_merges_models_and_reads_secure_credentials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir, "config.json")
            credentials_path = Path(tmpdir, "credentials.json")
            config_path.write_text(
                json.dumps(
                    {
                        "default_model": "openai/gpt-test",
                        "providers": {
                            "openai": {
                                "default_model": "gpt-test",
                                "options": {"reasoning_effort": "low"},
                                "models": {
                                    "gpt-test": {
                                        "description": "Test model",
                                        "context_window_tokens": 10_000,
                                        "max_output_tokens": 1_000,
                                    }
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            credentials_path.write_text(
                json.dumps(
                    {
                        "deepseek": {"type": "api", "key": "deepseek-file-key"},
                        "openai": {"type": "api", "key": "openai-file-key"},
                    }
                ),
                encoding="utf-8",
            )
            os.chmod(credentials_path, 0o600)

            with patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": ""},
            ):
                config = load_runtime_config(config_path, credentials_path)
                resolved = config.resolve_model()
                deepseek = config.resolve_model(
                    model_ref="deepseek/deepseek-v4-flash"
                )

            self.assertEqual(resolved.model.ref, "openai/gpt-test")
            self.assertEqual(resolved.api_key, "openai-file-key")
            self.assertEqual(resolved.options["reasoning_effort"], "low")
            self.assertEqual(deepseek.api_key, "deepseek-file-key")
            self.assertEqual(deepseek.options["thinking"], "enabled")
            self.assertIn("openai/gpt-5.6", [model.ref for model in config.models()])

            with patch.dict(os.environ, {"OPENAI_API_KEY": "environment-key"}):
                overridden = load_runtime_config(
                    config_path,
                    credentials_path,
                ).resolve_model()

            self.assertEqual(overridden.api_key, "environment-key")

    def test_provider_adapters_do_not_bypass_resolved_credentials(self):
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "unresolved-deepseek-key",
                "OPENAI_API_KEY": "unresolved-openai-key",
            },
        ):
            deepseek = create_provider_adapter(
                ProviderConfig(
                    provider_id="deepseek",
                    model_id="deepseek-v4-flash",
                    options={"thinking": "enabled"},
                )
            )
            openai = create_provider_adapter(
                ProviderConfig(
                    provider_id="openai",
                    model_id="gpt-5.6",
                    options={"reasoning_effort": "medium"},
                )
            )

        self.assertIsNone(deepseek.api_key)
        self.assertIsNone(openai.api_key)

    def test_runtime_config_rejects_readable_credentials_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = Path(tmpdir, "credentials.json")
            credentials_path.write_text(
                json.dumps({"openai": {"type": "api", "key": "secret"}}),
                encoding="utf-8",
            )
            os.chmod(credentials_path, 0o644)

            with self.assertRaisesRegex(ValueError, "mode 0600"):
                load_runtime_config(Path(tmpdir, "config.json"), credentials_path)

    def test_save_provider_credential_is_secure_and_preserves_other_providers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir, "config.json")
            credentials_path = Path(tmpdir, "auth", "credentials.json")
            with patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": ""},
            ):
                config = load_runtime_config(config_path, credentials_path)
                config = save_provider_credential(config, "deepseek", "deepseek-key")
                config = save_provider_credential(config, "openai", "openai-key")

                self.assertEqual(
                    config.resolve_model(model_ref="deepseek/deepseek-v4-flash").api_key,
                    "deepseek-key",
                )
                self.assertEqual(
                    config.resolve_model(model_ref="openai/gpt-5.6").api_key,
                    "openai-key",
                )

            saved = json.loads(credentials_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["deepseek"]["key"], "deepseek-key")
            self.assertEqual(saved["openai"]["key"], "openai-key")
            self.assertEqual(credentials_path.stat().st_mode & 0o777, 0o600)

    def test_list_files_ignores_session_state_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "app.py").write_text("print('hello')\n", encoding="utf-8")
            session_dir = Path(tmpdir, ".my-agent", "sessions", "session-1")
            session_dir.mkdir(parents=True)
            Path(session_dir, "messages.jsonl").write_text("{}\n", encoding="utf-8")

            files, _ = Workspace(tmpdir).list_files()

            self.assertEqual(files, ["app.py"])

    def test_session_store_round_trips_canonical_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            store = SessionStore(Path(tmpdir, "sessions"))
            state = store.create(workspace_path)
            state.messages.extend(
                [
                    UserMessage(content="Inspect files"),
                    AssistantMessage(
                        content="I will inspect first.",
                        reasoning_content="Need repository evidence.",
                        provider_data={
                            "openai": {
                                "input_items": [
                                    {
                                        "type": "reasoning",
                                        "encrypted_content": "opaque",
                                    }
                                ]
                            }
                        },
                        tool_calls=[
                            ToolCall(
                                call_id="call_1",
                                name="list_files",
                                arguments={"path": "."},
                            )
                        ],
                    ),
                    ToolResultMessage(
                        tool_call_id="call_1",
                        tool_name="list_files",
                        content=json.dumps({"ok": True, "stdout": "app.py"}),
                    ),
                ]
            )
            state.context_summary = "Earlier repository inspection completed."
            state.summarized_message_count = 1
            state.last_input_tokens = 123
            state.last_output_tokens = 45
            state.provider_id = "deepseek"
            state.model_id = "deepseek-v4-pro"
            state.mode_id = "build"
            state.permission_profile_id = "accept_edits"
            state.sandbox_mode = DANGER_FULL_ACCESS
            state.goal = GoalState(
                goal_id="goal-1",
                objective="Update and verify app.py",
                steps=(
                    GoalStep("Inspect app.py", "completed"),
                    GoalStep("Update app.py", "in_progress"),
                    GoalStep("Run tests", "pending"),
                ),
                status="active",
                updated_at="2026-10-24T00:00:00+00:00",
            )
            state.proposed_plan = ProposedPlan(
                plan_id=f"{state.session_id}:1:plan",
                source_turn_id=f"{state.session_id}:1",
                content="# Plan\n\nUpdate app.py.",
                status="approved",
            )
            state.messages.append(UserMessage(content="Retry this request"))
            state.pending_turn = PendingTurn(
                turn_id=f"{state.session_id}:2",
                task="Retry this request",
                message_start=3,
                step=2,
                tool_calls_executed=1,
                seen_calls={"call-key": 1},
                mode_id="build",
                approved_plan_id=state.proposed_plan.plan_id,
                last_error="API key missing",
                last_error_retryable=False,
                last_error_details={
                    "kind": "configuration",
                    "provider_id": "deepseek",
                    "action_required": "configure_api_key",
                },
                pause_reason="step_budget",
                pause_details={
                    "steps_completed": 12,
                    "step_budget": 12,
                    "next_step": 13,
                },
            )

            store.save(state)
            loaded = store.load(state.session_id)

            self.assertEqual(loaded.session_id, state.session_id)
            self.assertEqual(loaded.workspace, str(workspace_path.resolve()))
            self.assertEqual(loaded.messages, state.messages)
            self.assertEqual(loaded.context_summary, state.context_summary)
            self.assertEqual(loaded.summarized_message_count, 1)
            self.assertEqual(loaded.last_input_tokens, 123)
            self.assertEqual(loaded.last_output_tokens, 45)
            self.assertEqual(loaded.provider_id, "deepseek")
            self.assertEqual(loaded.model_id, "deepseek-v4-pro")
            self.assertEqual(loaded.mode_id, "build")
            self.assertEqual(loaded.permission_profile_id, "accept_edits")
            self.assertEqual(loaded.sandbox_mode, DANGER_FULL_ACCESS)
            self.assertEqual(loaded.goal, state.goal)
            self.assertEqual(loaded.proposed_plan, state.proposed_plan)
            self.assertEqual(loaded.pending_turn, state.pending_turn)
            self.assertTrue(loaded.metadata_path.exists())
            self.assertTrue(loaded.messages_path.exists())
            self.assertEqual(store.list_sessions()[0].session_id, state.session_id)

    def test_agent_loop_run_turn_reuses_history_and_records_turn_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            store = SessionStore(Path(tmpdir, "sessions"))
            state = store.create(workspace_path)
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        content="First answer.",
                        reasoning_content="First reasoning.",
                        usage=ModelUsage(input_tokens=10, output_tokens=2),
                    ),
                    _model_response(content="Second answer."),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([ListFilesTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=4,
            )

            first = loop.run_turn(state, "First question")
            second = loop.run_turn(state, "Second question")
            store.save(state)

            self.assertEqual(first.answer, "First answer.")
            self.assertEqual(first.stopped_by, "final_answer")
            self.assertEqual(second.answer, "Second answer.")
            self.assertEqual(len(provider.requests[0].messages), 1)
            self.assertEqual(len(provider.requests[1].messages), 3)
            self.assertEqual(
                provider.requests[1].messages[0],
                UserMessage(content="First question"),
            )
            self.assertEqual(
                provider.requests[1].messages[1],
                AssistantMessage(
                    content="First answer.",
                    reasoning_content="First reasoning.",
                ),
            )
            self.assertEqual(
                provider.requests[1].messages[2],
                UserMessage(content="Second question"),
            )
            self.assertEqual(store.load(state.session_id).messages, state.messages)

            events = [
                json.loads(line)
                for line in state.trace_path.read_text(encoding="utf-8").splitlines()
            ]
            turn_events = [event for event in events if event["event"] == "turn_started"]
            model_events = [event for event in events if event["event"] == "model_response"]
            self.assertEqual(len(turn_events), 2)
            self.assertNotEqual(turn_events[0]["turn_id"], turn_events[1]["turn_id"])
            self.assertEqual(model_events[0]["usage"]["input_tokens"], 10)
            self.assertIsNone(model_events[1]["usage"])

    def test_agent_loop_retries_pending_turn_without_duplicating_user_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            store = SessionStore(Path(tmpdir, "sessions"))
            state = store.create(workspace_path)
            provider = ScriptedProviderAdapter(
                [
                    ModelError("temporary provider failure", retryable=True),
                    _model_response(content="Recovered answer."),
                ]
            )
            events: list[AgentEvent] = []
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([ListFilesTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=4,
            )

            with self.assertRaisesRegex(ModelError, "temporary provider failure"):
                loop.run_turn(state, "Inspect this repository.", event_handler=events.append)

            self.assertIsNotNone(state.pending_turn)
            self.assertEqual(state.pending_turn.step, 1)
            self.assertEqual(
                sum(isinstance(message, UserMessage) for message in state.messages),
                1,
            )

            result = loop.retry_turn(state, event_handler=events.append)

            self.assertEqual(result.answer, "Recovered answer.")
            self.assertIsNone(state.pending_turn)
            self.assertEqual(
                sum(isinstance(message, UserMessage) for message in state.messages),
                1,
            )
            self.assertEqual(len(provider.requests), 2)
            self.assertEqual(provider.requests[0], provider.requests[1])
            self.assertIn("content_delta", [event.kind for event in events])

            trace_events = read_jsonl_trace(state.trace_path)
            event_names = [event["event"] for event in trace_events]
            self.assertEqual(event_names.count("turn_started"), 1)
            self.assertIn("turn_error", event_names)
            self.assertIn("turn_resumed", event_names)
            self.assertIn("final_answer", event_names)
            self.assertEqual(
                {event["turn_id"] for event in trace_events},
                {state.session_id + ":1"},
            )
            trajectory = make_trajectory(trace_events, source_path=state.trace_path)
            self.assertEqual(trajectory["schema_version"], "my-agent.trajectory.v8")
            self.assertEqual(trajectory["turns"][0]["outcome"]["status"], "completed")
            self.assertEqual(len(trajectory["turns"][0]["errors"]), 1)
            self.assertEqual(trajectory["metrics"]["turn_errors"], 1)

    def test_agent_loop_persists_structured_api_key_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            provider = ScriptedProviderAdapter(
                [
                    ModelError(
                        "API key is invalid",
                        kind=ModelErrorKind.AUTHENTICATION,
                        provider_id="deepseek",
                        model_id="deepseek-v4-flash",
                        status_code=401,
                    )
                ]
            )
            events: list[AgentEvent] = []
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=2,
            )

            with self.assertRaisesRegex(ModelError, "API key is invalid"):
                loop.run_turn(state, "hello", event_handler=events.append)

            self.assertEqual(state.pending_turn.last_error, "API key is invalid")
            self.assertFalse(state.pending_turn.last_error_retryable)
            self.assertEqual(
                state.pending_turn.last_error_details["action_required"],
                "configure_api_key",
            )
            error_event = next(event for event in events if event.kind == "turn_error")
            self.assertEqual(error_event.data["details"]["kind"], "authentication")

    def test_agent_loop_records_harness_retry_without_creating_pending_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            provider = ScriptedProviderAdapter(
                [
                    ModelError(
                        "server overloaded",
                        kind=ModelErrorKind.SERVER,
                        retryable=True,
                        provider_id="scripted",
                        model_id="scripted-model",
                        status_code=503,
                    ),
                    _model_response(content="Recovered answer."),
                ]
            )
            events: list[AgentEvent] = []
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider, max_attempts=2),
                tools=ToolRegistry([]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=2,
            )

            result = loop.run_turn(state, "Retry transient failures", events.append)
            trace_events = read_jsonl_trace(state.trace_path)
            retry = next(event for event in trace_events if event["event"] == "provider_retry")

            self.assertEqual(result.answer, "Recovered answer.")
            self.assertEqual(len(provider.requests), 2)
            self.assertEqual(retry["kind"], "server")
            self.assertEqual(retry["status_code"], 503)
            self.assertEqual(retry["attempt"], 2)
            request = next(event for event in trace_events if event["event"] == "model_request")
            self.assertEqual(request["provider_id"], "scripted")
            self.assertEqual(request["model_id"], "scripted-model")
            self.assertNotIn("turn_error", [event["event"] for event in trace_events])
            self.assertIn("provider_retry", [event.kind for event in events])

    def test_agent_loop_retry_after_tool_result_does_not_repeat_the_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            (workspace_path / "app.py").write_text("value = 1\n", encoding="utf-8")
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        tool_calls=[
                            ToolCall(
                                call_id="call_list",
                                name="list_files",
                                arguments={"path": "."},
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    ModelError("second model request failed", retryable=True),
                    _model_response(content="Found app.py."),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([ListFilesTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=4,
            )

            with self.assertRaisesRegex(ModelError, "second model request failed"):
                loop.run_turn(state, "Inspect files")

            self.assertEqual(state.pending_turn.step, 2)
            self.assertEqual(state.pending_turn.tool_calls_executed, 1)
            self.assertIsInstance(state.messages[-1], ToolResultMessage)

            result = loop.retry_turn(state)
            trace_events = read_jsonl_trace(state.trace_path)

            self.assertEqual(result.answer, "Found app.py.")
            self.assertEqual(
                sum(event["event"] == "tool_call" for event in trace_events),
                1,
            )
            self.assertEqual(len(provider.requests), 3)
            self.assertEqual(provider.requests[1], provider.requests[2])

    def test_agent_loop_abort_removes_pending_messages_and_records_outcome(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(ScriptedProviderAdapter([ModelError("failed")])),
                tools=ToolRegistry([]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=2,
            )

            with self.assertRaises(ModelError):
                loop.run_turn(state, "Failed request")
            pending = loop.abort_turn(state)
            trajectory = make_trajectory(read_jsonl_trace(state.trace_path), state.trace_path)

            self.assertEqual(pending.task, "Failed request")
            self.assertIsNone(state.pending_turn)
            self.assertEqual(state.messages, [])
            self.assertEqual(trajectory["turns"][0]["outcome"]["status"], "aborted")
            self.assertEqual(trajectory["metrics"]["turn_aborts"], 1)

    def test_step_budget_pauses_and_each_retry_gets_a_new_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        tool_calls=[
                            ToolCall(
                                call_id="call-1",
                                name="list_files",
                                arguments={"path": ".", "max_files": 5},
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _model_response(
                        tool_calls=[
                            ToolCall(
                                call_id="call-2",
                                name="list_files",
                                arguments={"path": ".", "max_files": 10},
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _model_response(content="Inspection complete."),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([ListFilesTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=1,
            )

            first_batch = loop.run_turn(state, "Inspect files")

            self.assertEqual(first_batch.stopped_by, "continuation_required")
            self.assertEqual(first_batch.answer, "")
            self.assertIsNotNone(state.pending_turn)
            self.assertEqual(state.pending_turn.step, 2)
            self.assertEqual(state.pending_turn.pause_reason, "step_budget")
            self.assertEqual(state.pending_turn.pause_details["next_step"], 2)
            self.assertNotEqual(
                getattr(state.messages[-1], "content", ""),
                "Stopped after reaching max_steps=1 without a final answer.",
            )

            second_batch = loop.retry_turn(state)

            self.assertEqual(second_batch.stopped_by, "continuation_required")
            self.assertEqual(state.pending_turn.step, 3)
            self.assertEqual(state.pending_turn.pause_details["steps_completed"], 2)

            completed = loop.retry_turn(state)
            trace_events = read_jsonl_trace(state.trace_path)
            trajectory = make_trajectory(trace_events, state.trace_path)

            self.assertEqual(completed.answer, "Inspection complete.")
            self.assertIsNone(state.pending_turn)
            self.assertEqual(len(provider.requests), 3)
            self.assertEqual(
                sum(
                    event["event"] == "turn_continuation_required"
                    for event in trace_events
                ),
                2,
            )
            self.assertEqual(
                sum(event["event"] == "final_answer" for event in trace_events),
                1,
            )
            self.assertEqual(len(trajectory["turns"][0]["continuation_events"]), 2)
            self.assertEqual(trajectory["metrics"]["turn_continuation_requests"], 2)

    def test_plan_mode_proposes_approved_plan_then_build_completes_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            plan_provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        content=(
                            "<proposed_plan>\n"
                            "# Plan\n\n"
                            "1. Update `app.py`.\n"
                            "2. Run the focused test.\n"
                            "</proposed_plan>"
                        )
                    )
                ]
            )
            plan_loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(plan_provider),
                tools=build_tool_registry(PLAN_MODE),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                mode=PLAN_MODE,
            )

            planned = plan_loop.run_turn(state, "Plan the change")
            approved = plan_loop.approve_plan(state)

            build_provider = ScriptedProviderAdapter(
                [_model_response(content="Implemented and validated the change.")]
            )
            build_loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(build_provider),
                tools=build_tool_registry(BUILD_MODE),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                mode=BUILD_MODE,
            )
            built = build_loop.run_turn(
                state,
                "Execute the approved plan.",
                approved_plan_id=approved.plan_id,
            )
            events = read_jsonl_trace(state.trace_path)

            self.assertEqual(planned.stopped_by, "proposed_plan")
            self.assertEqual(planned.answer, approved.content)
            self.assertNotIn("<proposed_plan>", state.messages[1].content)
            self.assertEqual(plan_provider.requests[0].system_prompt, PLAN_MODE.instructions)
            self.assertEqual(
                sorted(tool.name for tool in plan_provider.requests[0].tools),
                ["git_diff", "list_files", "read_file", "search"],
            )
            self.assertEqual(built.stopped_by, "final_answer")
            self.assertEqual(state.proposed_plan.status, "completed")
            trajectory = make_trajectory(events, state.trace_path)
            self.assertEqual(trajectory["schema_version"], "my-agent.trajectory.v8")
            self.assertEqual(trajectory["turns"][0]["mode_id"], "plan")
            self.assertEqual(
                trajectory["turns"][0]["outcome"]["stopped_by"],
                "proposed_plan",
            )
            self.assertEqual(trajectory["turns"][1]["mode_id"], "build")
            self.assertEqual(
                trajectory["turns"][1]["approved_plan_id"],
                approved.plan_id,
            )
            self.assertEqual(trajectory["metrics"]["plan_proposals"], 1)
            self.assertEqual(trajectory["metrics"]["plan_approvals"], 1)
            self.assertEqual(trajectory["metrics"]["plan_completions"], 1)
            self.assertEqual(
                [
                    event["event"]
                    for event in events
                    if event["event"]
                    in {"plan_proposed", "plan_approved", "plan_completed"}
                ],
                ["plan_proposed", "plan_approved", "plan_completed"],
            )

    def test_aborted_plan_execution_returns_plan_to_proposed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            state.proposed_plan = ProposedPlan(
                plan_id=f"{state.session_id}:1:plan",
                source_turn_id=f"{state.session_id}:1",
                content="# Plan\n\nUpdate app.py.",
                status="approved",
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(ScriptedProviderAdapter([ModelError("failed")])),
                tools=build_tool_registry(BUILD_MODE),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                mode=BUILD_MODE,
            )

            with self.assertRaises(ModelError):
                loop.run_turn(
                    state,
                    "Execute the approved plan.",
                    approved_plan_id=state.proposed_plan.plan_id,
                )
            loop.abort_turn(state)

            self.assertEqual(state.proposed_plan.status, "proposed")
            events = read_jsonl_trace(state.trace_path)
            incomplete = next(
                event for event in events if event["event"] == "plan_execution_incomplete"
            )
            self.assertEqual(incomplete["reason"], "aborted")

    def test_truncated_plan_execution_does_not_mark_plan_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            state.proposed_plan = ProposedPlan(
                plan_id=f"{state.session_id}:1:plan",
                source_turn_id=f"{state.session_id}:1",
                content="# Plan\n\nUpdate app.py.",
                status="approved",
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(
                    ScriptedProviderAdapter(
                        [_model_response(content="Partial result", finish_reason="length")]
                    )
                ),
                tools=build_tool_registry(BUILD_MODE),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                mode=BUILD_MODE,
            )

            result = loop.run_turn(
                state,
                "Execute the approved plan.",
                approved_plan_id=state.proposed_plan.plan_id,
            )

            self.assertEqual(result.stopped_by, "model_length")
            self.assertEqual(state.proposed_plan.status, "proposed")
            incomplete = next(
                event
                for event in read_jsonl_trace(state.trace_path)
                if event["event"] == "plan_execution_incomplete"
            )
            self.assertEqual(incomplete["reason"], "model_length")

    def test_plan_revision_requires_a_new_proposal_before_approval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            store = SessionStore(Path(tmpdir, "sessions"))
            state = store.create(workspace_path)
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(
                    ScriptedProviderAdapter(
                        [
                            _model_response(
                                content=(
                                    "<proposed_plan>\n# Plan\n\nUpdate app.py.\n"
                                    "</proposed_plan>"
                                )
                            )
                        ]
                    )
                ),
                tools=build_tool_registry(PLAN_MODE),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                mode=PLAN_MODE,
            )

            loop.run_turn(state, "Plan the change")
            loop.request_plan_revision(state)
            store.save(state)

            self.assertEqual(state.proposed_plan.status, "revising")
            with self.assertRaisesRegex(ValueError, "no proposed plan to approve"):
                loop.approve_plan(state)
            self.assertEqual(
                store.load(state.session_id).proposed_plan.status,
                "revising",
            )
            trajectory = make_trajectory(
                read_jsonl_trace(state.trace_path), state.trace_path
            )
            self.assertEqual(trajectory["metrics"]["plan_revision_requests"], 1)
            self.assertEqual(
                trajectory["turns"][0]["plan_events"][-1]["event"],
                "plan_revision_requested",
            )

    def test_agent_runtime_waits_for_patch_approval_then_resumes_same_turn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            target = workspace_path / "app.py"
            target.write_text("value = 1\n", encoding="utf-8")
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        tool_calls=[
                            ToolCall(
                                call_id="call_patch",
                                name="apply_patch",
                                arguments={
                                    "patch": (
                                        "*** Begin Patch\n"
                                        "*** Update File: app.py\n"
                                        "@@\n"
                                        "-value = 1\n"
                                        "+value = 2\n"
                                        "*** End Patch"
                                    )
                                },
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _model_response(content="Updated app.py."),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([ApplyPatchTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=3,
            )
            runtime = AgentRuntime(loop, state, interactive_approvals=True)

            runtime.send(StartTurn(user_input="Update app.py"))
            events = []
            while True:
                event = runtime.next_event(timeout=2)
                events.append(event)
                if event.kind == "approval_requested":
                    break

            self.assertEqual(runtime.state, RuntimeState.WAITING_APPROVAL)
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")
            runtime.send(
                ResolveApproval(
                    request_id=events[-1].data["request_id"],
                    decision=ApprovalDecision.ALLOW_ONCE,
                )
            )

            while events[-1].kind != "turn_completed":
                events.append(runtime.next_event(timeout=2))

            runtime.close()
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 2\n")
            self.assertIsNone(state.pending_turn)
            self.assertEqual(events[-1].result.answer, "Updated app.py.")
            event_names = [event.kind for event in events]
            self.assertIn("tool_proposed", event_names)
            self.assertIn("approval_resolved", event_names)
            self.assertIn("tool_started", event_names)
            trace_names = [
                event["event"] for event in read_jsonl_trace(state.trace_path)
            ]
            self.assertIn("approval_requested", trace_names)
            self.assertIn("approval_resolved", trace_names)
            trajectory = make_trajectory(
                read_jsonl_trace(state.trace_path),
                state.trace_path,
            )
            approval = trajectory["turns"][0]["steps"][0]["approvals"][0]
            self.assertEqual(approval["call_id"], "call_patch")
            self.assertEqual(approval["decision"], "allow_once")
            self.assertEqual(approval["source"], "prompt")
            self.assertTrue(approval["prompted"])
            self.assertEqual(trajectory["metrics"]["approval_requests"], 1)

    def test_agent_runtime_waits_for_user_at_step_budget_then_resumes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        tool_calls=[
                            ToolCall(
                                call_id="call-list",
                                name="list_files",
                                arguments={"path": ".", "max_files": 5},
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _model_response(content="Done."),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([ListFilesTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=1,
            )
            runtime = AgentRuntime(loop, state, interactive_approvals=True)

            runtime.send(StartTurn(user_input="Inspect files"))
            while True:
                paused = runtime.next_event(timeout=2)
                if paused.kind == "turn_continuation_required":
                    break

            runtime.wait()
            self.assertEqual(runtime.state, RuntimeState.WAITING_USER)
            self.assertEqual(paused.data["next_step"], 2)
            self.assertIsNotNone(state.pending_turn)

            runtime.send(StartTurn(retry=True))
            while True:
                completed = runtime.next_event(timeout=2)
                if completed.kind == "turn_completed":
                    break

            runtime.close()
            self.assertEqual(completed.result.answer, "Done.")
            self.assertEqual(runtime.state, RuntimeState.COMPLETED)
            self.assertIsNone(state.pending_turn)

    def test_trusted_profile_runs_hard_safe_patch_without_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            target = workspace_path / "app.py"
            target.write_text("value = 1\n", encoding="utf-8")
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        tool_calls=[
                            ToolCall(
                                call_id="call_patch",
                                name="apply_patch",
                                arguments={
                                    "patch": (
                                        "*** Begin Patch\n"
                                        "*** Update File: app.py\n"
                                        "@@\n"
                                        "-value = 1\n"
                                        "+value = 2\n"
                                        "*** End Patch"
                                    )
                                },
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _model_response(content="Updated app.py."),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([ApplyPatchTool()]),
                permissions=PermissionPolicy("trusted"),
                trace=TraceRecorder(state.trace_path),
                max_steps=3,
            )
            runtime = AgentRuntime(loop, state, interactive_approvals=True)

            runtime.send(StartTurn(user_input="Update app.py"))
            events = []
            while True:
                event = runtime.next_event(timeout=2)
                events.append(event)
                if event.kind == "turn_completed":
                    break

            runtime.close()
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 2\n")
            self.assertNotIn("approval_requested", [event.kind for event in events])
            resolved = [event for event in events if event.kind == "approval_resolved"]
            self.assertEqual(len(resolved), 1)
            self.assertEqual(resolved[0].data["source"], "profile:trusted")

    def test_agent_runtime_reuses_exact_session_approval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            target = workspace_path / "app.py"
            target.write_text("value = 1\n", encoding="utf-8")
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        tool_calls=[
                            ToolCall(
                                call_id="call_patch_1",
                                name="apply_patch",
                                arguments={
                                    "patch": (
                                        "*** Begin Patch\n"
                                        "*** Update File: app.py\n"
                                        "@@\n"
                                        "-value = 1\n"
                                        "+value = 2\n"
                                        "*** End Patch"
                                    )
                                },
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _model_response(
                        tool_calls=[
                            ToolCall(
                                call_id="call_patch_2",
                                name="apply_patch",
                                arguments={
                                    "patch": (
                                        "*** Begin Patch\n"
                                        "*** Update File: app.py\n"
                                        "@@\n"
                                        "-value = 2\n"
                                        "+value = 3\n"
                                        "*** End Patch"
                                    )
                                },
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _model_response(content="Updated app.py twice."),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([ApplyPatchTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=4,
            )
            runtime = AgentRuntime(loop, state, interactive_approvals=True)

            runtime.send(StartTurn(user_input="Update app.py twice"))
            events = []
            while True:
                event = runtime.next_event(timeout=2)
                events.append(event)
                if event.kind == "approval_requested":
                    runtime.send(
                        ResolveApproval(
                            request_id=event.data["request_id"],
                            decision=ApprovalDecision.ALLOW_SESSION,
                        )
                    )
                if event.kind == "turn_completed":
                    break

            runtime.close()
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 3\n")
            self.assertEqual(
                sum(event.kind == "approval_requested" for event in events),
                1,
            )
            resolved = [event for event in events if event.kind == "approval_resolved"]
            self.assertEqual(len(resolved), 2)
            self.assertEqual(resolved[0].data["source"], "prompt")
            self.assertEqual(resolved[1].data["source"], "session")
            self.assertEqual(
                runtime.permission_grants(),
                (ApprovalGrant("apply_patch", "app.py", "update app.py"),),
            )
            self.assertTrue(runtime.revoke_permission_grant("apply_patch", "app.py"))
            self.assertEqual(runtime.permission_grants(), ())

    def test_agent_runtime_interrupts_approval_with_complete_tool_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            first = workspace_path / "first.py"
            second = workspace_path / "second.py"
            first.write_text("value = 1\n", encoding="utf-8")
            second.write_text("value = 1\n", encoding="utf-8")
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        tool_calls=[
                            ToolCall(
                                call_id="call_first",
                                name="apply_patch",
                                arguments={
                                    "patch": (
                                        "*** Begin Patch\n"
                                        "*** Update File: first.py\n"
                                        "@@\n"
                                        "-value = 1\n"
                                        "+value = 2\n"
                                        "*** End Patch"
                                    )
                                },
                            ),
                            ToolCall(
                                call_id="call_second",
                                name="apply_patch",
                                arguments={
                                    "patch": (
                                        "*** Begin Patch\n"
                                        "*** Update File: second.py\n"
                                        "@@\n"
                                        "-value = 1\n"
                                        "+value = 2\n"
                                        "*** End Patch"
                                    )
                                },
                            ),
                        ],
                        finish_reason="tool_calls",
                    ),
                    _model_response(content="No files were changed."),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([ApplyPatchTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=3,
            )
            runtime = AgentRuntime(loop, state, interactive_approvals=True)

            runtime.send(StartTurn(user_input="Update both files"))
            while True:
                event = runtime.next_event(timeout=2)
                if event.kind == "approval_requested":
                    runtime.send(InterruptTurn())
                if event.kind == "turn_interrupted":
                    break

            self.assertEqual(first.read_text(encoding="utf-8"), "value = 1\n")
            self.assertEqual(second.read_text(encoding="utf-8"), "value = 1\n")
            tool_results = [
                message
                for message in state.messages
                if isinstance(message, ToolResultMessage)
            ]
            self.assertEqual(len(tool_results), 2)
            self.assertTrue(
                all(
                    json.loads(message.content)["metadata"]["status"] == "interrupted"
                    for message in tool_results
                )
            )

            runtime.send(StartTurn(retry=True))
            while True:
                event = runtime.next_event(timeout=2)
                if event.kind == "turn_completed":
                    break

            runtime.close()
            self.assertEqual(event.result.answer, "No files were changed.")
            self.assertEqual(len(provider.requests), 2)
            self.assertEqual(
                sum(
                    isinstance(message, ToolResultMessage)
                    for message in provider.requests[1].messages
                ),
                2,
            )

    def test_agent_runtime_denial_becomes_a_tool_observation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            target = workspace_path / "app.py"
            target.write_text("value = 1\n", encoding="utf-8")
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        tool_calls=[
                            ToolCall(
                                call_id="call_denied",
                                name="apply_patch",
                                arguments={
                                    "patch": (
                                        "*** Begin Patch\n"
                                        "*** Update File: app.py\n"
                                        "@@\n"
                                        "-value = 1\n"
                                        "+value = 2\n"
                                        "*** End Patch"
                                    )
                                },
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _model_response(content="The change was denied."),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([ApplyPatchTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=3,
            )
            runtime = AgentRuntime(loop, state, interactive_approvals=True)

            runtime.send(StartTurn(user_input="Update app.py"))
            events = []
            while True:
                event = runtime.next_event(timeout=2)
                events.append(event)
                if event.kind == "approval_requested":
                    runtime.send(
                        ResolveApproval(
                            request_id=event.data["request_id"],
                            decision=ApprovalDecision.DENY,
                        )
                    )
                if event.kind == "turn_completed":
                    break

            runtime.close()
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")
            self.assertNotIn("tool_started", [event.kind for event in events])
            trajectory = make_trajectory(
                read_jsonl_trace(state.trace_path),
                state.trace_path,
            )
            observation = trajectory["turns"][0]["steps"][0]["observations"][0]
            self.assertEqual(observation["output"]["metadata"]["status"], "denied")
            self.assertEqual(trajectory["metrics"]["approval_denials"], 1)

    def test_agent_runtime_interrupts_active_turn_and_keeps_it_retryable(self):
        class BlockingAdapter:
            def __init__(self):
                self.started = threading.Event()
                self.profile = ModelProfile(
                    provider_id="blocking",
                    model_id="blocking-model",
                    context_window_tokens=10_000,
                    max_output_tokens=1_000,
                    capabilities=ProviderCapabilities(),
                )

            def stream_once(
                self,
                request,
                cancel_event=None,
            ):
                self.started.set()
                while cancel_event is None or not cancel_event.wait(0.01):
                    pass
                raise_if_cancelled(cancel_event)
                yield

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            provider = BlockingAdapter()
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=2,
            )
            runtime = AgentRuntime(loop, state, interactive_approvals=True)

            runtime.send(StartTurn(user_input="Wait for interrupt"))
            self.assertTrue(provider.started.wait(1))
            runtime.send(InterruptTurn())

            while True:
                event = runtime.next_event(timeout=2)
                if event.kind == "turn_interrupted":
                    break

            runtime.close()
            self.assertEqual(runtime.state, RuntimeState.INTERRUPTED)
            self.assertIsNotNone(state.pending_turn)
            self.assertEqual(state.pending_turn.step, 1)
            self.assertIn(
                "turn_interrupted",
                [event["event"] for event in read_jsonl_trace(state.trace_path)],
            )
            trajectory = make_trajectory(
                read_jsonl_trace(state.trace_path),
                state.trace_path,
            )
            self.assertEqual(trajectory["outcome"]["status"], "interrupted")
            self.assertEqual(trajectory["metrics"]["turn_interruptions"], 1)

    def test_agent_runtime_switches_model_and_context_limits_while_idle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            initial = ScriptedProviderAdapter(
                [],
                context_window_tokens=10_000,
                max_output_tokens=1_000,
            )
            replacement = ScriptedProviderAdapter(
                [],
                context_window_tokens=20_000,
                max_output_tokens=2_000,
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(initial),
                tools=ToolRegistry([]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
            )
            runtime = AgentRuntime(loop, state, interactive_approvals=True)

            runtime.switch_model(_model(replacement))
            snapshot = loop.inspect_context(state)

            self.assertIs(loop.model.adapter, replacement)
            self.assertEqual(snapshot.context_window_tokens, 20_000)
            self.assertEqual(snapshot.reserve_output_tokens, 2_000)

            state.messages.append(UserMessage(content="pending"))
            state.pending_turn = PendingTurn(
                turn_id=f"{state.session_id}:1",
                task="pending",
                message_start=0,
            )
            with self.assertRaisesRegex(RuntimeError, "turn is pending"):
                runtime.switch_model(_model(initial))

    def test_agent_runtime_switches_sandbox_only_at_turn_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(ScriptedProviderAdapter([])),
                tools=ToolRegistry([]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
            )
            runtime = AgentRuntime(loop, state, interactive_approvals=True)

            runtime.switch_sandbox(DANGER_FULL_ACCESS)

            self.assertEqual(loop.sandbox_policy.mode, DANGER_FULL_ACCESS)
            state.pending_turn = PendingTurn(
                turn_id=f"{state.session_id}:1",
                task="pending",
                message_start=0,
            )
            with self.assertRaisesRegex(RuntimeError, "turn is pending"):
                runtime.switch_sandbox(WORKSPACE_WRITE)

    def test_context_manager_compacts_old_turns_and_preserves_recent_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            for index in range(6):
                state.messages.extend(
                    [
                        UserMessage(content=f"question-{index} " + "x" * 240),
                        AssistantMessage(content=f"answer-{index} " + "y" * 240),
                    ]
                )
            manager = ContextManager(
                context_window_tokens=900,
                reserve_output_tokens=200,
                compact_threshold=0.65,
                compact_target=0.5,
                summary_tokens=100,
            )

            snapshot = manager.prepare(
                state=state,
                tools=[ListFilesTool().spec()],
                system_prompt="You are a coding agent.",
            )

            self.assertGreater(state.summarized_message_count, 0)
            self.assertIsInstance(
                state.messages[state.summarized_message_count],
                UserMessage,
            )
            self.assertTrue(state.context_summary)
            self.assertEqual(
                snapshot.messages,
                state.messages[state.summarized_message_count :],
            )
            self.assertIn("<conversation_summary>", snapshot.system_prompt)
            self.assertLessEqual(snapshot.estimated_tokens, snapshot.input_budget)
            self.assertGreater(snapshot.compacted_messages, 0)
            self.assertEqual(snapshot.breakdown.total_tokens, snapshot.estimated_tokens)
            self.assertGreater(snapshot.breakdown.system_prompt_tokens, 0)
            self.assertGreater(snapshot.breakdown.tool_definition_tokens, 0)
            self.assertGreater(snapshot.breakdown.summary_tokens, 0)
            self.assertGreater(snapshot.breakdown.conversation_tokens, 0)
            self.assertEqual(
                snapshot.breakdown.system_prompt_tokens
                + snapshot.breakdown.summary_tokens,
                estimate_text_tokens(snapshot.system_prompt),
            )
            self.assertEqual(snapshot.context_window_tokens, 900)
            self.assertEqual(snapshot.reserve_output_tokens, 200)

            without_tools = manager.inspect(
                state=state,
                tools=[],
                system_prompt="You are a coding agent.",
            )
            self.assertEqual(without_tools.breakdown.tool_definition_tokens, 0)

    def test_cli_accepts_session_and_server_subcommands(self):
        parser = build_parser()
        ask = parser.parse_args(
            [
                "ask",
                "--no-stream",
                "--show-thinking",
                "--context-window-tokens",
                "200000",
                "--max-output-tokens",
                "16000",
                "--sandbox",
                "danger-full-access",
                "inspect",
                "this",
                "repo",
            ]
        )
        chat = parser.parse_args(["chat", "--workspace", "/tmp/repo"])
        resume = parser.parse_args(["resume", "session-123", "--workspace", "/tmp/repo"])
        serve = parser.parse_args(["serve", "--workspace", "/tmp/repo"])
        ui = parser.parse_args(
            ["ui", "--workspace", "/tmp/repo", "--port", "0", "--open-browser"]
        )

        self.assertEqual(ask.command, "ask")
        self.assertEqual(ask.prompt, ["inspect", "this", "repo"])
        self.assertFalse(ask.stream)
        self.assertTrue(ask.show_thinking)
        self.assertEqual(ask.context_window_tokens, 200000)
        self.assertEqual(ask.max_output_tokens, 16000)
        self.assertEqual(ask.sandbox, DANGER_FULL_ACCESS)
        self.assertIsNone(chat.sandbox)
        self.assertEqual(chat.command, "chat")
        self.assertEqual(resume.command, "resume")
        self.assertEqual(resume.session_id, "session-123")
        self.assertEqual(serve.command, "serve")
        self.assertFalse(hasattr(serve, "session_id"))
        self.assertFalse(hasattr(serve, "stream"))
        self.assertFalse(hasattr(serve, "show_thinking"))
        self.assertEqual(ui.command, "ui")
        self.assertEqual(ui.port, 0)
        self.assertTrue(ui.open_browser)
        self.assertFalse(hasattr(ui, "stream"))
        self.assertFalse(hasattr(ui, "show_thinking"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--workspace", "/tmp/repo", "inspect", "this", "repo"])

    def test_slash_command_completer_only_matches_command_prefixes(self):
        completer = SlashCommandCompleter(
            [
                SlashCommand("/context", "Show context usage"),
                SlashCommand("/retry", "Retry pending turn"),
            ]
        )

        matches = list(
            completer.get_completions(
                Document(text="/co", cursor_position=3),
                CompleteEvent(completion_requested=True),
            )
        )
        all_matches = list(
            completer.get_completions(
                Document(text="/", cursor_position=1),
                CompleteEvent(completion_requested=True),
            )
        )
        prose_matches = list(
            completer.get_completions(
                Document(text="explain /co", cursor_position=11),
                CompleteEvent(completion_requested=True),
            )
        )
        exact_matches = list(
            completer.get_completions(
                Document(text="/context", cursor_position=8),
                CompleteEvent(completion_requested=True),
            )
        )

        self.assertEqual([match.text for match in matches], ["/context"])
        self.assertEqual(matches[0].display_meta_text, "Show context usage")
        self.assertEqual(
            [match.text for match in all_matches],
            ["/context", "/retry"],
        )
        self.assertEqual(prose_matches, [])
        self.assertEqual(exact_matches, [])

    def test_selection_completer_puts_current_value_first(self):
        completer = SelectionCompleter(
            [
                SelectionOption("deepseek-v4-flash", "DeepSeek V4 Flash"),
                SelectionOption("deepseek-v4-pro", "DeepSeek V4 Pro"),
            ],
            current_value="deepseek-v4-pro",
        )

        completions = list(
            completer.get_completions(
                Document(),
                CompleteEvent(completion_requested=True),
            )
        )

        self.assertEqual(
            [completion.text for completion in completions],
            ["deepseek-v4-pro", "deepseek-v4-flash"],
        )
        self.assertEqual(
            completions[0].display_meta_text,
            "current | DeepSeek V4 Pro",
        )

    def test_tab_accepts_completion_then_submits_complete_slash_command(self):
        bindings = _completion_key_bindings(
            [SlashCommand("/context", "Show context usage")]
        )
        tab_handler = next(
            binding.handler
            for binding in bindings.bindings
            if binding.keys == (Keys.ControlI,)
        )
        accepted = []
        buffer = Buffer(
            document=Document(text="/co", cursor_position=3),
            accept_handler=lambda current: accepted.append(current.text),
            multiline=False,
        )
        buffer.complete_state = CompletionState(
            buffer.document,
            list(
                SlashCommandCompleter(
                    [SlashCommand("/context", "Show context usage")]
                ).get_completions(
                    buffer.document,
                    CompleteEvent(completion_requested=True),
                )
            ),
        )
        event = SimpleNamespace(app=SimpleNamespace(current_buffer=buffer))

        tab_handler(event)

        self.assertEqual(buffer.text, "/context")
        self.assertIsNone(buffer.complete_state)
        self.assertEqual(accepted, [])

        tab_handler(event)

        self.assertEqual(accepted, ["/context"])
        self.assertEqual(buffer.text, "")

    def test_slash_menu_uses_arrows_and_enter_to_run_command(self):
        commands = [
            SlashCommand("/context", "Show context usage"),
            SlashCommand("/model", "Choose the model"),
        ]
        bindings = _completion_key_bindings(commands)
        up_handler = next(
            binding.handler
            for binding in bindings.bindings
            if binding.keys == (Keys.Up,)
        )
        down_handler = next(
            binding.handler
            for binding in bindings.bindings
            if binding.keys == (Keys.Down,)
        )
        enter_handler = next(
            binding.handler
            for binding in bindings.bindings
            if binding.keys == (Keys.ControlM,)
        )
        accepted = []
        buffer = Buffer(
            document=Document(text="/", cursor_position=1),
            accept_handler=lambda current: accepted.append(current.text),
            multiline=False,
        )
        buffer.complete_state = CompletionState(
            buffer.document,
            list(
                SlashCommandCompleter(commands).get_completions(
                    buffer.document,
                    CompleteEvent(completion_requested=True),
                )
            ),
        )
        event = SimpleNamespace(app=SimpleNamespace(current_buffer=buffer))

        down_handler(event)
        self.assertEqual(buffer.complete_state.current_completion.text, "/context")
        down_handler(event)
        self.assertEqual(buffer.complete_state.current_completion.text, "/model")
        up_handler(event)
        self.assertEqual(buffer.complete_state.current_completion.text, "/context")
        down_handler(event)
        enter_handler(event)

        self.assertEqual(accepted, ["/model"])
        self.assertEqual(buffer.text, "")

    def test_tab_does_not_submit_regular_text(self):
        bindings = _completion_key_bindings(
            [SlashCommand("/context", "Show context usage")]
        )
        tab_handler = next(
            binding.handler
            for binding in bindings.bindings
            if binding.keys == (Keys.ControlI,)
        )
        accepted = []
        buffer = Buffer(
            document=Document(text="inspect this repo", cursor_position=17),
            accept_handler=lambda current: accepted.append(current.text),
            multiline=False,
        )
        event = SimpleNamespace(app=SimpleNamespace(current_buffer=buffer))

        tab_handler(event)

        self.assertEqual(buffer.text, "inspect this repo")
        self.assertEqual(accepted, [])

    def test_model_selector_uses_arrows_enter_and_escape(self):
        bindings = _selection_key_bindings(
            ["deepseek-v4-flash", "deepseek-v4-pro"]
        )
        down_handler = next(
            binding.handler
            for binding in bindings.bindings
            if binding.keys == (Keys.Down,)
        )
        enter_handler = next(
            binding.handler
            for binding in bindings.bindings
            if binding.keys == (Keys.ControlM,)
        )
        escape_handler = next(
            binding.handler
            for binding in bindings.bindings
            if binding.keys == (Keys.Escape,)
        )
        buffer = Buffer(
            document=Document(),
            multiline=False,
        )
        buffer.complete_state = CompletionState(
            buffer.document,
            [
                Completion("deepseek-v4-flash"),
                Completion("deepseek-v4-pro"),
            ],
            complete_index=0,
        )
        app = SimpleNamespace(
            current_buffer=buffer,
            result="unset",
        )
        app.exit = lambda *, result=None: setattr(app, "result", result)
        event = SimpleNamespace(app=app)

        down_handler(event)
        enter_handler(event)

        self.assertEqual(app.result, "deepseek-v4-pro")

        app.result = "unset"
        escape_handler(event)
        self.assertIsNone(app.result)

    def test_line_editor_passes_status_toolbar_and_secures_history(self):
        class FakePromptSession:
            def __init__(self):
                self.calls = []

            def prompt(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                return "edited"

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir, "state", "history")
            history_path.parent.mkdir()
            history_path.write_text("old command\n", encoding="utf-8")
            os.chmod(history_path, 0o644)
            fake_session = FakePromptSession()
            status_provider = lambda: [("class:toolbar", "context 1%")]
            editor = LineEditor(
                history_path=history_path,
                commands=[SlashCommand("/context", "Show context usage")],
                status_provider=status_provider,
                prompt_session=fake_session,
            )

            with editor:
                value = editor.read("you> ")

            self.assertEqual(value, "edited")
            self.assertEqual(fake_session.calls[0][0], "you> ")
            self.assertIs(fake_session.calls[0][1]["bottom_toolbar"], status_provider)
            self.assertEqual(history_path.stat().st_mode & 0o777, 0o600)

    def test_line_editor_enables_live_menu_with_explicit_terminal_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = DummyOutput()
            with redirect_stderr(io.StringIO()), patch.dict(
                os.environ, {"TERM": "dumb"}
            ), patch(
                "my_agent.terminal._interactive_output",
                return_value=output,
            ):
                editor = LineEditor(
                    history_path=Path(tmpdir, "history"),
                    commands=[SlashCommand("/context", "Show context usage")],
                )

            self.assertIs(editor.session._output, output)
            self.assertTrue(editor.session.default_buffer.complete_while_typing())

    def test_line_editor_opens_interactive_selection_menu(self):
        class FakePromptSession:
            def prompt(self, prompt, **kwargs):
                return "input"

        class FakeSelectionSession:
            def __init__(self):
                self.calls = []

            def prompt(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                return "deepseek-v4-pro"

        with tempfile.TemporaryDirectory() as tmpdir:
            selection_session = FakeSelectionSession()
            editor = LineEditor(
                history_path=Path(tmpdir, "history"),
                commands=[SlashCommand("/model", "Choose the model")],
                prompt_session=FakePromptSession(),
                selection_session=selection_session,
            )

            selected = editor.select(
                "model> ",
                [
                    SelectionOption("deepseek-v4-flash", "DeepSeek V4 Flash"),
                    SelectionOption("deepseek-v4-pro", "DeepSeek V4 Pro"),
                ],
                current_value="deepseek-v4-flash",
            )

        self.assertEqual(selected, "deepseek-v4-pro")
        self.assertEqual(selection_session.calls[0][0], "model> ")
        self.assertIsInstance(
            selection_session.calls[0][1]["completer"],
            SelectionCompleter,
        )
        self.assertTrue(callable(selection_session.calls[0][1]["pre_run"]))

    def test_context_usage_panel_and_toolbar_show_real_breakdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            state.messages.append(UserMessage(content="Inspect app.py"))
            manager = ContextManager(
                context_window_tokens=1_000,
                reserve_output_tokens=200,
            )
            snapshot = manager.inspect(
                state=state,
                tools=[ListFilesTool().spec()],
                system_prompt="You are a coding agent.",
            )

            panel = render_context_usage(
                snapshot,
                last_provider_input=123,
                width=24,
                color=False,
            )
            toolbar = "".join(
                text
                for _, text in format_context_status(
                    snapshot,
                    provider_id="deepseek",
                    model_id="deepseek-v4-flash",
                    mode_id="plan",
                    sandbox_mode=WORKSPACE_WRITE,
                )
            )

            self.assertIn("Context Usage", panel)
            self.assertIn("System prompt", panel)
            self.assertIn("Tool definitions", panel)
            self.assertIn("Conversation summary", panel)
            self.assertIn("Conversation", panel)
            self.assertIn("Protocol overhead", panel)
            self.assertIn("Last provider input", panel)
            self.assertIn("model deepseek/deepseek-v4-flash", toolbar)
            self.assertIn("mode plan", toolbar)
            self.assertIn("sandbox workspace-write", toolbar)
            self.assertIn("context", toolbar)
            self.assertIn("/ commands", toolbar)
            self.assertIn("Up/Down select", toolbar)
            self.assertIn("Enter run", toolbar)

    def test_chat_help_lists_itself(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "cli.py",
                    "chat",
                    "--workspace",
                    tmpdir,
                    "--sessions-dir",
                    str(Path(tmpdir, "sessions")),
                    "--config",
                    str(Path(tmpdir, "config.json")),
                    "--credentials",
                    str(Path(tmpdir, "credentials.json")),
                ],
                input="/goal\n/grants\n/sandbox\n/help\n/exit\n",
                text=True,
                capture_output=True,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Current sandbox: workspace-write", completed.stdout)
        self.assertIn("No current goal", completed.stdout)
        self.assertIn("No runtime permission grants", completed.stdout)
        self.assertIn("/help       Show slash commands", completed.stdout)
        self.assertIn("/grants", completed.stdout)
        self.assertIn("/goal", completed.stdout)
        self.assertIn("/sandbox", completed.stdout)
        self.assertIn("use Up/Down and Enter to run", completed.stdout)

    def test_chat_config_shows_paths_without_credentials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = Path(tmpdir, "credentials.json")
            credentials_path.write_text(
                json.dumps(
                    {"openai": {"type": "api", "key": "must-not-be-printed"}}
                ),
                encoding="utf-8",
            )
            os.chmod(credentials_path, 0o600)
            completed = subprocess.run(
                [
                    sys.executable,
                    "cli.py",
                    "chat",
                    "--workspace",
                    tmpdir,
                    "--sessions-dir",
                    str(Path(tmpdir, "sessions")),
                    "--config",
                    str(Path(tmpdir, "config.json")),
                    "--credentials",
                    str(credentials_path),
                ],
                input="/config\n/exit\n",
                text=True,
                capture_output=True,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Provider Configuration", completed.stdout)
        self.assertIn(str(credentials_path), completed.stdout)
        self.assertIn("deepseek, openai", completed.stdout)
        self.assertNotIn("must-not-be-printed", completed.stdout)

    def test_chat_switches_model_and_persists_it_in_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir, "sessions")
            class FakeLineEditor:
                supports_interactive_selection = True

                def __init__(self):
                    self.inputs = iter(["/model", "/summary", "/exit"])
                    self.selections = []

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return None

                def read(self, prompt):
                    return next(self.inputs)

                def select(self, prompt, options, *, current_value=None):
                    self.selections.append((prompt, tuple(options), current_value))
                    return "openai/gpt-5.6"

            editor = FakeLineEditor()
            output = io.StringIO()
            with patch("my_agent.cli.LineEditor", return_value=editor), redirect_stdout(output):
                return_code = main(
                    [
                        "chat",
                        "--workspace",
                        tmpdir,
                        "--sessions-dir",
                        str(sessions_dir),
                        "--config",
                        str(Path(tmpdir, "config.json")),
                        "--credentials",
                        str(Path(tmpdir, "credentials.json")),
                    ]
                )

            self.assertEqual(return_code, 0)
            self.assertIn("model: openai/gpt-5.6", output.getvalue())
            self.assertIn("model=openai/gpt-5.6", output.getvalue())
            self.assertEqual(editor.selections[0][0], "model> ")
            self.assertEqual(
                editor.selections[0][2],
                "deepseek/deepseek-v4-flash",
            )
            self.assertEqual(
                [option.value for option in editor.selections[0][1]],
                [
                    "deepseek/deepseek-v4-flash",
                    "deepseek/deepseek-v4-pro",
                    "openai/gpt-5.6",
                ],
            )
            session_id = next(path.name for path in sessions_dir.iterdir() if path.is_dir())
            state = SessionStore(sessions_dir).load(session_id)
            self.assertEqual(state.provider_id, "openai")
            self.assertEqual(state.model_id, "gpt-5.6")

            resumed = subprocess.run(
                [
                    sys.executable,
                    "cli.py",
                    "resume",
                    session_id,
                    "--workspace",
                    tmpdir,
                    "--sessions-dir",
                    str(sessions_dir),
                    "--config",
                    str(Path(tmpdir, "config.json")),
                    "--credentials",
                    str(Path(tmpdir, "credentials.json")),
                ],
                input="/summary\n/exit\n",
                text=True,
                capture_output=True,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
            )

            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertIn("model=openai/gpt-5.6", resumed.stdout)

    def test_chat_switches_mode_and_persists_plan_tool_boundary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir, "sessions")

            class FakeLineEditor:
                supports_interactive_selection = True

                def __init__(self):
                    self.inputs = iter(["/mode", "/summary", "/tools", "/exit"])
                    self.selections = []

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return None

                def read(self, prompt):
                    return next(self.inputs)

                def select(self, prompt, options, *, current_value=None):
                    self.selections.append((prompt, tuple(options), current_value))
                    return "plan"

            editor = FakeLineEditor()
            output = io.StringIO()
            with patch("my_agent.cli.LineEditor", return_value=editor), redirect_stdout(output):
                return_code = main(
                    [
                        "chat",
                        "--workspace",
                        tmpdir,
                        "--sessions-dir",
                        str(sessions_dir),
                        "--config",
                        str(Path(tmpdir, "config.json")),
                        "--credentials",
                        str(Path(tmpdir, "credentials.json")),
                    ]
                )

            self.assertEqual(return_code, 0)
            self.assertIn("mode: plan", output.getvalue())
            self.assertIn("mode=plan", output.getvalue())
            self.assertEqual(editor.selections[0][0], "mode> ")
            self.assertEqual(editor.selections[0][2], "build")
            self.assertEqual(
                [option.value for option in editor.selections[0][1]],
                ["build", "plan"],
            )
            session_id = next(path.name for path in sessions_dir.iterdir() if path.is_dir())
            state = SessionStore(sessions_dir).load(session_id)
            self.assertEqual(state.mode_id, "plan")
            tool_lines = {
                line
                for line in output.getvalue().splitlines()
                if line in build_tool_registry(BUILD_MODE).names()
            }
            self.assertEqual(
                tool_lines,
                {"git_diff", "list_files", "read_file", "search"},
            )

    def test_chat_offers_plan_action_and_executes_selected_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir, "sessions")
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        content=(
                            "I prepared the implementation plan.\n"
                            "<proposed_plan>\n"
                            "# Plan\n\n"
                            "1. Inspect the target.\n"
                            "2. Implement and validate.\n"
                            "</proposed_plan>\n\n"
                            "Tell me if you want changes; otherwise I can execute it."
                        )
                    ),
                    _model_response(content="Implemented and validated."),
                ]
            )

            def fake_build_runtime(args, workspace, trace_path):
                mode = get_mode(args.mode)
                registry = build_tool_registry(mode)
                return (
                    AgentLoop(
                        workspace=workspace,
                        model=_model(provider),
                        tools=registry,
                        permissions=PermissionPolicy(),
                        trace=TraceRecorder(trace_path),
                        max_steps=args.max_steps,
                        mode=mode,
                    ),
                    registry,
                )

            class FakeLineEditor:
                supports_interactive_selection = True

                def __init__(self):
                    self.inputs = iter(["Plan this change", "/exit"])
                    self.selections = []

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return None

                def read(self, prompt):
                    return next(self.inputs)

                def select(self, prompt, options, *, current_value=None):
                    self.selections.append((prompt, tuple(options), current_value))
                    return "Execute plan"

            editor = FakeLineEditor()
            output = io.StringIO()
            status_output = io.StringIO()
            with patch("my_agent.cli.LineEditor", return_value=editor), patch(
                "my_agent.cli._build_runtime", side_effect=fake_build_runtime
            ), patch(
                "my_agent.cli.StreamRenderer",
                side_effect=lambda **kwargs: StreamRenderer(
                    output=output,
                    status_output=status_output,
                    **kwargs,
                ),
            ), redirect_stdout(output):
                return_code = main(
                    [
                        "chat",
                        "--mode",
                        "plan",
                        "--workspace",
                        tmpdir,
                        "--sessions-dir",
                        str(sessions_dir),
                        "--config",
                        str(Path(tmpdir, "config.json")),
                        "--credentials",
                        str(Path(tmpdir, "credentials.json")),
                    ]
                )

            self.assertEqual(return_code, 0)
            self.assertIn("# Plan", output.getvalue())
            self.assertIn("plan approved:", output.getvalue())
            self.assertIn("assistant> Implemented and validated.", output.getvalue())
            self.assertNotIn("Tell me if you want changes", output.getvalue())
            self.assertEqual(editor.selections[0][0], "plan> ")
            self.assertEqual(editor.selections[0][2], "Execute plan")
            self.assertEqual(
                [option.value for option in editor.selections[0][1]],
                ["Execute plan", "Revise plan"],
            )
            self.assertEqual(len(provider.requests), 2)
            session_id = next(path.name for path in sessions_dir.iterdir() if path.is_dir())
            state = SessionStore(sessions_dir).load(session_id)
            self.assertEqual(state.mode_id, "build")
            self.assertEqual(state.proposed_plan.status, "completed")

    def test_pending_turn_requires_its_persisted_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                ["resume", "session-id", "--mode", "build"]
            )
            state = SessionStore(Path(tmpdir, "sessions")).create(tmpdir)
            state.mode_id = "plan"
            state.pending_turn = PendingTurn(
                turn_id=f"{state.session_id}:1",
                task="Plan this change",
                message_start=0,
                mode_id="plan",
            )

            with self.assertRaisesRegex(ValueError, "pending turn requires mode plan"):
                _restore_session_mode(args, state)

    def test_pending_turn_requires_its_persisted_sandbox(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                ["resume", "session-id", "--sandbox", DANGER_FULL_ACCESS]
            )
            state = SessionStore(Path(tmpdir, "sessions")).create(tmpdir)
            state.pending_turn = PendingTurn(
                turn_id=f"{state.session_id}:1",
                task="Continue this change",
                message_start=0,
            )

            with self.assertRaisesRegex(
                ValueError,
                "pending turn requires sandbox workspace-write",
            ):
                _restore_session_sandbox(args, state)

    def test_resume_full_model_reference_overrides_session_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                ["resume", "session-id", "--model", "openai/gpt-5.6"]
            )
            args.runtime_config = load_runtime_config(
                Path(tmpdir, "config.json"),
                Path(tmpdir, "credentials.json"),
            )
            state = SessionStore(Path(tmpdir, "sessions")).create(tmpdir)
            state.provider_id = "deepseek"
            state.model_id = "deepseek-v4-flash"

            _restore_session_model(args, state)
            model = _build_model(args)

        self.assertEqual(args.provider, None)
        self.assertEqual(model.profile.provider_id, "openai")
        self.assertEqual(model.profile.model_id, "gpt-5.6")

    def test_stream_renderer_displays_reasoning_content_and_tool_progress(self):
        output = io.StringIO()
        status = io.StringIO()
        renderer = StreamRenderer(
            output=output,
            status_output=status,
            answer_prefix="assistant> ",
            show_thinking=True,
        )

        event = {
            "session_id": "session-1",
            "turn_id": "session-1:1",
            "sequence": 1,
            "state": AgentState.RUNNING_MODEL,
            "step": 1,
        }
        renderer.handle(ModelStartedEvent(**event))
        renderer.handle(
            ItemDeltaEvent(
                **event,
                durable=False,
                item_id="reasoning-1",
                item_type="reasoning",
                delta="Inspecting.",
            )
        )
        renderer.handle(
            ItemDeltaEvent(
                **event,
                durable=False,
                item_id="assistant-1",
                item_type="assistant_message",
                delta="I found ",
            )
        )
        renderer.handle(
            ItemDeltaEvent(
                **event,
                durable=False,
                item_id="assistant-1",
                item_type="assistant_message",
                delta="app.py.",
            )
        )
        renderer.handle(
            ItemStartedEvent(
                **event,
                item_id="tool-1",
                item_type="tool_call",
                data={"name": "read_file", "arguments": {"path": "app.py"}},
            )
        )
        renderer.handle(
            ItemCompletedEvent(
                **event,
                item_id="tool-1",
                item_type="tool_call",
                data={"name": "read_file", "ok": True, "status": "success"},
            )
        )
        renderer.finish("I found app.py.")

        self.assertIn("thinking> Inspecting.", output.getvalue())
        self.assertIn("assistant> I found app.py.", output.getvalue())
        self.assertIn("tool> read_file", status.getvalue())
        self.assertIn("tool< read_file success", status.getvalue())

    def test_stream_renderer_streams_plan_content_and_hides_fragmented_protocol(self):
        output = io.StringIO()
        renderer = StreamRenderer(
            output=output,
            status_output=io.StringIO(),
            answer_prefix="assistant> ",
            hide_plan_protocol=True,
        )
        plan = "# Plan\n\nUpdate app.py and run focused tests."

        event = {
            "session_id": "session-1",
            "turn_id": "session-1:1",
            "sequence": 1,
            "state": AgentState.RUNNING_MODEL,
            "step": 1,
        }
        renderer.handle(ModelStartedEvent(**event))
        renderer.handle(
            ItemDeltaEvent(
                **event,
                durable=False,
                item_id="assistant-1",
                item_type="assistant_message",
                delta="Here is the plan.\n<pro",
            )
        )
        self.assertEqual(output.getvalue(), "assistant> Here is the plan.\n")
        renderer.handle(
            ItemDeltaEvent(
                **event,
                durable=False,
                item_id="assistant-1",
                item_type="assistant_message",
                delta="posed_plan>\n# Plan\n\nUpdate app.py and run focused tests.",
            )
        )
        self.assertIn("# Plan", output.getvalue())
        self.assertNotIn("proposed_plan", output.getvalue())
        renderer.handle(
            ItemDeltaEvent(
                **event,
                durable=False,
                item_id="assistant-1",
                item_type="assistant_message",
                delta="</proposed",
            )
        )
        renderer.handle(
            ItemDeltaEvent(
                **event,
                durable=False,
                item_id="assistant-1",
                item_type="assistant_message",
                delta="_plan>\nConfirm and I will execute it.",
            )
        )
        renderer.finish(plan)

        self.assertEqual(output.getvalue(), f"assistant> Here is the plan.\n{plan}\n")

    def test_stream_renderer_streams_ordinary_plan_reply_without_waiting(self):
        output = io.StringIO()
        renderer = StreamRenderer(
            output=output,
            status_output=io.StringIO(),
            answer_prefix="assistant> ",
            hide_plan_protocol=True,
        )

        event = {
            "session_id": "session-1",
            "turn_id": "session-1:1",
            "sequence": 1,
            "state": AgentState.RUNNING_MODEL,
            "step": 1,
        }
        renderer.handle(ModelStartedEvent(**event))
        renderer.handle(
            ItemDeltaEvent(
                **event,
                durable=False,
                item_id="assistant-1",
                item_type="assistant_message",
                delta="Which file should change?",
            )
        )

        self.assertEqual(output.getvalue(), "assistant> Which file should change?")
        renderer.finish("Which file should change?")
        self.assertEqual(output.getvalue(), "assistant> Which file should change?\n")

    def test_default_tool_registry_exposes_current_tools(self):
        self.assertEqual(
            build_tool_registry().names(),
            [
                "apply_patch",
                "exec_command",
                "git_diff",
                "list_files",
                "read_file",
                "search",
                "update_goal",
            ],
        )

    def test_plan_mode_exposes_only_read_only_tools_and_extracts_wrapped_protocol(self):
        self.assertEqual(
            build_tool_registry(PLAN_MODE).names(),
            ["git_diff", "list_files", "read_file", "search"],
        )
        self.assertEqual(get_mode("plan"), PLAN_MODE)
        self.assertEqual(
            extract_proposed_plan("<proposed_plan>\n# Plan\n</proposed_plan>"),
            "# Plan",
        )
        self.assertEqual(
            extract_proposed_plan(
                "Here is the plan:\n<proposed_plan>a plan</proposed_plan>\nConfirm?"
            ),
            "a plan",
        )
        self.assertIsNone(extract_proposed_plan("<proposed_plan>missing end tag"))
        with self.assertRaisesRegex(ValueError, "unknown collaboration mode"):
            get_mode("review")

    def test_update_goal_tool_persists_state_and_updates_next_model_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        tool_calls=[
                            ToolCall(
                                call_id="call_goal",
                                name="update_goal",
                                arguments={
                                    "objective": "Inspect and verify the repository",
                                    "steps": [
                                        {
                                            "description": "Inspect files",
                                            "status": "completed",
                                        },
                                        {
                                            "description": "Run validation",
                                            "status": "in_progress",
                                        },
                                    ],
                                },
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _model_response(content="Goal recorded."),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=build_tool_registry(BUILD_MODE),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=3,
            )
            events = []

            result = loop.run_turn(
                state,
                "Inspect and verify the repository",
                event_handler=events.append,
            )

            self.assertEqual(result.answer, "Goal recorded.")
            self.assertIsNotNone(state.goal)
            self.assertEqual(state.goal.objective, "Inspect and verify the repository")
            self.assertEqual(state.goal.steps[1].status, "in_progress")
            self.assertTrue(any(event.kind == "goal_updated" for event in events))
            self.assertNotIn("<current_goal>", provider.requests[0].system_prompt)
            self.assertIn("<current_goal>", provider.requests[1].system_prompt)
            self.assertIn("Run validation", provider.requests[1].system_prompt)
            trajectory = make_trajectory(
                read_jsonl_trace(state.trace_path),
                source_path=state.trace_path,
            )
            self.assertEqual(
                trajectory["turns"][0]["goal_events"][0]["event"],
                "goal_updated",
            )
            self.assertEqual(trajectory["metrics"]["goal_updates"], 1)

    def test_goal_rejects_multiple_in_progress_steps(self):
        with self.assertRaisesRegex(ValueError, "more than one in_progress"):
            update_goal(
                None,
                {
                    "objective": "Invalid goal",
                    "steps": [
                        {"description": "First", "status": "in_progress"},
                        {"description": "Second", "status": "in_progress"},
                    ],
                },
            )

    def test_goal_identity_changes_only_for_a_different_objective(self):
        initial = update_goal(
            None,
            {
                "objective": "Update app.py",
                "steps": [{"description": "Inspect", "status": "in_progress"}],
            },
        )
        progressed = update_goal(
            initial,
            {
                "objective": "Update app.py",
                "steps": [{"description": "Inspect", "status": "completed"}],
            },
        )
        replacement = update_goal(
            progressed,
            {
                "objective": "Update tests.py",
                "steps": [{"description": "Inspect tests", "status": "pending"}],
            },
        )

        self.assertEqual(progressed.goal_id, initial.goal_id)
        self.assertNotEqual(replacement.goal_id, initial.goal_id)

    def test_agent_loop_exposes_only_session_turn_entrypoint(self):
        self.assertFalse(hasattr(AgentLoop, "run"))

    def test_read_file_rejects_paths_outside_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Workspace(tmpdir)
            outside = Path(tmpdir).parent / "outside-secret.txt"
            outside.write_text("secret", encoding="utf-8")
            ctx = ToolContext(workspace=workspace, permissions=PermissionPolicy())

            result = ReadFileTool().run({"path": str(outside)}, ctx)

            self.assertFalse(result.ok)
            self.assertIn("outside workspace", result.stderr)

    def test_exec_command_allows_validation_commands_and_rejects_arbitrary_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test_sample.py").write_text(
                "import unittest\n\n"
                "class SampleTest(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertEqual(1 + 1, 2)\n",
                encoding="utf-8",
            )
            ctx = ToolContext(
                workspace=Workspace(tmpdir),
                permissions=PermissionPolicy(),
                timeout_seconds=10,
                sandbox_policy=SandboxPolicy(DANGER_FULL_ACCESS, Path(tmpdir)),
            )

            passing = ExecCommandTool().run(
                {"argv": ["python3", "-m", "unittest", "discover", "-s", "."]}, ctx
            )
            denied = ExecCommandTool().run(
                {"argv": ["python3", "-c", "print('unsafe')"]}, ctx
            )

            self.assertTrue(passing.ok, passing.stderr)
            self.assertEqual(passing.exit_code, 0)
            self.assertEqual(passing.metadata["status"], "success")
            self.assertEqual(passing.metadata["category"], "test")
            self.assertFalse(denied.ok)
            self.assertIn("not allowed", denied.stderr)
            self.assertEqual(denied.metadata["status"], "denied")

    def test_exec_command_policy_categorizes_supported_checks_and_builds(self):
        policy = PermissionPolicy()

        cases = [
            (["ruff", "check", "."], "check"),
            (["python3", "-m", "black", "--check", "."], "check"),
            (["npm", "run", "lint"], "check"),
            (["cargo", "check"], "check"),
            (["go", "build", "./..."], "build"),
        ]

        for argv, category in cases:
            with self.subTest(argv=argv):
                decision = policy.decide_command(argv)
                self.assertTrue(decision.allowed, decision.reason)
                self.assertEqual(decision.category, category)

        for argv in (["rm", "-rf", "."], ["bash", "-lc", "pytest"], ["./check.sh"]):
            with self.subTest(argv=argv):
                self.assertFalse(policy.decide_command(argv).allowed)

    def test_permission_profiles_control_approval_without_bypassing_hard_guards(self):
        supervised = PermissionPolicy("supervised")
        accept_edits = PermissionPolicy("accept_edits")
        trusted = PermissionPolicy("trusted")

        self.assertEqual(
            supervised.decide_permission("apply_patch", "app.py"),
            PermissionEffect.ASK,
        )
        self.assertEqual(
            accept_edits.decide_permission("apply_patch", "app.py"),
            PermissionEffect.ALLOW,
        )
        self.assertEqual(
            accept_edits.decide_permission("exec_command", "pytest -q"),
            PermissionEffect.ASK,
        )
        self.assertEqual(
            trusted.decide_permission("exec_command", "pytest -q"),
            PermissionEffect.ALLOW,
        )
        self.assertFalse(trusted.allow_file_change("update", ".git/config")[0])
        self.assertFalse(trusted.decide_command(["rm", "-rf", "."]).allowed)

    def test_non_interactive_approval_broker_denies_ask_actions(self):
        outcome = DenyApprovalBroker().authorize(
            PermissionRequest(
                action="exec_command",
                resource="pytest -q",
                description="run pytest -q",
            ),
            cancel_event=None,
            on_waiting=lambda: self.fail("non-interactive broker must not prompt"),
        )

        self.assertEqual(outcome.decision, ApprovalDecision.DENY)
        self.assertEqual(outcome.source, "unavailable")

    def test_exec_command_approval_describes_sandbox_boundary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Workspace(tmpdir)
            request = ExecCommandTool().permission_request(
                {"argv": ["pytest", "-q"]},
                ToolContext(
                    workspace=workspace,
                    permissions=PermissionPolicy(),
                    sandbox_policy=SandboxPolicy(WORKSPACE_WRITE, workspace.root),
                ),
            )

        self.assertIsNotNone(request)
        self.assertEqual(request.details["sandbox"]["mode"], WORKSPACE_WRITE)
        self.assertEqual(request.details["sandbox"]["network"], "denied")
        self.assertFalse(request.details["sandbox"]["dangerous"])

    def test_exec_command_validates_argv_and_workspace_relative_cwd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir, "workspace")
            nested = root / "nested"
            nested.mkdir(parents=True)
            (nested / "test_nested.py").write_text(
                "import unittest\n\n"
                "class NestedTest(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            outside = Path(tmpdir, "outside")
            outside.mkdir()
            (root / "outside-link").symlink_to(outside, target_is_directory=True)
            ctx = ToolContext(
                workspace=Workspace(root),
                permissions=PermissionPolicy(),
                sandbox_policy=SandboxPolicy(DANGER_FULL_ACCESS, root),
            )
            tool = ExecCommandTool()

            nested_result = tool.run(
                {
                    "argv": ["python3", "-m", "unittest", "discover", "-s", "."],
                    "cwd": "nested",
                },
                ctx,
            )
            invalid_argv = tool.run({"argv": "python3 -m unittest"}, ctx)
            parent_escape = tool.run(
                {"argv": ["python3", "-m", "unittest"], "cwd": "../outside"}, ctx
            )
            symlink_escape = tool.run(
                {"argv": ["python3", "-m", "unittest"], "cwd": "outside-link"}, ctx
            )

            self.assertTrue(nested_result.ok, nested_result.stderr)
            self.assertEqual(nested_result.metadata["cwd"], "nested")
            self.assertEqual(
                nested_result.metadata["sandbox_mode"],
                DANGER_FULL_ACCESS,
            )
            self.assertEqual(nested_result.metadata["sandbox_backend"], "none")
            self.assertFalse(invalid_argv.ok)
            self.assertEqual(invalid_argv.metadata["status"], "invalid_request")
            self.assertFalse(parent_escape.ok)
            self.assertIn("workspace-relative", parent_escape.stderr)
            self.assertFalse(symlink_escape.ok)
            self.assertIn("outside workspace", symlink_escape.stderr)

    def test_exec_command_strips_provider_secrets_from_child_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test_environment.py").write_text(
                "import os\n"
                "import unittest\n\n"
                "class EnvironmentTest(unittest.TestCase):\n"
                "    def test_provider_key_is_absent(self):\n"
                "        self.assertNotIn('DEEPSEEK_API_KEY', os.environ)\n"
                "        self.assertNotIn('OPENAI_API_KEY', os.environ)\n",
                encoding="utf-8",
            )
            ctx = ToolContext(
                workspace=Workspace(tmpdir),
                permissions=PermissionPolicy(),
                sandbox_policy=SandboxPolicy(DANGER_FULL_ACCESS, Path(tmpdir)),
            )

            with patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "should-not-leak",
                    "OPENAI_API_KEY": "should-not-leak",
                },
            ):
                result = ExecCommandTool().run(
                    {"argv": ["python3", "-m", "unittest", "discover", "-s", "."]},
                    ctx,
                )

            self.assertTrue(result.ok, result.stderr)
            self.assertNotIn("DEEPSEEK_API_KEY", result.metadata["environment_keys"])
            self.assertNotIn("OPENAI_API_KEY", result.metadata["environment_keys"])

    def test_exec_command_reports_nonzero_exit_and_truncates_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test_failure.py").write_text(
                "import unittest\n\n"
                "class FailureTest(unittest.TestCase):\n"
                "    def test_failure(self):\n"
                "        print('x' * 13000)\n"
                "        self.fail('expected failure')\n",
                encoding="utf-8",
            )
            ctx = ToolContext(
                workspace=Workspace(tmpdir),
                permissions=PermissionPolicy(),
                sandbox_policy=SandboxPolicy(DANGER_FULL_ACCESS, Path(tmpdir)),
            )

            result = ExecCommandTool().run(
                {"argv": ["python3", "-m", "unittest", "discover", "-s", "."]}, ctx
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["status"], "nonzero_exit")
            self.assertNotEqual(result.exit_code, 0)
            self.assertTrue(result.truncated)
            self.assertIn("...[truncated]", result.stdout)

    def test_exec_command_reports_spawn_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = ToolContext(
                workspace=Workspace(tmpdir),
                permissions=PermissionPolicy(),
                sandbox_policy=SandboxPolicy(DANGER_FULL_ACCESS, Path(tmpdir)),
            )

            with patch.dict(os.environ, {"PATH": "/definitely-missing"}):
                result = ExecCommandTool().run({"argv": ["ruff", "check", "."]}, ctx)

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["status"], "spawn_error")
            self.assertIsNone(result.exit_code)

    def test_exec_command_timeout_kills_the_process_group(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir, "child-survived.txt")
            Path(tmpdir, "test_timeout.py").write_text(
                "import subprocess\n"
                "import sys\n"
                "import time\n"
                "import unittest\n\n"
                "class TimeoutTest(unittest.TestCase):\n"
                "    def test_timeout(self):\n"
                f"        code = \"import time; time.sleep(2); open({str(marker)!r}, 'w').write('alive')\"\n"
                "        subprocess.Popen([sys.executable, '-c', code])\n"
                "        time.sleep(30)\n",
                encoding="utf-8",
            )
            ctx = ToolContext(
                workspace=Workspace(tmpdir),
                permissions=PermissionPolicy(),
                timeout_seconds=5,
                sandbox_policy=SandboxPolicy(DANGER_FULL_ACCESS, Path(tmpdir)),
            )

            result = ExecCommandTool().run(
                {
                    "argv": ["python3", "-m", "unittest", "discover", "-s", "."],
                    "timeout_seconds": 1,
                },
                ctx,
            )
            time.sleep(2.5)

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["status"], "timed_out")
            self.assertTrue(result.metadata["timed_out"])
            self.assertFalse(marker.exists())

    def test_exec_command_interrupt_kills_the_running_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test_interrupt.py").write_text(
                "import time\n"
                "import unittest\n\n"
                "class InterruptTest(unittest.TestCase):\n"
                "    def test_wait(self):\n"
                "        time.sleep(30)\n",
                encoding="utf-8",
            )
            cancel_event = threading.Event()
            ctx = ToolContext(
                workspace=Workspace(tmpdir),
                permissions=PermissionPolicy(),
                timeout_seconds=10,
                cancel_event=cancel_event,
                sandbox_policy=SandboxPolicy(DANGER_FULL_ACCESS, Path(tmpdir)),
            )
            timer = threading.Timer(0.2, cancel_event.set)
            timer.start()
            started = time.monotonic()

            result = ExecCommandTool().run(
                {"argv": ["python3", "-m", "unittest", "discover", "-s", "."]},
                ctx,
            )
            timer.join()

            self.assertLess(time.monotonic() - started, 2)
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["status"], "interrupted")
            self.assertTrue(result.metadata["interrupted"])

    def test_git_diff_reports_tracked_and_untracked_changes_without_mutating_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tracked = root / "app.py"
            unstaged = root / "other.py"
            tracked.write_text("value = 1\n", encoding="utf-8")
            unstaged.write_text("other = 1\n", encoding="utf-8")
            _commit_workspace(root)
            tool = GitDiffTool()
            ctx = ToolContext(workspace=Workspace(root), permissions=PermissionPolicy())
            clean = tool.run({"path": "."}, ctx)
            tracked.write_text("value = 2\n", encoding="utf-8")
            _git(root, "add", "app.py")
            unstaged.write_text("other = 2\n", encoding="utf-8")
            (root / "new.py").write_text("new = True\n", encoding="utf-8")
            state_dir = root / ".my-agent" / "sessions" / "session-1"
            state_dir.mkdir(parents=True)
            (state_dir / "trace.jsonl").write_text("{}\n", encoding="utf-8")
            before = _git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout

            result = tool.run({"path": "."}, ctx)

            after = _git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout
            self.assertTrue(clean.ok, clean.stderr)
            self.assertEqual(clean.stdout, "No changes.")
            self.assertTrue(result.ok, result.stderr)
            self.assertIn("diff --git a/app.py b/app.py", result.stdout)
            self.assertIn("-value = 1", result.stdout)
            self.assertIn("+value = 2", result.stdout)
            self.assertIn("diff --git a/other.py b/other.py", result.stdout)
            self.assertIn("[untracked files]\nnew.py", result.stdout)
            self.assertEqual(result.metadata["untracked_files"], ["new.py"])
            self.assertEqual(result.exit_code, 0)
            self.assertNotIn(".my-agent", result.stdout)
            self.assertEqual(before, after)

    def test_git_diff_scopes_nested_workspace_to_its_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repository = Path(tmpdir)
            workspace = repository / "workspace"
            workspace.mkdir()
            inside = workspace / "inside.py"
            outside = repository / "outside.py"
            inside.write_text("inside = 1\n", encoding="utf-8")
            outside.write_text("outside = 1\n", encoding="utf-8")
            _commit_workspace(repository)
            inside.write_text("inside = 2\n", encoding="utf-8")
            outside.write_text("outside = 2\n", encoding="utf-8")
            (workspace / "inside-new.py").write_text("inside_new = True\n", encoding="utf-8")
            (repository / "outside-new.py").write_text("outside_new = True\n", encoding="utf-8")
            tool = GitDiffTool()
            ctx = ToolContext(workspace=Workspace(workspace), permissions=PermissionPolicy())

            result = tool.run({"path": "."}, ctx)

            self.assertTrue(result.ok, result.stderr)
            self.assertIn("inside.py", result.stdout)
            self.assertIn("inside-new.py", result.stdout)
            self.assertNotIn("outside.py", result.stdout)
            self.assertNotIn("outside-new.py", result.stdout)

    def test_git_diff_filters_paths_and_rejects_workspace_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir, "workspace")
            root.mkdir()
            (root / "a.py").write_text("a = 1\n", encoding="utf-8")
            (root / "b.py").write_text("b = 1\n", encoding="utf-8")
            _commit_workspace(root)
            (root / "a.py").write_text("a = 2\n", encoding="utf-8")
            (root / "b.py").write_text("b = 2\n", encoding="utf-8")
            tool = GitDiffTool()
            ctx = ToolContext(workspace=Workspace(root), permissions=PermissionPolicy())

            filtered = tool.run({"path": "a.py"}, ctx)
            parent_escape = tool.run({"path": "../outside.py"}, ctx)
            absolute = tool.run({"path": str(root / "a.py")}, ctx)

            self.assertTrue(filtered.ok, filtered.stderr)
            self.assertIn("a.py", filtered.stdout)
            self.assertNotIn("b.py", filtered.stdout)
            self.assertFalse(parent_escape.ok)
            self.assertIn("workspace-relative", parent_escape.stderr)
            self.assertFalse(absolute.ok)
            self.assertIn("workspace-relative", absolute.stderr)

    def test_git_diff_truncates_large_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "large.txt"
            target.write_text("old\n", encoding="utf-8")
            _commit_workspace(root)
            target.write_text("new line\n" * 2_000, encoding="utf-8")
            tool = GitDiffTool()
            ctx = ToolContext(workspace=Workspace(root), permissions=PermissionPolicy())

            result = tool.run({"path": "."}, ctx)

            self.assertTrue(result.ok, result.stderr)
            self.assertTrue(result.truncated)
            self.assertTrue(result.stdout.endswith("...[truncated]"))

    def test_git_diff_requires_repository_with_head(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tool = GitDiffTool()
            ctx = ToolContext(workspace=Workspace(root), permissions=PermissionPolicy())

            not_repository = tool.run({"path": "."}, ctx)
            _git(root, "init", "-q")
            without_head = tool.run({"path": "."}, ctx)

            self.assertFalse(not_repository.ok)
            self.assertIn("not a Git repository", not_repository.stderr)
            self.assertFalse(without_head.ok)
            self.assertIn("HEAD commit", without_head.stderr)

    def test_apply_patch_updates_adds_and_deletes_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Workspace(tmpdir)
            ctx = ToolContext(workspace=workspace, permissions=PermissionPolicy())
            tool = ApplyPatchTool()
            Path(tmpdir, "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
            Path(tmpdir, "obsolete.txt").write_text("remove me\n", encoding="utf-8")

            updated = tool.run(
                {
                    "patch": """*** Begin Patch
*** Update File: app.py
@@
 def value():
-    return 1
+    return 2
*** End Patch"""
                },
                ctx,
            )
            added = tool.run(
                {
                    "patch": """*** Begin Patch
*** Add File: nested/new.py
+print('new')
*** End Patch"""
                },
                ctx,
            )
            deleted = tool.run(
                {
                    "patch": """*** Begin Patch
*** Delete File: obsolete.txt
*** End Patch"""
                },
                ctx,
            )

            self.assertTrue(updated.ok, updated.stderr)
            self.assertTrue(added.ok, added.stderr)
            self.assertTrue(deleted.ok, deleted.stderr)
            self.assertEqual(
                Path(tmpdir, "app.py").read_text(encoding="utf-8"),
                "def value():\n    return 2\n",
            )
            self.assertEqual(
                Path(tmpdir, "nested", "new.py").read_text(encoding="utf-8"),
                "print('new')\n",
            )
            self.assertFalse(Path(tmpdir, "obsolete.txt").exists())
            self.assertEqual(updated.metadata["operation"], "update")
            self.assertEqual(updated.metadata["changed_files"], ["app.py"])

    def test_apply_patch_context_mismatch_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir, "app.py")
            target.write_text("value = 1\n", encoding="utf-8")
            tool = ApplyPatchTool()
            ctx = ToolContext(workspace=Workspace(tmpdir), permissions=PermissionPolicy())

            result = tool.run(
                {
                    "patch": """*** Begin Patch
*** Update File: app.py
@@
-value = 2
+value = 3
*** End Patch"""
                },
                ctx,
            )

            self.assertFalse(result.ok)
            self.assertIn("context did not match", result.stderr)
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")
            self.assertFalse(result.metadata["applied"])

    def test_apply_patch_rejects_ambiguous_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir, "app.py")
            target.write_text("value = 1\nvalue = 1\n", encoding="utf-8")
            tool = ApplyPatchTool()
            ctx = ToolContext(workspace=Workspace(tmpdir), permissions=PermissionPolicy())

            result = tool.run(
                {
                    "patch": """*** Begin Patch
*** Update File: app.py
@@
-value = 1
+value = 2
*** End Patch"""
                },
                ctx,
            )

            self.assertFalse(result.ok)
            self.assertIn("ambiguous", result.stderr)
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\nvalue = 1\n")

    def test_apply_patch_rejects_multiple_file_operations_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir, "app.py")
            target.write_text("value = 1\n", encoding="utf-8")
            tool = ApplyPatchTool()
            ctx = ToolContext(workspace=Workspace(tmpdir), permissions=PermissionPolicy())

            result = tool.run(
                {
                    "patch": """*** Begin Patch
*** Update File: app.py
@@
-value = 1
+value = 2
*** Add File: extra.py
+value = 3
*** End Patch"""
                },
                ctx,
            )

            self.assertFalse(result.ok)
            self.assertIn("exactly one file operation", result.stderr)
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")
            self.assertFalse(Path(tmpdir, "extra.py").exists())

    def test_apply_patch_rejects_oversized_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ApplyPatchTool()
            ctx = ToolContext(workspace=Workspace(tmpdir), permissions=PermissionPolicy())
            oversized_patch = (
                "*** Begin Patch\n*** Add File: large.txt\n+"
                + "x" * 100_001
                + "\n*** End Patch"
            )

            result = tool.run({"patch": oversized_patch}, ctx)

            self.assertFalse(result.ok)
            self.assertIn("exceeds", result.stderr)
            self.assertFalse(Path(tmpdir, "large.txt").exists())

    def test_apply_patch_rejects_protected_and_symlink_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir, "workspace")
            root.mkdir()
            git_dir = root / ".git"
            git_dir.mkdir()
            git_config = git_dir / "config"
            git_config.write_text("protected\n", encoding="utf-8")
            outside = Path(tmpdir, "outside.txt")
            outside.write_text("outside\n", encoding="utf-8")
            (root / "link.txt").symlink_to(outside)
            tool = ApplyPatchTool()
            ctx = ToolContext(workspace=Workspace(root), permissions=PermissionPolicy())

            protected = tool.run(
                {
                    "patch": """*** Begin Patch
*** Update File: .GIT/config
@@
-protected
+changed
*** End Patch"""
                },
                ctx,
            )
            escaped = tool.run(
                {
                    "patch": """*** Begin Patch
*** Update File: link.txt
@@
-outside
+changed
*** End Patch"""
                },
                ctx,
            )
            parent_escape = tool.run(
                {
                    "patch": """*** Begin Patch
*** Update File: ../outside.txt
@@
-outside
+changed
*** End Patch"""
                },
                ctx,
            )

            self.assertFalse(protected.ok)
            self.assertIn("protected path", protected.stderr)
            self.assertFalse(escaped.ok)
            self.assertIn("symlink", escaped.stderr)
            self.assertFalse(parent_escape.ok)
            self.assertIn("must not contain '..'", parent_escape.stderr)
            self.assertEqual(git_config.read_text(encoding="utf-8"), "protected\n")
            self.assertEqual(outside.read_text(encoding="utf-8"), "outside\n")

    def test_agent_loop_executes_tool_call_and_writes_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            Path(workspace_path, "app.py").write_text("print('hello')\n", encoding="utf-8")
            store = SessionStore(Path(tmpdir, "sessions"))
            state = store.create(workspace_path)
            registry = ToolRegistry(
                [
                    ListFilesTool(),
                    ReadFileTool(),
                    SearchTool(),
                    ExecCommandTool(),
                ]
            )
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        content="I should inspect the files.",
                        reasoning_content="Need list_files before answering.",
                        tool_calls=[
                            ToolCall(
                                call_id="call_1",
                                name="list_files",
                                arguments={"path": "."},
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _model_response(content="Found app.py.", finish_reason="stop"),
                ]
            )

            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=registry,
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=4,
            )
            result = loop.run_turn(state, "Inspect this repository.")

            self.assertEqual(result.answer, "Found app.py.")
            self.assertIsInstance(provider.requests[0].messages[0], UserMessage)
            self.assertIsInstance(provider.requests[1].messages[1], AssistantMessage)
            self.assertEqual(
                provider.requests[1].messages[1].reasoning_content,
                "Need list_files before answering.",
            )
            self.assertIsInstance(provider.requests[1].messages[2], ToolResultMessage)
            events = [
                json.loads(line)
                for line in state.trace_path.read_text(encoding="utf-8").splitlines()
            ]
            event_names = [event["event"] for event in events]
            self.assertIn("turn_started", event_names)
            self.assertIn("model_response", event_names)
            self.assertIn("tool_call", event_names)
            self.assertIn("tool_result", event_names)
            self.assertIn("final_answer", event_names)

    def test_agent_loop_records_apply_patch_observation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            target = workspace_path / "app.py"
            target.write_text("value = 1\n", encoding="utf-8")
            store = SessionStore(Path(tmpdir, "sessions"))
            state = store.create(workspace_path)
            patch = """*** Begin Patch
*** Update File: app.py
@@
-value = 1
+value = 2
*** End Patch"""
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        tool_calls=[
                            ToolCall(
                                call_id="call_patch",
                                name="apply_patch",
                                arguments={"patch": patch},
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _model_response(content="Updated app.py.", finish_reason="stop"),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([ApplyPatchTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=4,
            )

            result = loop.run_turn(state, "Set value to 2.")
            events = read_jsonl_trace(state.trace_path)
            trajectory = make_trajectory(events, source_path=state.trace_path)
            observation = trajectory["turns"][0]["steps"][0]["observations"][0]

            self.assertEqual(result.answer, "Updated app.py.")
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 2\n")
            self.assertTrue(observation["output"]["metadata"]["applied"])
            self.assertEqual(
                observation["output"]["metadata"]["changed_files"],
                ["app.py"],
            )
            summary = trajectory["turns"][0]["execution_summary"]
            self.assertEqual(summary["changed_files"], ["app.py"])
            self.assertEqual(len(summary["evidence"]), 1)
            self.assertEqual(summary["evidence"][0]["call_id"], "call_patch")
            self.assertEqual(summary["evidence"][0]["operation"], "update")

    def test_agent_loop_records_exec_command_observation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            (workspace_path / "test_sample.py").write_text(
                "import unittest\n\n"
                "class SampleTest(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertEqual(2 + 2, 4)\n",
                encoding="utf-8",
            )
            store = SessionStore(Path(tmpdir, "sessions"))
            state = store.create(workspace_path)
            argv = ["python3", "-m", "unittest", "discover", "-s", "."]
            provider = ScriptedProviderAdapter(
                [
                    _model_response(
                        tool_calls=[
                            ToolCall(
                                call_id="call_exec",
                                name="exec_command",
                                arguments={"argv": argv, "cwd": "."},
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _model_response(content="Validation passed.", finish_reason="stop"),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                model=_model(provider),
                tools=ToolRegistry([ExecCommandTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=4,
                sandbox_policy=SandboxPolicy(DANGER_FULL_ACCESS, workspace_path),
            )

            result = loop.run_turn(state, "Run the tests.")
            trajectory = make_trajectory(
                read_jsonl_trace(state.trace_path),
                source_path=state.trace_path,
            )
            step = trajectory["turns"][0]["steps"][0]
            action = step["actions"][0]
            observation = step["observations"][0]

            self.assertEqual(result.answer, "Validation passed.")
            self.assertEqual(action["tool_name"], "exec_command")
            self.assertEqual(action["arguments"]["argv"], argv)
            self.assertTrue(observation["output"]["ok"])
            self.assertEqual(observation["output"]["metadata"]["status"], "success")
            self.assertEqual(observation["output"]["metadata"]["category"], "test")
            self.assertEqual(observation["output"]["metadata"]["cwd"], ".")
            summary = trajectory["turns"][0]["execution_summary"]
            self.assertEqual(summary["changed_files"], [])
            self.assertEqual(len(summary["evidence"]), 1)
            evidence = summary["evidence"][0]
            self.assertEqual(evidence["call_id"], action["action_id"])
            self.assertEqual(evidence["normalized_argv"], argv)
            self.assertEqual(evidence["exit_code"], 0)
            for field in ("status", "category", "cwd", "duration_ms", "timed_out", "interrupted"):
                self.assertEqual(evidence[field], observation["output"]["metadata"][field])

    def test_deepseek_provider_translates_canonical_messages_and_tools(self):
        provider = DeepSeekAdapter(
            api_key="test-key",
            model="deepseek-v4-flash",
            max_output_tokens=16000,
        )
        tool_schema = ListFilesTool().spec()
        messages = [
            UserMessage(content="Inspect files"),
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        call_id="call_1",
                        name="list_files",
                        arguments={"path": "."},
                    )
                ],
            ),
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="list_files",
                content=json.dumps({"ok": True, "stdout": "app.py"}),
            ),
        ]

        payload = provider.build_payload(
            ModelRequest(
                messages=tuple(messages),
                tools=(tool_schema,),
                system_prompt="system prompt",
            )
        )

        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(payload["max_tokens"], 16000)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "system prompt"})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "Inspect files"})
        self.assertEqual(payload["messages"][2]["role"], "assistant")
        self.assertEqual(payload["messages"][2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(payload["messages"][3]["role"], "tool")
        self.assertEqual(payload["messages"][3]["tool_call_id"], "call_1")
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["tools"][0]["function"]["name"], "list_files")

    def test_deepseek_provider_streams_and_assembles_reasoning_and_tool_calls(self):
        provider = DeepSeekAdapter(api_key="test-key", model="deepseek-v4-flash")
        chunks = [
            {
                "choices": [
                    {
                        "finish_reason": None,
                        "delta": {"reasoning_content": "I need to inspect "},
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": None,
                        "delta": {"reasoning_content": "files first."},
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": None,
                        "delta": {"content": "I will inspect the repository."},
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": None,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "list_files",
                                        "arguments": "{\"pa",
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": "th\":\".\"}"},
                                }
                            ]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        ]

        with patch.object(provider, "_stream_once", return_value=iter(chunks)):
            events = list(
                provider.stream_once(
                    ModelRequest(
                        messages=(UserMessage(content="Inspect files"),),
                        tools=(),
                        system_prompt="prompt",
                    )
                )
            )

        self.assertIsInstance(events[-1], ModelCompleted)
        response = events[-1].response
        self.assertEqual(
            [event.text for event in events if isinstance(event, ReasoningDelta)],
            ["I need to inspect ", "files first."],
        )
        self.assertEqual(
            [event.text for event in events if isinstance(event, TextDelta)],
            ["I will inspect the repository."],
        )
        self.assertEqual(response.reasoning, "I need to inspect files first.")
        self.assertEqual(response.text, "I will inspect the repository.")
        self.assertEqual(response.tool_calls[0].call_id, "call_1")
        self.assertEqual(response.tool_calls[0].name, "list_files")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "."})
        self.assertEqual(response.usage.input_tokens, 10)
        self.assertEqual(response.metadata.provider_id, "deepseek")

        messages = [
            UserMessage(content="Inspect files"),
            AssistantMessage(
                content=response.text,
                reasoning_content=response.reasoning,
                tool_calls=response.tool_calls,
            ),
        ]

        payload = provider.build_payload(
            ModelRequest(
                messages=tuple(messages),
                tools=(),
                system_prompt="system prompt",
            )
        )

        self.assertEqual(response.reasoning, "I need to inspect files first.")
        self.assertEqual(
            payload["messages"][2]["reasoning_content"],
            "I need to inspect files first.",
        )

    def test_openai_provider_translates_responses_input_and_tools(self):
        provider = OpenAIAdapter(
            api_key="test-key",
            model="gpt-5.6",
            max_output_tokens=16_000,
        )
        native_items = [
            {
                "id": "rs_1",
                "type": "reasoning",
                "encrypted_content": "opaque",
                "summary": [],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "list_files",
                "arguments": "{\"path\":\".\"}",
            },
        ]
        payload = provider.build_payload(
            ModelRequest(
                messages=(
                    UserMessage(content="Inspect files"),
                    AssistantMessage(
                        tool_calls=[
                            ToolCall(
                                call_id="call_1",
                                name="list_files",
                                arguments={"path": "."},
                            )
                        ],
                        provider_data={"openai": {"input_items": native_items}},
                    ),
                    ToolResultMessage(
                        tool_call_id="call_1",
                        tool_name="list_files",
                        content="app.py",
                    ),
                ),
                tools=(ListFilesTool().spec(),),
                system_prompt="system prompt",
            )
        )

        self.assertEqual(payload["model"], "gpt-5.6")
        self.assertEqual(payload["instructions"], "system prompt")
        self.assertEqual(payload["reasoning"], {"effort": "medium", "summary": "auto"})
        self.assertEqual(payload["include"], ["reasoning.encrypted_content"])
        self.assertEqual(payload["max_output_tokens"], 16_000)
        self.assertEqual(payload["input"][0], {"role": "user", "content": "Inspect files"})
        self.assertEqual(payload["input"][1:3], native_items)
        self.assertEqual(
            payload["input"][3],
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "app.py",
            },
        )
        self.assertEqual(payload["tools"][0]["name"], "list_files")
        self.assertNotIn("function", payload["tools"][0])

    def test_openai_provider_streams_and_preserves_continuation_items(self):
        provider = OpenAIAdapter(api_key="test-key", model="gpt-5.6")
        raw_output = [
            {
                "id": "rs_1",
                "type": "reasoning",
                "encrypted_content": "opaque",
                "summary": [{"type": "summary_text", "text": "Inspect first."}],
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "I will inspect the files."}
                ],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "list_files",
                "arguments": "{\"path\":\".\"}",
            },
        ]
        stream_events = [
            {"type": "response.created", "response": {"id": "resp_1"}},
            {
                "type": "response.reasoning_summary_text.delta",
                "delta": "Inspect first.",
            },
            {
                "type": "response.output_text.delta",
                "delta": "I will inspect the files.",
            },
            {
                "type": "response.output_item.added",
                "output_index": 2,
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "list_files",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 2,
                "delta": "{\"path\":\".\"}",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "model": "gpt-5.6-sol",
                    "status": "completed",
                    "output": raw_output,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 7,
                        "total_tokens": 17,
                        "input_tokens_details": {"cached_tokens": 3},
                        "output_tokens_details": {"reasoning_tokens": 2},
                    },
                },
            },
        ]

        with patch.object(provider, "_stream_once", return_value=iter(stream_events)):
            events = list(
                provider.stream_once(
                    ModelRequest(
                        messages=(UserMessage(content="Inspect files"),),
                        tools=(ListFilesTool().spec(),),
                    )
                )
            )

        response = events[-1].response
        self.assertIsInstance(events[-1], ModelCompleted)
        self.assertEqual(
            [event.text for event in events if isinstance(event, ReasoningDelta)],
            ["Inspect first."],
        )
        self.assertEqual(
            [event.text for event in events if isinstance(event, TextDelta)],
            ["I will inspect the files."],
        )
        self.assertEqual(response.reasoning, "Inspect first.")
        self.assertEqual(response.text, "I will inspect the files.")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "."})
        self.assertEqual(response.stop_reason, StopReason.TOOL_CALLS)
        self.assertEqual(response.usage.reasoning_tokens, 2)
        self.assertEqual(response.usage.cache_read_tokens, 3)
        self.assertEqual(response.metadata.response_id, "resp_1")
        self.assertEqual(response.metadata.response_model_id, "gpt-5.6-sol")
        self.assertEqual(response.provider_data["openai"]["input_items"], raw_output)

    def test_model_invoker_retries_transient_error_before_first_delta(self):
        provider = DeepSeekAdapter(
            api_key="test-key",
            model="deepseek-v4-flash",
        )
        attempts = 0

        def fake_stream_once(_payload, _cancel_event):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ModelError("server overloaded", retryable=True, status_code=503)
            return iter(
                [
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "delta": {"content": "Recovered."},
                            }
                        ],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                    }
                ]
            )

        with patch.object(provider, "_stream_once", new=fake_stream_once):
            events = list(
                _model(provider, max_attempts=2).stream(
                    ModelRequest(
                        messages=(UserMessage(content="Hello"),),
                        tools=(),
                        system_prompt="prompt",
                    )
                )
            )

        self.assertEqual(attempts, 2)
        self.assertEqual(sum(isinstance(event, RetryScheduled) for event in events), 1)
        self.assertIsInstance(events[-1], ModelCompleted)
        self.assertEqual(events[-1].response.text, "Recovered.")

    def test_model_invoker_does_not_retry_after_visible_delta(self):
        provider = DeepSeekAdapter(api_key="test-key")
        attempts = 0

        def interrupted_stream(_payload, _cancel_event):
            nonlocal attempts
            attempts += 1

            def chunks():
                yield {
                    "choices": [
                        {"finish_reason": None, "delta": {"content": "partial"}}
                    ]
                }
                raise ModelError("stream disconnected", retryable=True)

            return chunks()

        events = []
        with patch.object(provider, "_stream_once", new=interrupted_stream):
            with self.assertRaisesRegex(ModelError, "stream disconnected"):
                for event in _model(provider, max_attempts=3).stream(
                    ModelRequest(
                        messages=(UserMessage(content="Hello"),),
                        tools=(),
                        system_prompt="prompt",
                    )
                ):
                    events.append(event)

        self.assertEqual(attempts, 1)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], TextDelta)

    def test_model_invoker_requires_one_terminal_completed_event(self):
        class IncompleteAdapter:
            profile = ModelProfile(
                provider_id="incomplete",
                model_id="incomplete-model",
                context_window_tokens=10_000,
                max_output_tokens=1_000,
            )

            def stream_once(self, _request, _cancel_event=None):
                return iter(())

        with self.assertRaises(ModelError) as raised:
            list(
                _model(IncompleteAdapter()).stream(
                    ModelRequest(
                        messages=(UserMessage(content="Hello"),),
                        tools=(),
                    )
                )
            )

        self.assertEqual(raised.exception.kind, ModelErrorKind.PROTOCOL)
        self.assertTrue(raised.exception.retryable)

    def test_deepseek_adapter_rejects_invalid_tool_arguments(self):
        provider = DeepSeekAdapter(api_key="test-key")
        chunks = [
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "list_files",
                                        "arguments": "{bad json",
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        ]

        with patch.object(provider, "_stream_once", return_value=iter(chunks)):
            with self.assertRaises(ModelError) as raised:
                list(
                    provider.stream_once(
                        ModelRequest(
                            messages=(UserMessage(content="Inspect files"),),
                            tools=(),
                        )
                    )
                )

        self.assertEqual(raised.exception.kind, ModelErrorKind.PROTOCOL)
        self.assertFalse(raised.exception.retryable)

    def test_provider_factory_returns_selected_adapter_and_rejects_unknown_provider(self):
        adapter = create_provider_adapter(
            ProviderConfig(
                provider_id="deepseek",
                model_id="deepseek-v4-flash",
                options={"thinking": "enabled"},
            )
        )

        self.assertIsInstance(adapter, DeepSeekAdapter)
        self.assertEqual(adapter.profile.provider_id, "deepseek")
        self.assertEqual(adapter.profile.model_id, "deepseek-v4-flash")
        openai = create_provider_adapter(
            ProviderConfig(
                provider_id="openai",
                api_key="test-key",
                model_id="gpt-5.6",
                options={"reasoning_effort": "low"},
            )
        )
        self.assertIsInstance(openai, OpenAIAdapter)
        self.assertEqual(openai.profile.provider_id, "openai")
        self.assertEqual(openai.reasoning_effort, "low")
        with self.assertRaisesRegex(ValueError, "unsupported provider"):
            create_provider_adapter(ProviderConfig(provider_id="unknown"))

    def test_deepseek_sse_parser_ignores_keepalive_and_requires_done(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def __iter__(self):
                return iter(
                    [
                        b": keep-alive\n",
                        b"\n",
                        b'data: {"choices":[{"delta":{"content":"Hi"},'
                        b'"finish_reason":"stop"}]}\n',
                        b"data: [DONE]\n",
                    ]
                )

        provider = DeepSeekAdapter(api_key="test-key")
        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            chunks = list(provider._stream_once({"stream": True}, None))

        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "Hi")

    def test_openai_sse_parser_requires_terminal_event(self):
        completed = {
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "output": []},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def __iter__(self):
                return iter(
                    [
                        b": keep-alive\n",
                        b"event: response.completed\n",
                        f"data: {json.dumps(completed)}\n".encode(),
                    ]
                )

        provider = OpenAIAdapter(api_key="test-key")
        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            events = list(provider._stream_once({"stream": True}, None))

        self.assertEqual(events, [completed])

    def test_make_trajectory_groups_actions_and_observations(self):
        events = [
            {
                "ts": 100.0,
                "event": "turn_started",
                "session_id": "session-1",
                "turn_id": "session-1:1",
                "task": "Summarize repo",
                "workspace": "/tmp/repo",
                "max_steps": 4,
            },
            {
                "ts": 101.0,
                "event": "model_request",
                "session_id": "session-1",
                "turn_id": "session-1:1",
                "step": 1,
                "messages": 1,
                "tools": ["list_files"],
                "context": {
                    "estimated_tokens": 120,
                    "input_budget": 1000,
                    "active_messages": 1,
                    "summarized_messages": 0,
                },
            },
            {
                "ts": 102.0,
                "event": "model_response",
                "session_id": "session-1",
                "turn_id": "session-1:1",
                "step": 1,
                "content": "",
                "tool_calls": [
                    {
                        "call_id": "call_1",
                        "name": "list_files",
                        "arguments": {"path": "."},
                    }
                ],
            },
            {
                "ts": 103.0,
                "event": "tool_call",
                "session_id": "session-1",
                "turn_id": "session-1:1",
                "step": 1,
                "call": {
                    "call_id": "call_1",
                    "name": "list_files",
                    "arguments": {"path": "."},
                },
            },
            {
                "ts": 104.0,
                "event": "tool_result",
                "session_id": "session-1",
                "turn_id": "session-1:1",
                "step": 1,
                "call": {
                    "call_id": "call_1",
                    "name": "list_files",
                    "arguments": {"path": "."},
                },
                "result": {
                    "ok": True,
                    "stdout": "app.py",
                    "stderr": "",
                    "exit_code": None,
                    "truncated": False,
                    "path": None,
                    "metadata": {"count": 1},
                },
            },
            {
                "ts": 105.0,
                "event": "model_request",
                "session_id": "session-1",
                "turn_id": "session-1:1",
                "step": 2,
                "messages": 3,
                "tools": ["list_files"],
            },
            {
                "ts": 106.0,
                "event": "model_response",
                "session_id": "session-1",
                "turn_id": "session-1:1",
                "step": 2,
                "content": "Found app.py.",
                "tool_calls": [],
            },
            {
                "ts": 107.0,
                "event": "final_answer",
                "session_id": "session-1",
                "turn_id": "session-1:1",
                "step": 2,
                "answer": "Found app.py.",
            },
        ]

        trajectory = make_trajectory(events, source_path="trace.jsonl")

        self.assertEqual(
            trajectory["schema_version"],
            "my-agent.trajectory.v8",
        )
        self.assertEqual(trajectory["workspace"], "/tmp/repo")
        turn = trajectory["turns"][0]
        self.assertEqual(turn["task"], "Summarize repo")
        self.assertEqual(len(turn["steps"]), 2)
        self.assertEqual(turn["steps"][0]["actions"][0]["action_id"], "call_1")
        self.assertEqual(turn["steps"][0]["observations"][0]["action_id"], "call_1")
        self.assertEqual(turn["steps"][0]["observations"][0]["output"]["stdout"], "app.py")
        self.assertEqual(
            turn["steps"][0]["model_request"]["context"]["estimated_tokens"],
            120,
        )
        self.assertEqual(turn["final_answer"], "Found app.py.")
        self.assertEqual(trajectory["metrics"]["tool_calls"], 1)
        self.assertEqual(trajectory["metrics"]["failed_tool_calls"], 0)
        self.assertEqual(trajectory["metrics"]["duration_seconds"], 7.0)

    def test_make_trajectory_rejects_legacy_trace(self):
        events = [
            {
                "ts": 100.0,
                "event": "run_started",
                "task": "Legacy task",
                "workspace": "/tmp/repo",
                "max_steps": 4,
            }
        ]

        with self.assertRaisesRegex(ValueError, "turn_started"):
            make_trajectory(events)

    def test_make_trajectory_keeps_session_turns_separate(self):
        events = [
            {
                "ts": 100.0,
                "event": "turn_started",
                "session_id": "session-1",
                "turn_id": "session-1:1",
                "task": "First question",
                "workspace": "/tmp/repo",
                "max_steps": 4,
            },
            {
                "ts": 101.0,
                "event": "model_request",
                "session_id": "session-1",
                "turn_id": "session-1:1",
                "step": 1,
                "messages": 1,
                "tools": [],
            },
            {
                "ts": 102.0,
                "event": "model_response",
                "session_id": "session-1",
                "turn_id": "session-1:1",
                "step": 1,
                "content": "First answer",
                "tool_calls": [],
            },
            {
                "ts": 103.0,
                "event": "final_answer",
                "session_id": "session-1",
                "turn_id": "session-1:1",
                "step": 1,
                "answer": "First answer",
            },
            {
                "ts": 104.0,
                "event": "turn_started",
                "session_id": "session-1",
                "turn_id": "session-1:2",
                "task": "Second question",
                "workspace": "/tmp/repo",
                "max_steps": 4,
            },
            {
                "ts": 105.0,
                "event": "model_request",
                "session_id": "session-1",
                "turn_id": "session-1:2",
                "step": 1,
                "messages": 3,
                "tools": [],
            },
            {
                "ts": 106.0,
                "event": "model_response",
                "session_id": "session-1",
                "turn_id": "session-1:2",
                "step": 1,
                "content": "Second answer",
                "tool_calls": [],
            },
            {
                "ts": 107.0,
                "event": "final_answer",
                "session_id": "session-1",
                "turn_id": "session-1:2",
                "step": 1,
                "answer": "Second answer",
            },
        ]

        trajectory = make_trajectory(events, source_path="trace.jsonl")

        self.assertEqual(
            trajectory["schema_version"],
            "my-agent.trajectory.v8",
        )
        self.assertEqual(trajectory["session_id"], "session-1")
        self.assertEqual(len(trajectory["turns"]), 2)
        self.assertEqual(trajectory["turns"][0]["task"], "First question")
        self.assertEqual(trajectory["turns"][0]["final_answer"], "First answer")
        self.assertEqual(trajectory["turns"][1]["task"], "Second question")
        self.assertEqual(trajectory["turns"][1]["final_answer"], "Second answer")
        self.assertEqual(len(trajectory["turns"][0]["steps"]), 1)
        self.assertEqual(len(trajectory["turns"][1]["steps"]), 1)

    def test_read_jsonl_trace_rejects_bad_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir, "bad.jsonl")
            trace_path.write_text("{not json}\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                read_jsonl_trace(trace_path)


if __name__ == "__main__":
    unittest.main()
