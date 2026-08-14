import io
import json
import os
import subprocess
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from my_agent.agent_loop import AgentEvent, AgentLoop
from my_agent.cli import build_parser, build_tool_registry
from my_agent.context import ContextManager
from my_agent.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from my_agent.permissions import PermissionPolicy
from my_agent.provider import ProviderError, ProviderResponse, ScriptedProvider
from my_agent.providers.deepseek import DeepSeekProvider
from my_agent.session import PendingTurn, SessionStore
from my_agent.terminal import LineEditor, StreamRenderer
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


class MyAgentTests(unittest.TestCase):
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
            state.messages.append(UserMessage(content="Retry this request"))
            state.pending_turn = PendingTurn(
                turn_id=f"{state.session_id}:2",
                task="Retry this request",
                message_start=3,
                step=2,
                tool_calls_executed=1,
                seen_calls={"call-key": 1},
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
            provider = ScriptedProvider(
                [
                    ProviderResponse(
                        content="First answer.",
                        reasoning_content="First reasoning.",
                    ),
                    ProviderResponse(content="Second answer."),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                provider=provider,
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
            self.assertEqual(len(provider.requests[0]), 1)
            self.assertEqual(len(provider.requests[1]), 3)
            self.assertEqual(provider.requests[1][0], UserMessage(content="First question"))
            self.assertEqual(
                provider.requests[1][1],
                AssistantMessage(
                    content="First answer.",
                    reasoning_content="First reasoning.",
                ),
            )
            self.assertEqual(provider.requests[1][2], UserMessage(content="Second question"))
            self.assertEqual(store.load(state.session_id).messages, state.messages)

            events = [
                json.loads(line)
                for line in state.trace_path.read_text(encoding="utf-8").splitlines()
            ]
            turn_events = [event for event in events if event["event"] == "turn_started"]
            self.assertEqual(len(turn_events), 2)
            self.assertNotEqual(turn_events[0]["turn_id"], turn_events[1]["turn_id"])

    def test_agent_loop_retries_pending_turn_without_duplicating_user_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            store = SessionStore(Path(tmpdir, "sessions"))
            state = store.create(workspace_path)
            provider = ScriptedProvider(
                [
                    ProviderError("temporary provider failure", retryable=True),
                    ProviderResponse(content="Recovered answer."),
                ]
            )
            events: list[AgentEvent] = []
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                provider=provider,
                tools=ToolRegistry([ListFilesTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=4,
            )

            with self.assertRaisesRegex(ProviderError, "temporary provider failure"):
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
            self.assertEqual(trajectory["schema_version"], "my-agent.trajectory.v3")
            self.assertEqual(trajectory["turns"][0]["outcome"]["status"], "completed")
            self.assertEqual(len(trajectory["turns"][0]["errors"]), 1)
            self.assertEqual(trajectory["metrics"]["turn_errors"], 1)

    def test_agent_loop_retry_after_tool_result_does_not_repeat_the_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir, "workspace")
            workspace_path.mkdir()
            (workspace_path / "app.py").write_text("value = 1\n", encoding="utf-8")
            state = SessionStore(Path(tmpdir, "sessions")).create(workspace_path)
            provider = ScriptedProvider(
                [
                    ProviderResponse(
                        tool_calls=[
                            ToolCall(
                                call_id="call_list",
                                name="list_files",
                                arguments={"path": "."},
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    ProviderError("second model request failed", retryable=True),
                    ProviderResponse(content="Found app.py."),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                provider=provider,
                tools=ToolRegistry([ListFilesTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=4,
            )

            with self.assertRaisesRegex(ProviderError, "second model request failed"):
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
                provider=ScriptedProvider([ProviderError("failed")]),
                tools=ToolRegistry([]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=2,
            )

            with self.assertRaises(ProviderError):
                loop.run_turn(state, "Failed request")
            pending = loop.abort_turn(state)
            trajectory = make_trajectory(read_jsonl_trace(state.trace_path), state.trace_path)

            self.assertEqual(pending.task, "Failed request")
            self.assertIsNone(state.pending_turn)
            self.assertEqual(state.messages, [])
            self.assertEqual(trajectory["turns"][0]["outcome"]["status"], "aborted")
            self.assertEqual(trajectory["metrics"]["turn_aborts"], 1)

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

    def test_cli_only_accepts_current_subcommands(self):
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
                "inspect",
                "this",
                "repo",
            ]
        )
        chat = parser.parse_args(["chat", "--workspace", "/tmp/repo"])
        resume = parser.parse_args(["resume", "session-123", "--workspace", "/tmp/repo"])

        self.assertEqual(ask.command, "ask")
        self.assertEqual(ask.prompt, ["inspect", "this", "repo"])
        self.assertFalse(ask.stream)
        self.assertTrue(ask.show_thinking)
        self.assertEqual(ask.context_window_tokens, 200000)
        self.assertEqual(ask.max_output_tokens, 16000)
        self.assertEqual(chat.command, "chat")
        self.assertEqual(resume.command, "resume")
        self.assertEqual(resume.session_id, "session-123")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--workspace", "/tmp/repo", "inspect", "this", "repo"])

    def test_line_editor_enables_navigation_and_persists_history(self):
        class FakeReadline:
            def __init__(self):
                self.loaded = None
                self.written = None
                self.bindings = []
                self.history_length = None

            def read_history_file(self, path):
                self.loaded = path

            def write_history_file(self, path):
                self.written = path
                Path(path).write_text("history\n", encoding="utf-8")

            def parse_and_bind(self, binding):
                self.bindings.append(binding)

            def set_history_length(self, length):
                self.history_length = length

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir, "state", "history")
            history_path.parent.mkdir()
            history_path.write_text("old command\n", encoding="utf-8")
            fake_readline = FakeReadline()
            editor = LineEditor(
                history_path=history_path,
                readline_module=fake_readline,
                input_func=lambda prompt: f"{prompt}edited",
            )

            with editor:
                value = editor.read("you> ")

            self.assertEqual(value, "you> edited")
            self.assertEqual(fake_readline.loaded, str(history_path))
            self.assertEqual(fake_readline.written, str(history_path))
            self.assertIn("set editing-mode emacs", fake_readline.bindings)
            self.assertEqual(fake_readline.history_length, 500)
            self.assertEqual(history_path.stat().st_mode & 0o777, 0o600)

    def test_chat_help_lists_itself(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    "python3",
                    "cli.py",
                    "chat",
                    "--workspace",
                    tmpdir,
                    "--sessions-dir",
                    str(Path(tmpdir, "sessions")),
                ],
                input="/help\n/exit\n",
                text=True,
                capture_output=True,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("/trace /tools /help /exit", completed.stdout)

    def test_stream_renderer_displays_reasoning_content_and_tool_progress(self):
        output = io.StringIO()
        status = io.StringIO()
        renderer = StreamRenderer(
            output=output,
            status_output=status,
            answer_prefix="assistant> ",
            show_thinking=True,
        )

        renderer.handle(AgentEvent(kind="model_started"))
        renderer.handle(AgentEvent(kind="reasoning_delta", text="Inspecting."))
        renderer.handle(AgentEvent(kind="content_delta", text="I found "))
        renderer.handle(AgentEvent(kind="content_delta", text="app.py."))
        renderer.handle(
            AgentEvent(
                kind="tool_started",
                data={"name": "read_file", "arguments": {"path": "app.py"}},
            )
        )
        renderer.handle(
            AgentEvent(
                kind="tool_finished",
                data={"name": "read_file", "ok": True, "status": "success"},
            )
        )
        renderer.finish("I found app.py.")

        self.assertIn("thinking> Inspecting.", output.getvalue())
        self.assertIn("assistant> I found app.py.", output.getvalue())
        self.assertIn("tool> read_file", status.getvalue())
        self.assertIn("tool< read_file success", status.getvalue())

    def test_default_tool_registry_exposes_current_tools(self):
        self.assertEqual(
            build_tool_registry().names(),
            ["apply_patch", "exec_command", "git_diff", "list_files", "read_file", "search"],
        )

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
            ctx = ToolContext(workspace=Workspace(root), permissions=PermissionPolicy())
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
                "        self.assertNotIn('DEEPSEEK_API_KEY', os.environ)\n",
                encoding="utf-8",
            )
            ctx = ToolContext(workspace=Workspace(tmpdir), permissions=PermissionPolicy())

            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "should-not-leak"}):
                result = ExecCommandTool().run(
                    {"argv": ["python3", "-m", "unittest", "discover", "-s", "."]},
                    ctx,
                )

            self.assertTrue(result.ok, result.stderr)
            self.assertNotIn("DEEPSEEK_API_KEY", result.metadata["environment_keys"])

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
            ctx = ToolContext(workspace=Workspace(tmpdir), permissions=PermissionPolicy())

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
            ctx = ToolContext(workspace=Workspace(tmpdir), permissions=PermissionPolicy())

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
            provider = ScriptedProvider(
                [
                    ProviderResponse(
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
                    ProviderResponse(content="Found app.py.", finish_reason="stop"),
                ]
            )

            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                provider=provider,
                tools=registry,
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=4,
            )
            result = loop.run_turn(state, "Inspect this repository.")

            self.assertEqual(result.answer, "Found app.py.")
            self.assertIsInstance(provider.requests[0][0], UserMessage)
            self.assertIsInstance(provider.requests[1][1], AssistantMessage)
            self.assertEqual(
                provider.requests[1][1].reasoning_content,
                "Need list_files before answering.",
            )
            self.assertIsInstance(provider.requests[1][2], ToolResultMessage)
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
            provider = ScriptedProvider(
                [
                    ProviderResponse(
                        tool_calls=[
                            ToolCall(
                                call_id="call_patch",
                                name="apply_patch",
                                arguments={"patch": patch},
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    ProviderResponse(content="Updated app.py.", finish_reason="stop"),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                provider=provider,
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
            provider = ScriptedProvider(
                [
                    ProviderResponse(
                        tool_calls=[
                            ToolCall(
                                call_id="call_exec",
                                name="exec_command",
                                arguments={"argv": argv, "cwd": "."},
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    ProviderResponse(content="Validation passed.", finish_reason="stop"),
                ]
            )
            loop = AgentLoop(
                workspace=Workspace(workspace_path),
                provider=provider,
                tools=ToolRegistry([ExecCommandTool()]),
                permissions=PermissionPolicy(),
                trace=TraceRecorder(state.trace_path),
                max_steps=4,
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

    def test_deepseek_provider_translates_canonical_messages_and_tools(self):
        provider = DeepSeekProvider(
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

        payload = provider.build_payload(messages, [tool_schema], "system prompt")

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
        provider = DeepSeekProvider(api_key="test-key", model="deepseek-v4-flash")
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
            events = list(provider.stream([UserMessage(content="Inspect files")], [], "prompt"))

        response = events[-1].response
        self.assertIsNotNone(response)
        self.assertEqual(
            [event.text for event in events if event.kind == "reasoning_delta"],
            ["I need to inspect ", "files first."],
        )
        self.assertEqual(
            [event.text for event in events if event.kind == "content_delta"],
            ["I will inspect the repository."],
        )
        self.assertEqual(response.reasoning_content, "I need to inspect files first.")
        self.assertEqual(response.content, "I will inspect the repository.")
        self.assertEqual(response.tool_calls[0].call_id, "call_1")
        self.assertEqual(response.tool_calls[0].name, "list_files")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "."})
        self.assertEqual(response.usage.input_tokens, 10)

        messages = [
            UserMessage(content="Inspect files"),
            AssistantMessage(
                content=response.content,
                reasoning_content=response.reasoning_content,
                tool_calls=response.tool_calls,
            ),
        ]

        payload = provider.build_payload(messages, [], "system prompt")

        self.assertEqual(response.reasoning_content, "I need to inspect files first.")
        self.assertEqual(
            payload["messages"][2]["reasoning_content"],
            "I need to inspect files first.",
        )

    def test_deepseek_provider_retries_transient_error_before_first_delta(self):
        provider = DeepSeekProvider(
            api_key="test-key",
            model="deepseek-v4-flash",
            max_retries=1,
            retry_base_seconds=0,
        )
        attempts = 0

        def fake_stream_once(_payload):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ProviderError("server overloaded", retryable=True, status_code=503)
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
            events = list(provider.stream([UserMessage(content="Hello")], [], "prompt"))

        self.assertEqual(attempts, 2)
        self.assertEqual([event.kind for event in events].count("retry"), 1)
        self.assertEqual(events[-1].response.content, "Recovered.")

    def test_deepseek_provider_does_not_auto_retry_after_visible_delta(self):
        provider = DeepSeekProvider(
            api_key="test-key",
            max_retries=2,
            retry_base_seconds=0,
        )
        attempts = 0

        def interrupted_stream(_payload):
            nonlocal attempts
            attempts += 1

            def chunks():
                yield {
                    "choices": [
                        {"finish_reason": None, "delta": {"content": "partial"}}
                    ]
                }
                raise ProviderError("stream disconnected", retryable=True)

            return chunks()

        events = []
        with patch.object(provider, "_stream_once", new=interrupted_stream):
            with self.assertRaisesRegex(ProviderError, "stream disconnected"):
                for event in provider.stream([UserMessage(content="Hello")], [], "prompt"):
                    events.append(event)

        self.assertEqual(attempts, 1)
        self.assertEqual([event.kind for event in events], ["content_delta"])

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

        provider = DeepSeekProvider(api_key="test-key")
        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            chunks = list(provider._stream_once({"stream": True}))

        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "Hi")

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
            "my-agent.trajectory.v3",
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
            "my-agent.trajectory.v3",
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
