import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from my_agent.agent_loop import AgentLoop
from my_agent.cli import build_tool_registry
from my_agent.context import estimate_text_tokens
from my_agent.summarization import history_record
from my_agent.database import StorageError
from my_agent.event_store import EventStore
from my_agent.execution import ProcessRunner, _read_capture
from my_agent.messages import ToolCall, ToolResultMessage
from my_agent.model_invoker import ModelInvoker
from my_agent.modes import PLAN_MODE
from my_agent.permissions import PermissionPolicy
from my_agent.provider import ModelMetadata, ModelResponse, ScriptedProviderAdapter, StopReason
from my_agent.protocol import ItemCompletedEvent, StartTurnCommand
from my_agent.runtime import AgentRuntime
from my_agent.sandbox import DANGER_FULL_ACCESS, SandboxPolicy
from my_agent.service import AgentService
from my_agent.session import SessionStore
from my_agent.tool_output import OutputQuotaError, ToolOutputStore
from my_agent.tools import COMMAND_PREVIEW_BYTES, ExecCommandTool, ReadToolOutputTool, ToolContext
from my_agent.trace import TraceRecorder
from my_agent.trajectory import read_jsonl_trace
from my_agent.workspace import Workspace


MARKER = "MIDDLE-EVIDENCE"
OUTPUT = "x" * 50_000 + MARKER + "z" * 50_000
COMMAND = {"argv": ["python3", "-m", "unittest", "discover", "-s", "."]}


def reply(call=None):
    return ModelResponse.from_parts(
        text="Done" if call is None else "", tool_calls=[call] if call else [],
        stop_reason=StopReason.TOOL_CALLS if call else StopReason.STOP,
        metadata=ModelMetadata(provider_id="scripted", model_id="scripted-model"))


class ReadOutputAdapter(ScriptedProviderAdapter):
    def stream_once(self, request, cancel_event=None):
        if len(self.requests) == 1:
            result = next(m for m in reversed(request.messages) if isinstance(m, ToolResultMessage))
            artifact = json.loads(result.content)["metadata"]["output_artifact"]
            self._replies.append(reply(ToolCall("read-1", "read_tool_output", {
                "artifact_id": artifact["artifact_id"], "stream": "stdout", "offset": 50_000, "limit": 64})))
        elif len(self.requests) == 2:
            self._replies.append(reply())
        yield from super().stream_once(request, cancel_event)


class OutputRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "test_output.py").write_text(
            "import sys, unittest\nfrom pathlib import Path\n"
            "class Output(unittest.TestCase):\n"
            "    def test_output(self):\n"
            "        p = Path('runs.txt')\n"
            "        p.write_text((p.read_text() if p.exists() else '') + 'run\\n')\n"
            f"        sys.stdout.write('x' * 50000 + {MARKER!r} + 'z' * 50000)\n")
        self.store = SessionStore(self.root / "sessions")
        self.state = self.store.create(self.workspace)
        self.outputs = ToolOutputStore(self.store.root, self.state.session_id)
        self.ctx = ToolContext(workspace=Workspace(self.workspace), permissions=PermissionPolicy("trusted"),
            sandbox_policy=SandboxPolicy(DANGER_FULL_ACCESS, self.workspace), output_store=self.outputs,
            turn_id=self.state.session_id + ":1", call_id="exec-1")

    def loop(self, adapter, state=None, mode=None):
        state = state or self.state
        options = {"mode": mode} if mode is not None else {}
        return AgentLoop(workspace=self.ctx.workspace, model=ModelInvoker(adapter),
            tools=build_tool_registry(mode) if mode else build_tool_registry(),
            permissions=self.ctx.permissions, trace=TraceRecorder(state.trace_path),
            sandbox_policy=self.ctx.sandbox_policy, **options)

    def test_preview_keeps_head_and_tail_but_small_output_is_unchanged(self):
        self.assertEqual(_read_capture(io.BytesIO(b"small"), 20), ("small", False))
        text, truncated = _read_capture(io.BytesIO(OUTPUT.encode()), COMMAND_PREVIEW_BYTES)
        self.assertTrue(truncated)
        self.assertTrue(text.startswith("x" * 20))
        self.assertTrue(text.endswith("z" * 20))
        self.assertNotIn(MARKER, text)
        self.assertLess(len(text), COMMAND_PREVIEW_BYTES + 32)

    def test_real_loop_reads_omitted_middle_without_rerunning_command(self):
        adapter = ReadOutputAdapter([reply(ToolCall("exec-1", "exec_command", COMMAND))])
        loop = self.loop(adapter)
        result = loop.run_turn(self.state, "Run the test, then inspect its middle output")
        self.assertEqual(result.stopped_by, "final_answer")
        self.assertEqual(len(adapter.requests), 3)
        before = "\n".join(m.content for m in adapter.requests[1].messages)
        after = "\n".join(m.content for m in adapter.requests[2].messages)
        self.assertNotIn(MARKER, before)
        self.assertIn(MARKER, after)
        results = [m for m in self.state.messages if isinstance(m, ToolResultMessage)]
        command = json.loads(results[0].content)
        artifact = command["metadata"]["output_artifact"]
        self.assertEqual(artifact["streams"]["stdout"]["stored_bytes"], len(OUTPUT))
        self.assertFalse(artifact["streams"]["stdout"]["storage_truncated"])
        self.assertEqual(artifact["call_id"], "exec-1")
        self.assertEqual(artifact["turn_id"], self.state.session_id + ":1")
        self.assertEqual(command["exit_code"], 0)
        self.assertTrue(command["ok"])
        self.assertTrue(command["truncated"])
        self.assertLess(estimate_text_tokens(results[0].content), estimate_text_tokens(OUTPUT) // 10)
        self.assertEqual((self.workspace / "runs.txt").read_text(), "run\n")
        self.assertIn(artifact["artifact_id"], history_record(results[0])["text"])
        trace = read_jsonl_trace(self.state.trace_path)
        self.assertEqual(sum(e["event"] == "tool_result" and e["call"]["name"] == "exec_command" for e in trace), 1)

        # A resumed session in read-only Plan mode can retrieve retained evidence, not execute it.
        self.store.save(self.state)
        resumed = SessionStore(self.store.root).load(self.state.session_id)
        reader = ScriptedProviderAdapter([reply(ToolCall("read-resumed", "read_tool_output", {
            "artifact_id": artifact["artifact_id"], "stream": "stdout", "offset": 50_000, "limit": 64})), reply()])
        self.loop(reader, resumed, PLAN_MODE).run_turn(resumed, "Inspect the previously saved output")
        last = next(m for m in reversed(resumed.messages) if isinstance(m, ToolResultMessage))
        self.assertTrue(json.loads(last.content)["ok"])
        self.assertIn(MARKER, json.loads(last.content)["stdout"])
        self.assertEqual((self.workspace / "runs.txt").read_text(), "run\n")

    def test_storage_failure_or_quota_preserves_actual_command_outcome(self):
        for error, reason in ((StorageError("disk failure"), "output_storage_failed"),
                              (OutputQuotaError("full"), "retention_quota_reached")):
            with self.subTest(reason=reason), patch.object(self.outputs, "save", side_effect=error):
                result = ExecCommandTool().run(COMMAND, self.ctx)
            self.assertTrue(result.ok)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.metadata["status"], "success")
            self.assertEqual(result.metadata["output_artifact"], {"status": "unavailable", "reason": reason})
        self.assertEqual((self.workspace / "runs.txt").read_text(), "run\nrun\n")

    def service(self, adapter):
        loop = self.loop(adapter)
        events = EventStore(self.store.root)
        service = AgentService(tool_registry_factory=build_tool_registry,
                               save_session=self.store.save, event_store=events)
        service.register(AgentRuntime(loop, self.state, interactive_approvals=False))
        self.addCleanup(service.close)
        return service, events

    def test_service_checkpoints_and_event_replay_keep_the_same_artifact_reference(self):
        adapter = ReadOutputAdapter([reply(ToolCall("exec-1", "exec_command", COMMAND))])
        service, events = self.service(adapter)
        stream = adapter.stream_once
        checked = []
        def verify(request, cancel_event=None):
            if adapter.requests:
                saved = self.store.load(self.state.session_id, recover=False)
                result = next(m for m in saved.messages if isinstance(m, ToolResultMessage))
                artifact = json.loads(result.content)["metadata"]["output_artifact"]
                self.assertEqual(self.outputs.read(artifact["artifact_id"], "stdout", 50_000, 64)["text"][:len(MARKER)], MARKER)
                self.assertNotIn(MARKER, result.content)
                checked.append(artifact["artifact_id"])
            return stream(request, cancel_event)
        with patch.object(adapter, "stream_once", side_effect=verify):
            service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Run and inspect"))
            service.wait(self.state.session_id, timeout=5)
        self.assertEqual(len(checked), 2)
        completed = next(event for event in events.replay(self.state.session_id)
                         if isinstance(event, ItemCompletedEvent) and event.data.get("name") == "exec_command")
        self.assertEqual(completed.data["result"]["metadata"]["output_artifact"]["artifact_id"], checked[0])
        self.assertEqual((self.workspace / "runs.txt").read_text(), "run\n")

    def test_result_checkpoint_failure_does_not_turn_an_artifact_into_execution_confirmation(self):
        adapter = ScriptedProviderAdapter([reply(ToolCall("exec-1", "exec_command", COMMAND)), reply()])
        service, _ = self.service(adapter)
        save = self.store.save
        def fail_result(state):
            if any(isinstance(m, ToolResultMessage) for m in state.messages):
                raise OSError("injected result checkpoint failure")
            save(state)
        with patch.object(service, "_save_session", side_effect=fail_result):
            service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Run once"))
            service.wait(self.state.session_id, timeout=5)
        self.assertEqual(len(adapter.requests), 1)
        restored = self.store.load(self.state.session_id)
        self.assertEqual(restored.pending_turn.pause_reason, "recovery_required")
        self.assertEqual(len(restored.pending_turn.pause_details["unknown_attempts"]), 1)
        self.assertEqual((self.workspace / "runs.txt").read_text(), "run\n")
        with self.store.database.transaction() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM tool_outputs").fetchone()[0], 1)

    def test_denied_and_unstarted_commands_never_create_artifacts(self):
        denied = ExecCommandTool().run({"argv": ["python3", "-c", "print(1)"]}, self.ctx)
        self.assertFalse(denied.ok)
        with patch("my_agent.execution.subprocess.Popen", side_effect=OSError("cannot spawn")):
            unstarted = ExecCommandTool().run(COMMAND, self.ctx)
        self.assertEqual(unstarted.metadata["status"], "spawn_error")
        for result in (denied, unstarted):
            self.assertNotIn("output_artifact", result.metadata)
        with self.outputs.database.transaction() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM tool_outputs").fetchone()[0], 0)

    def test_nonzero_exit_still_saves_both_streams(self):
        def save(stdout, stderr):
            return self.outputs.save(stdout, stderr, turn_id=self.ctx.turn_id, call_id="failure")
        process = ProcessRunner(COMMAND_PREVIEW_BYTES).run(
            [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"],
            self.workspace, 5, sandbox_policy=self.ctx.sandbox_policy, output_sink=save)
        self.assertEqual(process.exit_code, 3)
        self.assertEqual(process.status, "nonzero_exit")
        artifact_id = process.output_artifact["artifact_id"]
        self.assertEqual(self.outputs.read(artifact_id, "stdout")["text"], "out\n")
        self.assertEqual(self.outputs.read(artifact_id, "stderr")["text"], "err\n")

    def test_interrupted_command_retains_partial_output(self):
        cancel = threading.Event()
        timer = threading.Timer(0.3, cancel.set)
        timer.start()
        self.addCleanup(timer.join)
        process = ProcessRunner(COMMAND_PREVIEW_BYTES).run(
            [sys.executable, "-c", "import time; print('before-cancel', flush=True); time.sleep(30)"],
            self.workspace, 5, cancel_event=cancel, sandbox_policy=self.ctx.sandbox_policy,
            output_sink=lambda out, err: self.outputs.save(out, err, turn_id=self.ctx.turn_id, call_id="cancel"))
        self.assertEqual(process.status, "interrupted")
        self.assertEqual(self.outputs.read(process.output_artifact["artifact_id"], "stdout")["text"], "before-cancel\n")

    def test_timed_out_command_retains_partial_output_without_claiming_success(self):
        process = ProcessRunner(COMMAND_PREVIEW_BYTES).run(
            [sys.executable, "-c", "import time; print('before-timeout', flush=True); time.sleep(30)"],
            self.workspace, 1, sandbox_policy=self.ctx.sandbox_policy,
            output_sink=lambda out, err: self.outputs.save(out, err, turn_id=self.ctx.turn_id, call_id="timeout"))
        self.assertEqual(process.status, "timed_out")
        self.assertNotEqual(process.exit_code, 0)
        self.assertEqual(self.outputs.read(process.output_artifact["artifact_id"], "stdout")["text"], "before-timeout\n")

    def test_read_tool_rejects_paths_cross_session_ids_and_unbounded_ranges(self):
        artifact = self.outputs.save(io.BytesIO(b"evidence"), io.BytesIO(), turn_id=self.ctx.turn_id, call_id="saved")
        arguments = {"artifact_id": artifact["artifact_id"], "stream": "stdout"}
        tool = ReadToolOutputTool()
        for bad in ({**arguments, "path": "/etc/passwd"}, {**arguments, "offset": True},
                    {**arguments, "limit": 1_000_000}, {**arguments, "artifact_id": "../secret"}):
            self.assertEqual(tool.run(bad, self.ctx).metadata["status"], "invalid_request")
        self.ctx.output_store = ToolOutputStore(self.store.root, "other-session")
        self.assertEqual(tool.run(arguments, self.ctx).metadata["status"], "not_found")


if __name__ == "__main__":
    unittest.main()
