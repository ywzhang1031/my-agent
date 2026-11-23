import io
import json
import multiprocessing
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from my_agent.agent_loop import AgentLoop
from my_agent.attempt_store import AttemptStore
from my_agent.context import ContextManager, ContextWindowError, estimate_request_tokens
from my_agent.event_store import EventStore
from my_agent.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from my_agent.model_invoker import ModelInvoker
from my_agent.output_checkpoint import compact_v1, older_observations
from my_agent.permissions import PermissionPolicy
from my_agent.protocol import ContextCompactedEvent, StartTurnCommand
from my_agent.provider import ModelMetadata, ModelResponse, ScriptedProviderAdapter
from my_agent.runtime import AgentRuntime
from my_agent.sandbox import DANGER_FULL_ACCESS, SandboxPolicy
from my_agent.service import AgentService
from my_agent.session import PendingTurn, SessionStore, message_to_dict
from my_agent.tool_output import ToolOutputStore
from my_agent.tools import ExecCommandTool, ReadToolOutputTool, ToolRegistry, ToolResult
from my_agent.trace import TraceRecorder
from my_agent.workspace import Workspace


def reply(*calls):
    return ModelResponse.from_parts(text="" if calls else "done", tool_calls=calls,
        reasoning="fixture reasoning", provider_data={"input_items": [{"id": "opaque-id"}]},
        metadata=ModelMetadata(provider_id="scripted", model_id="scripted-model"))


def service_for(store, state, adapter):
    tools = ToolRegistry([ExecCommandTool(), ReadToolOutputTool()])
    loop = AgentLoop(Workspace(state.workspace), ModelInvoker(adapter), tools, PermissionPolicy("trusted"),
                     TraceRecorder(state.trace_path),
                     sandbox_policy=SandboxPolicy(DANGER_FULL_ACCESS, Path(state.workspace)))
    base = estimate_request_tokens(loop.instructions, [], tools.specs())
    loop.context_manager = ContextManager(context_window_tokens=base + 2550, reserve_output_tokens=400)
    service = AgentService(tool_registry_factory=lambda _: tools, save_session=store.save,
                           event_store=EventStore(store.root))
    service.register(AgentRuntime(loop, state, interactive_approvals=False))
    return service, loop


def request_contents(messages):
    return [message_to_dict(m) for m in messages]


def crash_at_checkpoint(root, session_id, phase):
    store = SessionStore(root)
    state = store.load(session_id)
    capture = Path(state.workspace) / "expected.json"

    class Adapter(ScriptedProviderAdapter):
        def stream_once(self, request, cancel_event=None):
            if len(self.requests) == 1:
                meta = json.loads(request.messages[-1].content)["metadata"]["output_artifact"]
                self._replies.append(reply(*[ToolCall(f"page-{n}", "read_tool_output", {
                    "artifact_id": meta["artifact_id"], "stream": "stdout", "offset": 12000+n,
                    "limit": size}) for n, size in enumerate((4096, 32))]))
            if len(self.requests) == 2 and phase == "after":
                capture.write_text(json.dumps(request_contents(request.messages)))
                os._exit(32)
            yield from super().stream_once(request, cancel_event)

    adapter = Adapter([reply(ToolCall("cmd", "exec_command", {
        "argv": ["python3", "-m", "unittest", "discover", "-s", "."]}))])
    service, loop = service_for(store, state, adapter)
    publish = service._publish_runtime_event
    def publish_or_crash(handle, event):
        publish(handle, event)
        if event.kind == "context_compacted" and phase == "before":
            capture.write_text(json.dumps(request_contents(loop.inspect_context(state).messages)))
            os._exit(31)  # Session, event and revision are still in the open SQLite transaction.
    with patch.object(service, "_publish_runtime_event", side_effect=publish_or_crash):
        service.dispatch(StartTurnCommand(session_id=session_id, user_input="Run once and inspect evidence"))
        service.wait(session_id, timeout=10)
        raise AssertionError(f"missed {phase} crash point: {state.pending_turn}")


class OutputCheckpointTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workspace = Path(temp.name).resolve()
        self.store = SessionStore(self.workspace / "sessions")
        self.state = self.store.create(self.workspace)

    def check_restart(self, phase):
        (self.workspace / "AGENTS.md").write_text("Keep the original constraint.")
        (self.workspace / "test_sample.py").write_text(
            "import unittest\nfrom pathlib import Path\nclass Sample(unittest.TestCase):\n"
            "    def test_work(self):\n"
            "        with open('runs.txt', 'a') as f: f.write('run\\n')\n"
            "        print('a'*12000 + 'MIDDLE-EVIDENCE' + 'z'*12000)\n")
        child = multiprocessing.get_context("spawn").Process(target=crash_at_checkpoint,
            args=(self.store.root, self.state.session_id, phase))
        child.start()
        try:
            child.join(20)
            self.assertEqual(child.exitcode, 31 if phase == "before" else 32)
        finally:
            if child.is_alive():
                child.kill()
                child.join()
        state = self.store.load(self.state.session_id)
        pending = state.pending_turn
        self.assertEqual((pending.pause_reason, pending.step, pending.tool_calls_executed),
                         ("recovery_required", 3, 3))
        self.assertEqual(pending.pause_details["unknown_attempts"], [])
        self.assertEqual(len(pending.compacted_outputs), int(phase == "after"))
        events = EventStore(self.store.root)
        def compactions():
            return [e for e in events.replay(state.session_id) if isinstance(e, ContextCompactedEvent)]
        self.assertEqual(len(compactions()), int(phase == "after"))
        original = list(state.messages)
        adapter = ScriptedProviderAdapter([reply()])
        service, loop = service_for(self.store, state, adapter)
        self.addCleanup(service.close)
        raw = loop.inspect_context(replace(state, pending_turn=replace(pending, compacted_outputs=())))
        self.assertGreater(raw.estimated_tokens, raw.input_budget)
        (self.workspace / "AGENTS.md").write_text("Do not substitute new rules on resume.")
        self.assertEqual(adapter.requests, [])
        with self.assertRaisesRegex(ValueError, "confirmation required"):
            service.dispatch(StartTurnCommand(session_id=state.session_id, retry=True))
        service.dispatch(StartTurnCommand(session_id=state.session_id, retry=True, recovery_confirmed=True))
        service.wait(state.session_id, timeout=5)
        self.assertIsNone(state.pending_turn)
        self.assertEqual(len(adapter.requests), 1)
        request = adapter.requests[0]
        self.assertEqual(request_contents(request.messages), json.loads((self.workspace / "expected.json").read_text()))
        self.assertEqual(state.messages[:-1], original)
        self.assertIn("Keep the original constraint.", request.messages[0].content)
        self.assertEqual(request.messages[-3:], tuple(original[-3:]))  # Entire newest batch stays verbatim.
        short = next(m for m in request.messages if getattr(m, "tool_call_id", "") == "cmd")
        self.assertTrue(json.loads(short.content)["metadata"]["context_compacted"])
        self.assertIn("MIDDLE-EVIDENCE", request.messages[-2].content)
        self.assertEqual((self.workspace / "runs.txt").read_text(), "run\n")
        self.assertEqual(len(compactions()), 1)
        snapshot = compactions()[0].snapshot
        self.assertEqual(snapshot["compacted_outputs"], 1)
        self.assertEqual(snapshot["estimated_tokens_before_compaction"], raw.estimated_tokens)
        self.assertLessEqual(snapshot["estimated_tokens"], raw.input_budget)

    def test_crash_before_compaction_commit(self):
        self.check_restart("before")

    def test_crash_after_compaction_commit(self):
        self.check_restart("after")

    def test_only_committed_recoverable_observations_can_shrink(self):
        state = self.state
        outputs = ToolOutputStore(self.store.root, state.session_id)
        artifact = outputs.save(io.BytesIO(b"x"*6000), io.BytesIO(), turn_id="prior", call_id="cmd")
        page = outputs.read(artifact["artifact_id"], "stdout")
        body = page.pop("text")
        result = ToolResult(True, stdout=body, metadata={"status": "output_read", **page}).to_dict()
        call = ToolCall("page", "read_tool_output", {"artifact_id": artifact["artifact_id"], "stream": "stdout"})
        observation = ToolResultMessage("page", call.name, json.dumps(result, ensure_ascii=False))
        state.pending_turn = PendingTurn("active", "task", 0)
        state.messages = [UserMessage("task"), AssistantMessage(tool_calls=[call]), observation,
            AssistantMessage(tool_calls=[ToolCall("last", "read_file", {})]), ToolResultMessage("last", "read_file", "{}")]
        tools = [ReadToolOutputTool().spec()]
        manager = ContextManager(context_window_tokens=850, reserve_output_tokens=100)
        def prepare():
            return manager.prepare(state=state, tools=tools, system_prompt="prompt", output_store=outputs)
        # Bytes exist, but the matching execution receipt has not been committed.
        with self.assertRaises(ContextWindowError):
            prepare()
        self.assertFalse(state.pending_turn.compacted_outputs)
        AttemptStore(self.store.database).finish(state.session_id, "active", {
            "attempt_id": "receipt", "name": call.name, "call_id": call.call_id, "result": result})
        self.assertEqual(list(older_observations(state.messages[:-1], 0)), [])
        with self.assertRaises(ContextWindowError):
            ContextManager(context_window_tokens=100, reserve_output_tokens=50).prepare(
                state=state, tools=tools, system_prompt="prompt", output_store=outputs)
        self.assertFalse(state.pending_turn.compacted_outputs)
        snapshot = prepare()
        self.assertEqual(snapshot.compacted_outputs, 1)
        metadata = json.loads(snapshot.messages[2].content)["metadata"]
        self.assertEqual((metadata["offset"], metadata["next_offset"], metadata["turn_id"]), (0, 4096, "prior"))
        self.assertEqual(prepare().compacted_outputs, 0)
        with self.store.database.transaction() as connection:
            connection.execute("DELETE FROM tool_outputs WHERE artifact_id=?", (artifact["artifact_id"],))
        with self.assertRaisesRegex(ContextWindowError, "missing artifact"):
            manager.prepare(state=state, tools=tools, system_prompt="prompt", output_store=outputs,
                            revalidate_outputs=True)
        self.assertEqual(state.messages[2], observation)
        state.messages[2] = replace(observation, content=observation.content + " ")
        with self.assertRaisesRegex(ContextWindowError, "original observation"):
            prepare()

    def test_unknown_unavailable_and_partial_storage_are_not_candidates(self):
        for status, ref in [("unknown", {}), ("success", {"status": "unavailable"}),
                            ("nonzero_exit", {"status": "stored", "turn_id": "t", "call_id": "c",
                             "streams": {"stdout": {"storage_truncated": True}}})]:
            result = ToolResult(False, stderr="e"*2000, exit_code=2,
                                metadata={"status": status, "output_artifact": ref})
            message = ToolResultMessage("c", "exec_command", json.dumps(result.to_dict()))
            self.assertIsNone(compact_v1(message, "t"))

    def test_unsatisfied_budget_does_not_commit_partial_history_summary(self):
        self.state.messages = [UserMessage("old"*800), UserMessage("required"*3000)]
        self.state.pending_turn = PendingTurn("t", self.state.messages[-1].content, 1)
        manager = ContextManager(context_window_tokens=800, reserve_output_tokens=50)
        with self.assertRaises(ContextWindowError):
            manager.prepare(state=self.state, tools=[], system_prompt="prompt")
        self.assertEqual((self.state.summarized_message_count, self.state.context_summary), (0, ""))
