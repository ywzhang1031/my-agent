import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from my_agent.agent_loop import AgentLoop
from my_agent.cli import _print_project_rules
from my_agent.context import ContextManager, ContextWindowError, estimate_request_tokens
from my_agent.database import StorageError
from my_agent.event_store import EventStore
from my_agent.messages import AssistantMessage, ToolCall, UserMessage
from my_agent.model_invoker import ModelInvoker, RetryPolicy
from my_agent.permissions import PermissionPolicy
from my_agent.protocol import AbortTurnCommand, StartTurnCommand, TurnCompletedEvent, TurnFailedEvent, TurnStartedEvent, event_from_dict, event_to_dict
from my_agent.provider import ModelMetadata, ModelResponse, ScriptedProviderAdapter, StopReason
from my_agent.rules import MAX_RULE_BYTES, load_project_rules
from my_agent.runtime import AgentRuntime
from my_agent.service import AgentService
from my_agent.session import SessionStore, message_from_dict, message_to_dict
from my_agent.terminal import render_context_usage
from my_agent.tools import ReadFileTool, ToolRegistry
from my_agent.trace import TraceRecorder
from my_agent.workspace import Workspace


def response(text="Done"):
    return ModelResponse.from_parts(text=text, metadata=ModelMetadata(provider_id="scripted", model_id="scripted-model"))


class ProjectRulesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "AGENTS.md"
        self.store = SessionStore(self.root / ".my-agent" / "sessions")
        self.state = self.store.create(self.root)

    def loop(self, replies):
        provider = ScriptedProviderAdapter(replies)
        loop = AgentLoop(workspace=Workspace(self.root), model=ModelInvoker(provider, RetryPolicy(max_attempts=1)),
                         tools=ToolRegistry([]), permissions=PermissionPolicy(), trace=TraceRecorder(self.state.trace_path))
        return loop, provider

    def service(self, replies):
        loop, provider = self.loop(replies)
        events = EventStore(self.store.root)
        service = AgentService(tool_registry_factory=lambda _: ToolRegistry([]), save_session=self.store.save, event_store=events)
        service.register(AgentRuntime(loop, self.state, interactive_approvals=False))
        self.addCleanup(service.close)
        subscription, _ = service.subscribe_events(session_ids=(self.state.session_id,))
        return service, subscription, events, provider

    def finish(self, service, subscription):
        while True:
            event = service.next_event(subscription, timeout=5)
            if isinstance(event, (TurnCompletedEvent, TurnFailedEvent)):
                service.wait(self.state.session_id)
                return event

    def test_missing_empty_unicode_and_limit(self):
        self.assertEqual(load_project_rules(self.root).status, "missing")
        self.path.write_bytes(b" \n")
        self.assertEqual(load_project_rules(self.root).status, "empty")
        text = "Use unittest.\n\u4e0d\u8981\u4fee\u6539\u751f\u6210\u6587\u4ef6\u3002\n"
        self.path.write_text(text, encoding="utf-8")
        rules = load_project_rules(self.root)
        self.assertEqual(rules.content, text)
        self.assertEqual(rules.size_bytes, len(text.encode()))
        self.assertEqual(rules.sha256, hashlib.sha256(text.encode()).hexdigest())
        self.assertNotIn("content", rules.manifest())
        self.path.write_bytes(b"x" * MAX_RULE_BYTES)
        self.assertEqual(load_project_rules(self.root).status, "loaded")
        self.path.write_bytes(b"x" * (MAX_RULE_BYTES + 1))
        self.assertEqual(load_project_rules(self.root).status, "error")
        self.assertEqual(load_project_rules(self.root).content, "")

    def test_rejects_symlink_directory_fifo_invalid_encoding_and_unreadable_file(self):
        target = self.root / "secret.txt"
        target.write_text("must not be loaded")
        self.path.symlink_to(target)
        self.assertEqual(load_project_rules(self.root).status, "error")
        self.path.unlink()
        self.path.mkdir()
        self.assertEqual(load_project_rules(self.root).status, "error")
        self.path.rmdir()
        os.mkfifo(self.path)
        self.assertEqual(load_project_rules(self.root).status, "error")
        self.path.unlink()
        self.path.write_bytes(b"\xff")
        self.assertEqual(load_project_rules(self.root).status, "error")
        with patch("my_agent.rules.os.open", side_effect=PermissionError("denied")):
            self.assertIn("denied", load_project_rules(self.root).error)

    def test_does_not_discover_parent_or_nested_rules(self):
        child = self.root / "child"
        child.mkdir()
        self.path.write_text("parent rules")
        self.assertEqual(load_project_rules(child).status, "missing")
        self.path.unlink()
        (child / "AGENTS.md").write_text("nested rules")
        self.assertEqual(load_project_rules(self.root).status, "missing")

    def test_injection_is_user_guidance_and_only_counted_once(self):
        self.path.write_text("Use unittest. Do not edit generated files.")
        loop, provider = self.loop([response()])
        loop.run_turn(self.state, "Explain the project")
        request = provider.requests[0]
        self.assertIn("Use unittest", request.messages[0].content)
        self.assertNotIn("Use unittest", request.system_prompt)
        self.assertEqual(request.messages[1], UserMessage("Explain the project"))
        self.assertTrue(all(m.project_rules is None for m in request.messages if isinstance(m, UserMessage)))
        snapshot = loop.inspect_context(self.state)
        self.assertGreater(snapshot.breakdown.project_rules_tokens, 0)
        self.assertEqual(snapshot.estimated_tokens, estimate_request_tokens(snapshot.system_prompt, snapshot.messages, []))
        self.assertEqual(sum(v for k, v in snapshot.breakdown.to_dict().items() if k != "total_tokens"), snapshot.estimated_tokens)
        self.assertEqual(sum("Use unittest" in m.content for m in snapshot.messages), 1)
        output = io.StringIO()
        _print_project_rules(snapshot, file=output)
        self.assertIn("SHA-256", output.getvalue())
        self.assertIn("Project rules", render_context_usage(snapshot, color=False))

    def test_reopened_retry_reuses_snapshot_and_next_turn_loads_new_version(self):
        self.path.write_text("original policy")
        loop, _ = self.loop([RuntimeError("request failed")])
        with self.assertRaisesRegex(RuntimeError, "request failed"):
            loop.run_turn(self.state, "First")
        frozen = self.state.current_project_rules()
        self.store.save(self.state)
        self.path.write_text("changed policy")
        self.state = self.store.load(self.state.session_id)
        loop, provider = self.loop([response(), response()])
        loop.retry_turn(self.state)
        self.assertIn("original policy", provider.requests[0].messages[0].content)
        loop.run_turn(self.state, "Second")
        self.assertIn("changed policy", provider.requests[1].messages[0].content)
        self.assertNotIn("original policy", " ".join(m.content for m in provider.requests[1].messages))
        self.assertEqual(self.state.messages[0].project_rules, frozen)

    def test_missing_file_snapshot_stays_missing_on_retry(self):
        loop, _ = self.loop([RuntimeError("request failed")])
        with self.assertRaisesRegex(RuntimeError, "request failed"):
            loop.run_turn(self.state, "First")
        self.path.write_text("newly added rules")
        loop, provider = self.loop([response()])
        loop.retry_turn(self.state)
        self.assertEqual(provider.requests[0].messages, (UserMessage("First"),))

    def test_rules_stay_frozen_between_tool_steps(self):
        self.path.write_text("frozen policy")
        call = ModelResponse.from_parts(tool_calls=[ToolCall("read-1", "read_file", {"path": "AGENTS.md"})],
            stop_reason=StopReason.TOOL_CALLS, metadata=ModelMetadata(provider_id="scripted", model_id="scripted-model"))
        loop, provider = self.loop([call, response()])
        loop.tools = ToolRegistry([ReadFileTool()])
        def update_file(event):
            if event.kind == "tool_started":
                self.path.write_text("policy changed during tool execution")
        loop.run_turn(self.state, "Read the file", event_handler=update_file)
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(provider.requests[0].messages[0], provider.requests[1].messages[0])
        self.assertIn("frozen policy", provider.requests[1].messages[0].content)
        self.assertIn("changed", self.path.read_text())

    def test_crash_recovery_keeps_recorded_rules_after_file_removal(self):
        self.path.write_text("persisted policy")
        loop, _ = self.loop([RuntimeError("interrupted request")])
        with self.assertRaisesRegex(RuntimeError, "interrupted request"):
            loop.run_turn(self.state, "Resume later")
        self.state.checkpoint_status = "running"
        self.store.save(self.state)
        self.path.unlink()
        self.state = self.store.load(self.state.session_id)
        loop, provider = self.loop([response()])
        with self.assertRaisesRegex(ValueError, "recovery confirmation"):
            loop.retry_turn(self.state)
        loop.retry_turn(self.state, recovery_confirmed=True)
        self.assertIn("persisted policy", provider.requests[0].messages[0].content)

    def test_legacy_messages_do_not_gain_invented_snapshots(self):
        message = message_from_dict({"role": "user", "content": "old turn"})
        self.assertIsNone(message.project_rules)
        self.assertEqual(message_to_dict(message), {"role": "user", "content": "old turn"})
        self.state.messages.append(message)
        self.path.write_text("new rules")
        loop, _ = self.loop([])
        self.assertIsNone(loop.inspect_context(self.state).project_rules)

    def test_compaction_keeps_current_rules_and_does_not_summarize_old_rules(self):
        self.path.write_text("old policy")
        old = load_project_rules(self.root)
        self.path.write_text("current policy")
        current = load_project_rules(self.root)
        self.state.messages = [UserMessage("old question " * 500, old), AssistantMessage("answer " * 500), UserMessage("new question", current)]
        manager = ContextManager(context_window_tokens=1500, reserve_output_tokens=200, summary_tokens=100)
        snapshot = manager.prepare(state=self.state, tools=[], system_prompt="System")
        self.assertGreater(snapshot.summarized_messages, 0)
        self.assertIn("current policy", snapshot.messages[0].content)
        self.assertNotIn("old policy", snapshot.system_prompt)
        self.assertEqual(self.state.messages[0].project_rules, old)

    def test_rules_over_context_budget_fail_without_model_request(self):
        self.path.write_text("x" * 4000)
        loop, provider = self.loop([response()])
        loop.context_manager = ContextManager(context_window_tokens=500, reserve_output_tokens=100)
        with self.assertRaises(ContextWindowError):
            loop.run_turn(self.state, "Hello")
        self.assertEqual(provider.requests, [])
        self.assertEqual(self.state.current_project_rules().content, "x" * 4000)

    def test_invalid_rules_fail_visibly_without_poisoning_service(self):
        self.path.write_bytes(b"x" * (MAX_RULE_BYTES + 1))
        service, sub, _, provider = self.service([response()])
        service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Hello"))
        failed = self.finish(service, sub)
        self.assertIsInstance(failed, TurnFailedEvent)
        self.assertIn("AGENTS.md exceeds", failed.error)
        self.assertEqual(provider.requests, [])
        self.assertIsNone(service.session_view(self.state.session_id).persistence_error)
        self.assertEqual(service.inspect_context(self.state.session_id).project_rules["status"], "error")
        service.dispatch(AbortTurnCommand(session_id=self.state.session_id))
        self.path.write_text("corrected rules")
        service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Try again"))
        self.assertIsInstance(self.finish(service, sub), TurnCompletedEvent)
        self.assertIn("corrected rules", provider.requests[0].messages[0].content)

    def test_snapshot_committed_before_request_and_manifest_event_is_replayable(self):
        self.path.write_text("recorded rules")
        service, sub, events, provider = self.service([response()])
        stream = provider.stream_once
        def verify_checkpoint(request, cancel_event=None):
            persisted = self.store.load(self.state.session_id, recover=False)
            self.assertEqual(persisted.current_project_rules().content, "recorded rules")
            return stream(request, cancel_event)
        with patch.object(provider, "stream_once", side_effect=verify_checkpoint):
            service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Hello"))
            self.assertIsInstance(self.finish(service, sub), TurnCompletedEvent)
        started = next(e for e in events.replay(self.state.session_id) if isinstance(e, TurnStartedEvent))
        self.assertEqual(started.project_rules["status"], "loaded")
        self.assertNotIn("content", started.project_rules)
        self.assertEqual(event_from_dict(event_to_dict(started)), started)
        traces = [json.loads(line) for line in self.state.trace_path.read_text().splitlines()]
        request = next(e for e in traces if e["event"] == "model_request")
        self.assertEqual(request["context"]["project_rules"]["sha256"], started.project_rules["sha256"])

    def test_checkpoint_failure_never_sends_uncommitted_rules(self):
        self.path.write_text("uncommitted rules")
        service, _, _, provider = self.service([response()])
        with patch.object(service, "_save_session", side_effect=OSError("injected failure")):
            with self.assertRaises(StorageError):
                service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Hello"))
            service.wait(self.state.session_id)
        self.assertEqual(provider.requests, [])
        self.assertIsNone(self.store.load(self.state.session_id).current_project_rules())
