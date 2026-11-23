import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from my_agent.agent_loop import AgentLoop
from my_agent.app_server import AppServer
from my_agent.cli import _chat, build_parser
from my_agent.context import ContextManager, ContextWindowError
from my_agent.event_store import EventStore
from my_agent.messages import AssistantMessage, ToolCall, UserMessage
from my_agent.model_invoker import ModelInvoker
from my_agent.permissions import PermissionPolicy
from my_agent.protocol import ContextCompactedEvent, StartTurnCommand
from my_agent.provider import ModelMetadata, ModelResponse, ScriptedProviderAdapter, StopReason
from my_agent.providers.deepseek import DeepSeekAdapter
from my_agent.providers.openai import OpenAIAdapter
from my_agent.rpc import RequestEnvelope
from my_agent.runtime import AgentRuntime
from my_agent.service import AgentService
from my_agent.session import PendingTurn, ProposedPlan, SessionStore
from my_agent.tools import ToolRegistry
from my_agent.trace import TraceRecorder
from my_agent.workspace import Workspace


def reply(text="", **kwargs):
    return ModelResponse.from_parts(text=text, metadata=ModelMetadata("scripted", "scripted-model"), **kwargs)


class ModelCompactionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = SessionStore(self.root / "sessions")
        self.state = self.store.create(self.root)
        self.source = [UserMessage("Keep RPC semantics. " * 80),
                       AssistantMessage("Investigated. " * 70, reasoning_content="private reasoning",
                                        provider_data={"opaque": "nonportable"})]
        self.state.messages = list(self.source)
        self.store.save(self.state)
        self.events = EventStore(self.store.root)
        self.service = AgentService(tool_registry_factory=lambda _: ToolRegistry([]),
                                   event_store=self.events, save_session=self.store.save)
        self.addCleanup(self.service.close)

    def runtime(self, replies):
        self.adapter = ScriptedProviderAdapter(replies)
        self.loop = AgentLoop(Workspace(self.root), ModelInvoker(self.adapter), ToolRegistry([]),
                              PermissionPolicy(), TraceRecorder(self.state.trace_path))
        self.service.register(AgentRuntime(self.loop, self.state, interactive_approvals=False))

    def test_manual_rpc_focus_exactly_once_no_turn_restore_and_continue(self):
        self.runtime([reply("Retain RPC semantics."), reply("Continued")])
        app = AppServer(workspace=self.root, session_store=self.store, agent_service=self.service,
                        runtime_factory=lambda *_: None, models=(), modes=())
        self.addCleanup(app.close)
        request = RequestEnvelope(method="context.compact", params={"session_id": self.state.session_id, "focus": "RPC compatibility"})
        result = app.handle(request)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(app.handle(request), result)
        self.assertEqual(len(self.adapter.requests), 1)
        req = self.adapter.requests[0]
        body = json.loads(req.messages[0].content)
        self.assertEqual(body["focus"], "RPC compatibility")
        self.assertIn("Keep RPC semantics.", body["transcript_fragment"])
        self.assertNotIn("private reasoning", req.messages[0].content)
        self.assertNotIn("nonportable", req.messages[0].content)
        for payload, limit_key in [(DeepSeekAdapter().build_payload(req), "max_tokens"),
                                   (OpenAIAdapter().build_payload(req), "max_output_tokens")]:
            self.assertNotIn("tools", payload)
            self.assertEqual(payload[limit_key], req.max_output_tokens)
        saved = self.store.load(self.state.session_id)
        self.assertEqual((saved.messages, saved.turn_counter, saved.pending_turn), (self.source, 0, None))
        self.assertEqual((saved.summarized_message_count, saved.context_summary), (2, "Retain RPC semantics."))
        events = self.events.replay(saved.session_id)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], ContextCompactedEvent)
        self.assertIsNone(events[0].turn_id)
        self.assertEqual(events[0].snapshot["trigger"], "manual")
        self.assertEqual(self.service.compact_context(saved.session_id).compacted_messages, 0)
        self.loop.run_turn(saved, "Continue")
        self.assertEqual(len(self.adapter.requests), 2)
        normal = self.adapter.requests[-1]
        self.assertNotIn(saved.context_summary, normal.system_prompt)
        self.assertIn(saved.context_summary, normal.messages[0].content)
        self.assertEqual(normal.messages[-1].content, "Continue")

    def test_cli_dispatches_focused_compact_without_sending_a_turn(self):
        adapter = ScriptedProviderAdapter([reply("Retain RPC semantics.")])
        loop = AgentLoop(Workspace(self.root), ModelInvoker(adapter), ToolRegistry([]),
                         PermissionPolicy(), TraceRecorder(self.state.trace_path))
        args = build_parser().parse_args(["resume", self.state.session_id])
        output = io.StringIO()
        with (patch("my_agent.cli._build_runtime", return_value=(loop, loop.tools)),
              patch("my_agent.cli._export_session_artifacts"),
              patch("my_agent.cli.LineEditor") as editor, redirect_stdout(output)):
            editor.return_value.__enter__.return_value.read.side_effect = ["/compact Keep the RPC decision", "/exit"]
            self.assertEqual(_chat(args, self.store, self.state), 0)
        self.assertIn("Context: ~", output.getvalue())
        self.assertEqual(len(adapter.requests), 1)
        self.assertEqual(json.loads(adapter.requests[0].messages[0].content)["focus"], "Keep the RPC decision")
        self.assertEqual(self.state.messages, self.source)

    def test_automatic_summary_committed_before_task_request(self):
        self.runtime([reply("Keep the RPC decision."), reply("Finished")])
        self.loop.context_manager = ContextManager(context_window_tokens=4000, reserve_output_tokens=300,
                                                  compact_threshold=0.18, compact_target=0.1)
        stream = self.adapter.stream_once
        seen = []
        def observe(request, cancel_event=None):
            if self.adapter.requests:
                saved = self.store.load(self.state.session_id, recover=False)
                seen.append((saved.context_summary, saved.summarized_message_count, saved.pending_turn.step,
                             request.messages[-1].content))
            return stream(request, cancel_event)
        with patch.object(self.adapter, "stream_once", side_effect=observe):
            self.service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Continue debugging"))
            self.service.wait(self.state.session_id)
        self.assertEqual(seen, [("Keep the RPC decision.", 2, 1, "Continue debugging")])
        self.assertEqual(len(self.adapter.requests), 2)
        self.assertIsNone(self.state.pending_turn)

    def test_rejected_model_output_preserves_cursor_events_and_original_messages(self):
        self.runtime([reply(), reply("partial", stop_reason=StopReason.LENGTH),
                      reply(tool_calls=[ToolCall("x", "exec_command", {})]), reply("a"*3500)])
        for reason in ("invalid handoff", "invalid handoff", "invalid handoff", "would not reduce"):
            with self.assertRaisesRegex(ContextWindowError, reason):
                self.service.compact_context(self.state.session_id)
            self.assertEqual((self.state.context_summary, self.state.summarized_message_count), ("", 0))
            self.assertEqual(self.store.load(self.state.session_id).messages, self.source)
        self.assertEqual(self.events.replay(self.state.session_id), [])

    def test_transaction_failure_after_save_does_not_publish_or_advance_live_state(self):
        self.runtime([reply("Keep RPC semantics.")])
        revision = self.state.revision
        with patch.object(self.events, "append", side_effect=OSError("cannot persist")), self.assertRaises(OSError):
            self.service.compact_context(self.state.session_id)
        for state in (self.state, self.store.load(self.state.session_id)):
            self.assertEqual((state.revision, state.context_summary, state.summarized_message_count), (revision, "", 0))
        self.assertEqual(self.events.replay(self.state.session_id), [])

    def test_failed_auto_checkpoint_does_not_send_next_request(self):
        self.runtime([reply("Keep RPC semantics."), reply("must not run")])
        self.loop.context_manager = ContextManager(context_window_tokens=4000, reserve_output_tokens=300,
                                                  compact_threshold=0.18, compact_target=0.1)
        append = self.events.append
        def fail(event):
            if isinstance(event, ContextCompactedEvent):
                raise OSError("checkpoint failure")
            append(event)
        with patch.object(self.events, "append", side_effect=fail):
            self.service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Continue debugging"))
            self.service.wait(self.state.session_id)
        self.assertEqual(len(self.adapter.requests), 1)
        for state in (self.state, self.store.load(self.state.session_id)):
            self.assertEqual((state.context_summary, state.summarized_message_count), ("", 0))

    def test_pending_turn_and_outstanding_plan_are_not_replaced(self):
        self.runtime([])
        self.state.pending_turn = PendingTurn("t", "task", 0)
        with self.assertRaisesRegex(RuntimeError, "pending"):
            self.service.compact_context(self.state.session_id)
        self.state.pending_turn = None
        self.state.proposed_plan = ProposedPlan("p", "t", self.source[-1].content)
        self.assertEqual(self.service.compact_context(self.state.session_id).compacted_messages, 0)
        self.assertEqual(self.adapter.requests, [])

    def test_chunked_summary_preserves_focus_and_consumes_each_source_fragment_once(self):
        self.runtime([reply("First handoff"), reply("Combined handoff"), reply("Final handoff")])
        self.loop.context_manager = ContextManager(context_window_tokens=1600, reserve_output_tokens=200)
        source = "unique line. " * 600
        self.state.messages = [UserMessage(source), AssistantMessage("outcome")]
        self.service.compact_context(self.state.session_id, "outcome")
        records = [json.loads(r.messages[0].content) for r in self.adapter.requests]
        self.assertGreater(len(records), 1)
        self.assertTrue(all(r["focus"] == "outcome" for r in records))
        self.assertEqual(records[1]["previous_handoff"], "First handoff")
        transcript = "".join(r["transcript_fragment"] for r in records)
        self.assertEqual([json.loads(line)["text"] for line in transcript.splitlines()], [source, "outcome"])
