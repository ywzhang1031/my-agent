import tempfile
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from my_agent.agent_loop import AgentLoop
from my_agent.database import StorageError
from my_agent.event_store import EventStore
from my_agent.messages import ToolCall
from my_agent.model_invoker import ModelInvoker, RetryPolicy
from my_agent.permissions import PermissionPolicy
from my_agent.protocol import RenameSessionCommand, StartTurnCommand, TurnCompletedEvent, TurnFailedEvent
from my_agent.provider import ModelMetadata, ModelResponse, ScriptedProviderAdapter, StopReason
from my_agent.runtime import AgentRuntime
from my_agent.service import AgentService
from my_agent.session import SessionStore
from my_agent.tools import ApplyPatchTool, ToolRegistry
from my_agent.trace import TraceRecorder
from my_agent.workspace import Workspace


def response(text="", calls=()):
    return ModelResponse.from_parts(
        text=text, tool_calls=calls,
        stop_reason=StopReason.TOOL_CALLS if calls else StopReason.STOP,
        metadata=ModelMetadata(provider_id="scripted", model_id="scripted-model"),
    )


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = SessionStore(self.root / ".my-agent" / "sessions")
        self.state = self.store.create(self.root)
        self.events = EventStore(self.store.root)

    def service(self, replies):
        self.provider = ScriptedProviderAdapter(replies)
        tools = ToolRegistry([ApplyPatchTool()])
        loop = AgentLoop(
            workspace=Workspace(self.root),
            model=ModelInvoker(self.provider, RetryPolicy(max_attempts=1)),
            tools=tools, permissions=PermissionPolicy(profile_id="trusted"),
            trace=TraceRecorder(self.state.trace_path),
        )
        runtime = AgentRuntime(loop, self.state, interactive_approvals=False)
        service = AgentService(tool_registry_factory=lambda _: tools,
                               save_session=self.store.save, event_store=self.events)
        service.register(runtime)
        self.addCleanup(service.close)
        subscription, _ = service.subscribe_events(session_ids=(self.state.session_id,))
        return service, subscription

    def terminal(self, service, subscription):
        while True:
            event = service.next_event(subscription, timeout=3)
            if isinstance(event, (TurnCompletedEvent, TurnFailedEvent)):
                service.wait(self.state.session_id)
                return event

    def test_terminal_notification_follows_committed_state(self):
        service, subscription = self.service([response("Committed answer")])
        service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Hello"))
        terminal = self.terminal(service, subscription)
        self.assertIsInstance(terminal, TurnCompletedEvent)
        restored = self.store.load(self.state.session_id)
        self.assertIsNone(restored.pending_turn)
        self.assertEqual(restored.messages[-1].content, "Committed answer")
        self.assertEqual(self.events.replay(self.state.session_id)[-1].event_id, terminal.event_id)

    def test_failed_checkpoint_prevents_tool_side_effect(self):
        call = ToolCall("patch-1", "apply_patch", {"patch":
            "*** Begin Patch\n*** Add File: never.txt\n+must not run\n*** End Patch"})
        service, subscription = self.service([response(calls=[call]), response("Done")])
        original = service._save_session
        def fail_before_tool(state):
            if state.messages and getattr(state.messages[-1], "tool_calls", ()):
                raise OSError("injected disk full")
            original(state)
        with patch.object(service, "_save_session", side_effect=fail_before_tool):
            service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Create file"))
            terminal = self.terminal(service, subscription)
        self.assertIsInstance(terminal, TurnFailedEvent)
        self.assertFalse(terminal.durable)
        self.assertFalse((self.root / "never.txt").exists())
        self.assertEqual(len(self.provider.requests), 1)
        with self.assertRaises(StorageError):
            service.dispatch(StartTurnCommand(session_id=self.state.session_id, retry=True))
        restored = self.store.load(self.state.session_id)
        self.assertEqual(restored.pending_turn.pause_reason, "recovery_required")

    def test_control_plane_failure_rolls_back_state_and_projection(self):
        service, _ = self.service([])
        before = service.projection(self.state.session_id)
        with patch.object(self.events, "append", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                service.dispatch(RenameSessionCommand(session_id=self.state.session_id, title="Unsaved"))
        self.assertEqual(self.store.load(self.state.session_id).title, "")
        self.assertEqual(service.projection(self.state.session_id), before)

    def test_stream_sequence_reservation_survives_restart(self):
        service, subscription = self.service([response("First answer")])
        service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Hello"))
        first = self.terminal(service, subscription)
        service.close()
        self.state = self.store.load(self.state.session_id)
        restored_service, restored_subscription = self.service([response("Second answer")])
        restored_service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Again"))
        second = self.terminal(restored_service, restored_subscription)
        self.assertGreater(second.sequence, first.sequence)
        self.assertGreater(second.sequence, 1024)

    def test_result_commit_failure_keeps_unknown_attempt_and_requires_confirmation(self):
        call = ToolCall("patch-unknown", "apply_patch", {"patch":
            "*** Begin Patch\n*** Add File: changed.txt\n+created once\n*** End Patch"})
        service, subscription = self.service([response(calls=[call]), response("Done")])
        original = self.events.append
        def fail_result(event):
            if event.event_type.value == "item.completed" and getattr(event, "item_type", "") == "tool_call":
                raise OSError("result commit failed")
            original(event)
        with patch.object(self.events, "append", side_effect=fail_result):
            service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Create file"))
            terminal = self.terminal(service, subscription)
        self.assertIsInstance(terminal, TurnFailedEvent)
        self.assertEqual((self.root / "changed.txt").read_text(), "created once\n")
        service.close()
        self.state = self.store.load(self.state.session_id)
        self.assertEqual(self.state.pending_turn.pause_reason, "recovery_required")
        attempts = self.state.pending_turn.pause_details["unknown_attempts"]
        self.assertEqual(attempts[0]["call_id"], "patch-unknown")
        restored, subscription = self.service([response("Inspect before further edits")])
        with self.assertRaisesRegex(ValueError, "confirmation required"):
            restored.dispatch(StartTurnCommand(session_id=self.state.session_id, retry=True))
        restored.dispatch(StartTurnCommand(session_id=self.state.session_id, retry=True, recovery_confirmed=True))
        self.assertIsInstance(self.terminal(restored, subscription), TurnCompletedEvent)
        observations = [m for m in self.state.messages if getattr(m, "tool_call_id", "") == "patch-unknown"]
        self.assertEqual(len(observations), 1)
        self.assertEqual(json.loads(observations[0].content)["metadata"]["status"], "unknown")
        self.assertEqual((self.root / "changed.txt").read_text(), "created once\n")

    def test_process_death_after_patch_recovers_without_automatic_replay(self):
        script = """
import os, sys
from pathlib import Path
from my_agent.agent_loop import AgentLoop
from my_agent.event_store import EventStore
from my_agent.messages import ToolCall
from my_agent.model_invoker import ModelInvoker
from my_agent.permissions import PermissionPolicy
from my_agent.protocol import StartTurnCommand
from my_agent.provider import ModelResponse, ModelMetadata, ScriptedProviderAdapter
from my_agent.runtime import AgentRuntime
from my_agent.service import AgentService
from my_agent.session import SessionStore
from my_agent.tools import ApplyPatchTool, ToolRegistry
from my_agent.trace import TraceRecorder
from my_agent.workspace import Workspace
store = SessionStore(sys.argv[1])
state = store.load(sys.argv[2])
class DyingPatch(ApplyPatchTool):
    def run(self, arguments, ctx):
        result = super().run(arguments, ctx)
        assert result.ok
        os._exit(29)
call = ToolCall('crash-call', 'apply_patch', {'patch': '*** Begin Patch\\n*** Add File: crash.txt\\n+one write\\n*** End Patch'})
provider = ScriptedProviderAdapter([ModelResponse.from_parts(tool_calls=[call], metadata=ModelMetadata(provider_id='scripted', model_id='scripted-model'))])
tools = ToolRegistry([DyingPatch()])
loop = AgentLoop(workspace=Workspace(state.workspace), model=ModelInvoker(provider), tools=tools,
    permissions=PermissionPolicy(profile_id='trusted'), trace=TraceRecorder(state.trace_path))
service = AgentService(tool_registry_factory=lambda _: tools, save_session=store.save, event_store=EventStore(store.root))
service.register(AgentRuntime(loop, state, interactive_approvals=False))
service.dispatch(StartTurnCommand(session_id=state.session_id, user_input='Create file'))
service.wait(state.session_id, timeout=5)
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.store.root), self.state.session_id],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 29, result.stderr)
        self.state = self.store.load(self.state.session_id)
        self.assertEqual(self.state.pending_turn.pause_reason, "recovery_required")
        self.assertEqual(self.state.pending_turn.pause_details["unknown_attempts"][0]["call_id"], "crash-call")
        restored, _ = self.service([])
        self.assertEqual(self.provider.requests, [])
        self.assertEqual((self.root / "crash.txt").read_text(), "one write\n")
