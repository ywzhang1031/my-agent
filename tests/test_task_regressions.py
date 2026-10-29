"""Fixed model actions + real isolated tools: harness regression, not model capability eval."""
import shutil
import sys
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from my_agent.agent_loop import AgentLoop
from my_agent.cli import build_tool_registry
from my_agent.control import ApprovalDecision
from my_agent.event_store import EventStore
from my_agent.messages import ToolCall
from my_agent.model_invoker import ModelInvoker
from my_agent.permissions import PermissionPolicy
from my_agent.protocol import ApprovalRequestedEvent, ResolveApprovalCommand, StartTurnCommand, TurnCompletedEvent, TurnFailedEvent
from my_agent.provider import ModelMetadata, ModelResponse, ScriptedProviderAdapter, StopReason
from my_agent.runtime import AgentRuntime
from my_agent.sandbox import DANGER_FULL_ACCESS, SandboxPolicy
from my_agent.service import AgentService
from my_agent.session import SessionStore
from my_agent.trace import TraceRecorder
from my_agent.verification import read_verification
from my_agent.workspace import Workspace


class TaskRegressionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="agent-task-regression-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, {"PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")})
        environment.start()
        self.addCleanup(environment.stop)
        shutil.copytree(Path(__file__).parent / "fixtures" / "verification", self.root, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        self.store = SessionStore(self.root / ".my-agent" / "sessions")
        self.state = self.store.create(self.root)
        self.call_count = 0

    def response(self, text="", calls=()):
        return ModelResponse.from_parts(text=text, tool_calls=calls,
            stop_reason=StopReason.TOOL_CALLS if calls else StopReason.STOP,
            metadata=ModelMetadata(provider_id="scripted", model_id="scripted-model"))

    def tool(self, name, arguments):
        self.call_count += 1
        return self.response(calls=[ToolCall(f"call-{self.call_count}", name, arguments)])

    def edit(self, *, regress=False):
        old = "    return a + b  # fixed" if regress else "    return a - b"
        new = "    return a - b  # regression" if regress else "    return a + b  # fixed"
        return self.tool("apply_patch", {"patch": f"*** Begin Patch\n*** Update File: calc.py\n@@\n-{old}\n+{new}\n*** End Patch"})

    def check(self):
        return self.tool("exec_command", {"argv": ["python3", "-m", "unittest", "discover", "-s", "."], "cwd": "."})

    def open_service(self, replies, *, supervised=False):
        provider = ScriptedProviderAdapter(replies)
        mode = "supervised" if supervised else "trusted"
        loop = AgentLoop(workspace=Workspace(self.root), model=ModelInvoker(provider),
                         tools=build_tool_registry(), permissions=PermissionPolicy(profile_id=mode),
                         trace=TraceRecorder(self.state.trace_path),
                         sandbox_policy=SandboxPolicy(DANGER_FULL_ACCESS, self.root))
        events = EventStore(self.store.root)
        service = AgentService(tool_registry_factory=build_tool_registry, save_session=self.store.save, event_store=events)
        service.register(AgentRuntime(loop, self.state, interactive_approvals=supervised))
        subscription, _ = service.subscribe_events(session_ids=[self.state.session_id])
        self.addCleanup(service.close)
        return service, subscription, events, provider

    def drive(self, service, subscription, *, retry=False, recovery_confirmed=False, prompt="Fix addition and verify it"):
        service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input=prompt,
                                         retry=retry, recovery_confirmed=recovery_confirmed))
        while True:
            event = service.next_event(subscription_id=subscription, timeout=10)
            if isinstance(event, ApprovalRequestedEvent):
                service.dispatch(ResolveApprovalCommand(session_id=self.state.session_id,
                    request_id=event.request_id, decision=ApprovalDecision.DENY))
            if isinstance(event, (TurnCompletedEvent, TurnFailedEvent)):
                service.wait(self.state.session_id, timeout=5)
                return event

    def view(self):
        return read_verification(self.store.database, self.state.session_id)

    def test_fix_failing_test_then_validate(self):
        service, sub, _, _ = self.open_service([self.check(), self.edit(), self.check(), self.response("Done")])
        self.assertIsInstance(self.drive(service, sub), TurnCompletedEvent)
        view = self.view()
        self.assertEqual(view["status"], "checks_passed")
        self.assertEqual([check["status"] for check in view["checks"]], ["failed", "passed"])
        self.assertIn("return a + b", (self.root / "calc.py").read_text())

    def test_edit_without_validation_can_end_but_is_unverified(self):
        service, sub, _, provider = self.open_service([self.edit(), self.response("Done")])
        self.assertIsInstance(self.drive(service, sub), TurnCompletedEvent)
        self.assertEqual(self.view()["status"], "unverified")
        self.assertEqual(len(provider.requests), 2)

    def test_edit_after_successful_test_marks_evidence_stale(self):
        service, sub, _, _ = self.open_service([self.edit(), self.check(), self.edit(regress=True), self.response("Done")])
        self.assertIsInstance(self.drive(service, sub), TurnCompletedEvent)
        self.assertEqual(self.view()["status"], "stale")
        self.assertEqual(self.view()["checks"][0]["exit_code"], 0)
        self.assertIn("return a - b", (self.root / "calc.py").read_text())

    def test_rejected_permission_does_not_modify_workspace(self):
        original = (self.root / "calc.py").read_bytes()
        service, sub, _, _ = self.open_service([self.edit(), self.response("Approval denied")], supervised=True)
        self.assertIsInstance(self.drive(service, sub), TurnCompletedEvent)
        self.assertEqual((self.root / "calc.py").read_bytes(), original)
        self.assertEqual(self.view()["changed_files"], [])
        self.assertNotEqual(self.view()["status"], "checks_passed")

    def test_missing_commit_requires_recovery_and_does_not_replay_patch(self):
        service, sub, events, _ = self.open_service([self.edit(), self.response("Unused")])
        append = events.append
        def fail_result(event):
            if event.event_type.value == "item.completed" and getattr(event, "item_type", "") == "tool_call":
                raise OSError("injected result commit failure")
            append(event)
        with patch.object(events, "append", side_effect=fail_result):
            self.assertIsInstance(self.drive(service, sub), TurnFailedEvent)
        service.close()
        self.state = self.store.load(self.state.session_id)
        self.assertEqual(self.state.pending_turn.pause_reason, "recovery_required")
        self.assertEqual(self.view()["status"], "unknown")
        modified = (self.root / "calc.py").stat().st_mtime_ns
        restored, sub, _, provider = self.open_service([self.response("Inspect the existing edit before continuing")])
        with self.assertRaises(ValueError):
            self.drive(restored, sub, retry=True)
        self.assertIsInstance(self.drive(restored, sub, retry=True, recovery_confirmed=True), TurnCompletedEvent)
        self.assertEqual((self.root / "calc.py").stat().st_mtime_ns, modified)
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(self.view()["status"], "unknown")

    def test_readonly_answer_needs_no_validation_command(self):
        original = (self.root / "calc.py").read_bytes()
        service, sub, _, provider = self.open_service([
            self.tool("read_file", {"path": "calc.py"}), self.response("The function currently subtracts")])
        self.assertIsInstance(self.drive(service, sub, prompt="Explain the function without editing or running tests"), TurnCompletedEvent)
        self.assertEqual(self.view()["status"], "no_evidence")
        self.assertEqual((self.root / "calc.py").read_bytes(), original)
        self.assertEqual(len(provider.requests), 2)
