import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from my_agent.agent_loop import AgentLoop
from my_agent.cli import build_tool_registry, main
from my_agent.config import load_runtime_config
from my_agent.context import ContextManager, ContextWindowError, estimate_request_tokens
from my_agent.database import StorageError
from my_agent.event_store import EventStore
from my_agent.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from my_agent.model_invoker import ModelInvoker, RetryPolicy
from my_agent.modes import PLAN_MODE
from my_agent.permissions import PermissionPolicy
from my_agent.protocol import StartTurnCommand, TurnCompletedEvent, TurnFailedEvent, TurnStartedEvent, command_from_dict, command_to_dict, event_from_dict, event_to_dict
from my_agent.provider import ModelMetadata, ModelResponse, ScriptedProviderAdapter, StopReason
from my_agent.runtime import AgentRuntime
from my_agent.service import AgentService
from my_agent.session import SessionStore
from my_agent.skills import MAX_SKILL_BYTES, SkillSettings, TurnSkills, discover_skills, load_skill, validate_skill_names
from my_agent.tools import LoadSkillTool, ToolRegistry
from my_agent.trace import TraceRecorder
from my_agent.workspace import Workspace, WorkspaceError


def response(text="Done", name=None):
    return ModelResponse.from_parts(text=text if name is None else "", tool_calls=[] if name is None else [ToolCall("skill-1", "load_skill", {"name": name})],
        stop_reason=StopReason.STOP if name is None else StopReason.TOOL_CALLS,
        metadata=ModelMetadata(provider_id="scripted", model_id="scripted-model"))


def content(request):
    return "\n".join(message.content for message in request.messages)


class SkillRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.store = SessionStore(self.root / ".my-agent" / "sessions")
        self.state = self.store.create(self.root)
        self.path = self.skill(self.root, "test-review", body="PRIVATE_REVIEW_WORKFLOW")

    def skill(self, root, name, directory=".agents/skills", body="Workflow", description="Review tests for edge cases"):
        path = root / directory / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n", encoding="utf-8")
        return path

    def loop(self, replies, **kwargs):
        provider = ScriptedProviderAdapter(replies)
        loop = AgentLoop(workspace=Workspace(self.root), model=ModelInvoker(provider, RetryPolicy(max_attempts=1)),
                         tools=ToolRegistry([LoadSkillTool()]), permissions=PermissionPolicy(),
                         trace=TraceRecorder(self.state.trace_path), **kwargs)
        return loop, provider

    def test_model_load_is_lazy_counted_once_and_has_provenance(self):
        loop, provider = self.loop([response(name="test-review"), response()])
        loop.run_turn(self.state, "Review the tests")
        self.assertIn("Review tests for edge cases", content(provider.requests[0]))
        self.assertNotIn("PRIVATE_REVIEW_WORKFLOW", content(provider.requests[0]))
        self.assertEqual(content(provider.requests[1]).count("PRIVATE_REVIEW_WORKFLOW"), 1)
        self.assertNotIn("PRIVATE_REVIEW_WORKFLOW", provider.requests[1].system_prompt)
        self.assertTrue(all(message.skills is None for message in provider.requests[1].messages if isinstance(message, UserMessage)))
        result = next(message for message in self.state.messages if isinstance(message, ToolResultMessage))
        self.assertNotIn("PRIVATE_REVIEW_WORKFLOW", result.content)
        manifest = json.loads(result.content)["metadata"]["skill"]
        self.assertEqual(manifest["reason"], "model")
        self.assertEqual(manifest["path"], str(self.path))
        self.assertEqual(manifest["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        snapshot = loop.inspect_context(self.state)
        self.assertGreater(snapshot.breakdown.skill_catalog_tokens, 0)
        self.assertGreater(snapshot.breakdown.skill_body_tokens, 0)
        self.assertEqual(snapshot.estimated_tokens, estimate_request_tokens(snapshot.system_prompt, snapshot.messages, loop.tools.specs()))

    def test_manual_load_precedes_first_request_and_is_turn_scoped(self):
        loop, provider = self.loop([response(), response()])
        loop.run_turn(self.state, "Review", skills=("test-review",))
        self.assertIn("PRIVATE_REVIEW_WORKFLOW", content(provider.requests[0]))
        self.assertEqual(self.state.current_skills().loaded[0].reason, "manual")
        self.assertFalse(any(isinstance(message, ToolResultMessage) for message in self.state.messages))
        loop.run_turn(self.state, "Unrelated question")
        self.assertNotIn("PRIVATE_REVIEW_WORKFLOW", content(provider.requests[1]))
        self.assertEqual(self.state.current_skills().loaded, [])
        self.assertIn("PRIVATE_REVIEW_WORKFLOW", self.state.messages[0].skills.loaded[0].body)

    def test_pending_turn_restores_loaded_body_even_after_file_deletion(self):
        loop, _ = self.loop([response(name="test-review")], max_steps=1)
        result = loop.run_turn(self.state, "Review")
        self.assertEqual(result.stopped_by, "continuation_required")
        frozen = self.state.current_skills().to_dict()
        self.store.save(self.state)
        self.path.unlink()
        self.skill(self.root, "new-skill")
        self.state = self.store.load(self.state.session_id)
        loop, provider = self.loop([response()])
        loop.retry_turn(self.state)
        self.assertIn("PRIVATE_REVIEW_WORKFLOW", content(provider.requests[0]))
        self.assertNotIn("new-skill", content(provider.requests[0]))
        self.assertEqual(self.state.current_skills().to_dict(), frozen)

    def test_loaded_skill_is_cached_but_unloaded_metadata_changes_are_rejected(self):
        skills = TurnSkills(discover_skills(self.root))
        first = skills.load("test-review", "manual")
        self.path.write_text("changed")
        self.assertIs(skills.load("test-review", "model"), first)
        self.assertEqual(len(skills.loaded), 1)
        self.assertEqual(first.reason, "manual")
        with self.assertRaises(ValueError):
            load_skill(skills.catalog.skills[0], "model")

    def test_metadata_change_after_turn_start_is_a_tool_error(self):
        loop, provider = self.loop([response(name="test-review"), response()])
        def mutate(event):
            if event.kind == "tool_started":
                self.skill(self.root, "test-review", body="NEW_BODY", description="Changed description")
        loop.run_turn(self.state, "Review", event_handler=mutate)
        result = next(message for message in self.state.messages if isinstance(message, ToolResultMessage))
        self.assertIn("metadata changed", result.content)
        self.assertEqual(self.state.current_skills().loaded, [])
        self.assertNotIn("NEW_BODY", content(provider.requests[1]))

    def test_unknown_path_or_oversized_skill_cannot_expand_read_access(self):
        skills = TurnSkills(discover_skills(self.root))
        for name in ("../secret", str(self.root / "secret"), "unknown"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                skills.load(name, "model")
        self.path.write_bytes(self.path.read_bytes() + b"x" * MAX_SKILL_BYTES)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            skills.load("test-review", "model")
        self.assertEqual(skills.loaded, [])

    def test_body_validation_rejects_empty_invalid_utf8_fifo_and_links(self):
        for invalid in (b"", b"\xff"):
            self.skill(self.root, "test-review", body="")
            catalog = discover_skills(self.root)
            with self.path.open("ab") as handle:
                handle.write(invalid)
            with self.assertRaises(ValueError):
                load_skill(catalog.skills[0], "manual")
        self.skill(self.root, "test-review")
        catalog = discover_skills(self.root)
        self.path.unlink()
        os.mkfifo(self.path)
        with self.assertRaisesRegex(ValueError, "regular"):
            load_skill(catalog.skills[0], "model")
        self.path.unlink()
        target = self.root / "secret"
        target.write_text("secret")
        self.path.symlink_to(target)
        with self.assertRaises(ValueError):
            load_skill(catalog.skills[0], "model")

    def test_skill_budget_failure_does_not_keep_rejected_body(self):
        self.skill(self.root, "test-review", body="x" * 8000)
        manager = ContextManager(context_window_tokens=1500, reserve_output_tokens=200)
        loop, provider = self.loop([response(name="test-review"), response()], context_manager=manager)
        loop.run_turn(self.state, "Review")
        self.assertEqual(self.state.current_skills().loaded, [])
        self.assertIn("context budget", content(provider.requests[1]))
        loop, provider = self.loop([response()], context_manager=manager)
        with self.assertRaises(ContextWindowError):
            loop.run_turn(self.state, "Manual review", skills=("test-review",))
        self.assertEqual(provider.requests, [])

    def test_compaction_keeps_current_body_without_summarizing_old_body(self):
        old = TurnSkills(discover_skills(self.root))
        old.load("test-review", "manual")
        self.skill(self.root, "test-review", body="CURRENT_BODY")
        current = TurnSkills(discover_skills(self.root))
        current.load("test-review", "manual")
        self.state.messages = [UserMessage("old question " * 500, skills=old), AssistantMessage("answer " * 500), UserMessage("new", skills=current)]
        manager = ContextManager(context_window_tokens=1800, reserve_output_tokens=200, summary_tokens=100)
        snapshot = manager.prepare(state=self.state, tools=[], system_prompt="System")
        self.assertGreater(snapshot.summarized_messages, 0)
        self.assertIn("CURRENT_BODY", " ".join(message.content for message in snapshot.messages))
        self.assertNotIn("PRIVATE_REVIEW_WORKFLOW", snapshot.system_prompt)

    def test_manual_validation_and_command_roundtrip(self):
        for names in ("test-review", ["../outside"], ["test-review"] * 2, [str(i) for i in range(9)], [True]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                validate_skill_names(names)
        command = StartTurnCommand(session_id="test", user_input="Review", skills=("test-review",))
        self.assertEqual(command_from_dict(command_to_dict(command)), command)

    def test_invocation_flags_are_enforced_not_just_hidden(self):
        text = self.path.read_text().replace("description:", "disable-model-invocation: true\ndescription:")
        self.path.write_text(text)
        skills = TurnSkills(discover_skills(self.root))
        self.assertEqual(skills.catalog_prompt(), "")
        with self.assertRaisesRegex(ValueError, "manual-only"):
            skills.load("test-review", "model")
        self.assertEqual(skills.load("test-review", "manual").reason, "manual")
        self.path.write_text(text.replace("disable-model-invocation: true", "user-invocable: false"))
        skills = TurnSkills(discover_skills(self.root))
        self.assertIn("test-review", skills.catalog_prompt())
        with self.assertRaisesRegex(ValueError, "manual invocation"):
            skills.load("test-review", "manual")
        self.assertEqual(skills.load("test-review", "model").reason, "model")
        self.path.write_text(text.replace("true", "'true'"))
        self.assertEqual(discover_skills(self.root).skills, ())

    def test_loaded_skill_count_is_bounded_without_eviction(self):
        for index in range(9):
            self.skill(self.root, f"skill-{index}")
        skills = TurnSkills(discover_skills(self.root))
        for index in range(8):
            skills.load(f"skill-{index}", "model")
        with self.assertRaisesRegex(ValueError, "At most 8"):
            skills.load("skill-8", "model")
        self.assertEqual(len(skills.loaded), 8)

    def test_plan_keeps_write_tools_unavailable_after_skill_load(self):
        self.skill(self.root, "test-review", body="Use apply_patch and exec_command to change files.")
        loop, provider = self.loop([response()], mode=PLAN_MODE)
        loop.tools = build_tool_registry(PLAN_MODE)
        loop.run_turn(self.state, "Plan only", skills=("test-review",))
        names = {tool.name for tool in provider.requests[0].tools}
        self.assertIn("load_skill", names)
        self.assertNotIn("apply_patch", names)
        self.assertNotIn("exec_command", names)

    def test_cross_product_sources_precedence_and_external_body_loading(self):
        home = self.root / "home"
        workspace = home / "projects" / "repo"
        workspace.mkdir(parents=True)
        self.skill(home / "projects", "parent-only")
        self.skill(home, "personal-only", directory=".codex/skills", body="PERSONAL_BODY")
        self.skill(home / "other-project", "unrelated")
        self.skill(home, "same-name", directory=".claude/skills")
        self.skill(workspace, "same-name", directory=".claude/skills", body="PROJECT_CLAUDE")
        winning = self.skill(workspace, "same-name", body="PROJECT_AGENTS")
        extra = self.root / "extra"
        self.skill(extra, "extra-only", directory="")
        settings = SkillSettings(True, True, (str(extra),))
        with patch.dict(os.environ, {"CODEX_HOME": str(home / ".codex"), "XDG_CONFIG_HOME": str(home / ".config")}):
            catalog = discover_skills(workspace, settings, home=home)
        self.assertEqual({item.name for item in catalog.skills}, {"parent-only", "personal-only", "same-name", "extra-only"})
        by_name = {item.name: item for item in catalog.skills}
        self.assertEqual(by_name["same-name"].path, str(winning))
        self.assertEqual(by_name["personal-only"].source.scope, "personal")
        self.assertIn("PERSONAL_BODY", load_skill(by_name["personal-only"], "manual").body)
        self.assertEqual(sum("Shadowed by" in issue.message for issue in catalog.diagnostics), 2)
        with self.assertRaises(WorkspaceError):
            Workspace(workspace).resolve(by_name["personal-only"].path)

    def test_config_can_disable_shared_discovery_and_add_roots(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps({"skills": {"discover_parents": False, "discover_personal": False, "paths": [str(self.root)]}}))
        config = load_runtime_config(config_path, self.root / "credentials.json")
        self.assertEqual(config.skills, SkillSettings(False, False, (str(self.root),)))
        for raw in ({"unknown": True}, {"paths": ["relative/path"]}, {"discover_personal": "false"}, {"paths": "bad"}):
            config_path.write_text(json.dumps({"skills": raw}))
            with self.assertRaises(ValueError):
                load_runtime_config(config_path, self.root / "credentials.json")

    def test_manual_error_is_visible_before_any_model_call(self):
        loop, provider = self.loop([response()])
        with self.assertRaisesRegex(ContextWindowError, "not in this turn's catalog"):
            loop.run_turn(self.state, "Review", skills=("missing-skill",))
        self.assertEqual(provider.requests, [])
        self.assertIsNotNone(self.state.pending_turn)
        self.assertIsNotNone(self.state.current_skills().error)

    def test_checkpoint_before_model_and_replayable_source_records(self):
        loop, provider = self.loop([response(name="test-review"), response()])
        events = EventStore(self.store.root)
        service = AgentService(tool_registry_factory=lambda _: loop.tools, save_session=self.store.save, event_store=events)
        service.register(AgentRuntime(loop, self.state, interactive_approvals=False))
        self.addCleanup(service.close)
        subscription, _ = service.subscribe_events(session_ids=(self.state.session_id,))
        stream = provider.stream_once
        calls = []
        def verify(request, cancel_event=None):
            saved = self.store.load(self.state.session_id, recover=False).current_skills()
            self.assertIsNotNone(saved)
            self.assertEqual(len(saved.loaded), len(calls))
            calls.append(request)
            return stream(request, cancel_event)
        with patch.object(provider, "stream_once", side_effect=verify):
            service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Review"))
            while True:
                event = service.next_event(subscription, timeout=5)
                if isinstance(event, (TurnCompletedEvent, TurnFailedEvent)):
                    self.assertIsInstance(event, TurnCompletedEvent)
                    break
            service.wait(self.state.session_id)
        self.assertEqual(len(calls), 2)
        replay = events.replay(self.state.session_id)
        started = next(event for event in replay if isinstance(event, TurnStartedEvent))
        self.assertEqual(event_from_dict(event_to_dict(started)), started)
        self.assertNotIn("PRIVATE_REVIEW_WORKFLOW", json.dumps(started.skills))
        traces = [json.loads(line) for line in self.state.trace_path.read_text().splitlines()]
        request = [item for item in traces if item["event"] == "model_request"][-1]
        self.assertEqual(request["context"]["skills"]["loaded"][0]["reason"], "model")

    def test_checkpoint_failure_never_sends_uncommitted_manual_body(self):
        loop, provider = self.loop([response()])
        service = AgentService(tool_registry_factory=lambda _: loop.tools, save_session=self.store.save, event_store=EventStore(self.store.root))
        service.register(AgentRuntime(loop, self.state, interactive_approvals=False))
        self.addCleanup(service.close)
        with patch.object(service, "_save_session", side_effect=OSError("injected failure")):
            with self.assertRaises(StorageError):
                service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Review", skills=("test-review",)))
            service.wait(self.state.session_id)
        self.assertEqual(provider.requests, [])
        self.assertIsNone(self.store.load(self.state.session_id).current_skills())

    def test_tool_checkpoint_failure_stops_before_uncommitted_body_reaches_model(self):
        loop, provider = self.loop([response(name="test-review"), response()])
        service = AgentService(tool_registry_factory=lambda _: loop.tools, save_session=self.store.save, event_store=EventStore(self.store.root))
        service.register(AgentRuntime(loop, self.state, interactive_approvals=False))
        self.addCleanup(service.close)
        save = self.store.save
        def fail_loaded(state):
            if state.current_skills() and state.current_skills().loaded:
                raise OSError("injected skill checkpoint failure")
            save(state)
        with patch.object(service, "_save_session", side_effect=fail_loaded):
            service.dispatch(StartTurnCommand(session_id=self.state.session_id, user_input="Review"))
            service.wait(self.state.session_id)
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(self.store.load(self.state.session_id, recover=False).current_skills().loaded, [])

    def test_cli_selection_is_forwarded_once_and_can_be_cleared(self):
        class Editor:
            supports_interactive_selection = True
            def __init__(self):
                self.inputs = iter(["/skills", "Review", "Unrelated", "/skills test-review", "/skills --clear", "Cleared", "/exit"])
                self.options = []
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def read(self, prompt):
                return next(self.inputs)
            def select(self, prompt, options, **kwargs):
                self.options = list(options)
                return "test-review"
        editor = Editor()
        provider = ScriptedProviderAdapter([response(), response(), response()])
        config = self.root / "config.json"
        config.write_text(json.dumps({"skills": {"discover_parents": False, "discover_personal": False}}))
        with patch("my_agent.cli.LineEditor", return_value=editor), patch("my_agent.cli._build_model", return_value=ModelInvoker(provider)), redirect_stdout(io.StringIO()):
            result = main(["chat", "--workspace", str(self.root), "--sessions-dir", str(self.root / "cli-sessions"),
                           "--config", str(config), "--credentials", str(self.root / "credentials.json")])
        self.assertEqual(result, 0)
        self.assertIn("test-review", [option.value for option in editor.options])
        self.assertIn("PRIVATE_REVIEW_WORKFLOW", content(provider.requests[0]))
        self.assertNotIn("PRIVATE_REVIEW_WORKFLOW", content(provider.requests[1]))
        self.assertNotIn("PRIVATE_REVIEW_WORKFLOW", content(provider.requests[2]))
