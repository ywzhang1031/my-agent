import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from my_agent.agent_loop import AgentLoop
from my_agent.cli import build_tool_registry
from my_agent.control import raise_if_cancelled
from my_agent.goals import (
    MAX_GOAL_OBJECTIVE_CHARS,
    MAX_GOAL_STEP_CHARS,
    MAX_GOAL_STEPS,
    goal_from_dict,
    update_goal,
)
from my_agent.messages import ToolCall, ToolResultMessage
from my_agent.model_invoker import ModelInvoker, RetryPolicy
from my_agent.modes import PLAN_MODE
from my_agent.permissions import PermissionPolicy
from my_agent.provider import ModelMetadata, ModelResponse, ScriptedProviderAdapter, StopReason
from my_agent.runtime import AgentRuntime, StartTurn
from my_agent.session import SessionStore
from my_agent.tools import ToolContext, UpdateGoalTool
from my_agent.trace import TraceRecorder
from my_agent.trajectory import read_jsonl_trace
from my_agent.workspace import Workspace


def goal_arguments(status="in_progress", objective="Inspect the repository"):
    return {
        "objective": objective,
        "steps": [{"description": "Inspect files", "status": status}],
    }


def model_response(**kwargs):
    return ModelResponse.from_parts(
        metadata=ModelMetadata(provider_id="scripted", model_id="scripted-model"),
        **kwargs,
    )


class GoalValidationTests(unittest.TestCase):
    def test_tool_rejects_invalid_json_without_mutating_goal(self):
        original = update_goal(None, goal_arguments())
        state = SimpleNamespace(goal=original)
        invalid = [None, [], "goal", 42, {}, {"objective": "Missing steps"}]
        for objective in (None, [], {}, 42, True, "", "  ", "x" * (MAX_GOAL_OBJECTIVE_CHARS + 1)):
            invalid.append({**goal_arguments(), "objective": objective})
        for steps in (None, {}, "steps", [], [None], [{}]):
            invalid.append({**goal_arguments(), "steps": steps})
        for status in ([], {}, None, 1, True, "", "blocked", "COMPLETED"):
            invalid.append(goal_arguments(status))
        for description in (None, [], {}, 42, "", "  ", "x" * (MAX_GOAL_STEP_CHARS + 1)):
            invalid.append({
                **goal_arguments(),
                "steps": [{"description": description, "status": "pending"}],
            })
        invalid.extend([
            {**goal_arguments(), "status": "completed"},
            {**goal_arguments(), "goal_id": "model-supplied-id"},
            {**goal_arguments(), "steps": [
                {"description": "Inspect", "status": "pending", "extra": True}
            ]},
            {**goal_arguments(), "steps": [
                {"description": "Inspect", "status": "pending"},
                {"description": " Inspect ", "status": "pending"},
            ]},
            {**goal_arguments(), "steps": [
                {"description": str(i), "status": "pending"}
                for i in range(MAX_GOAL_STEPS + 1)
            ]},
            {**goal_arguments(), "steps": [
                {"description": "Inspect", "status": "in_progress"},
                {"description": "Verify", "status": "in_progress"},
            ]},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            ctx = ToolContext(
                workspace=Workspace(tmp),
                permissions=PermissionPolicy(),
                goal_updater=lambda arguments: AgentLoop._update_goal(state, arguments),
            )
            for arguments in invalid:
                with self.subTest(arguments=arguments):
                    state.goal = original
                    result = UpdateGoalTool().run(arguments, ctx)
                    self.assertFalse(result.ok)
                    self.assertEqual(result.metadata["status"], "invalid_goal")
                    self.assertTrue(result.stderr)
                    self.assertIs(state.goal, original)

    def test_goal_identity_and_completion_are_derived_from_valid_snapshots(self):
        initial = update_goal(None, goal_arguments())
        completed = update_goal(initial, goal_arguments("completed"))
        reopened = update_goal(completed, goal_arguments("pending"))
        replacement = update_goal(reopened, goal_arguments(objective="New objective"))

        self.assertEqual(completed.status, "completed")
        self.assertEqual(reopened.status, "active")
        self.assertEqual(initial.goal_id, completed.goal_id)
        self.assertEqual(initial.goal_id, reopened.goal_id)
        self.assertNotEqual(initial.goal_id, replacement.goal_id)
        self.assertEqual(initial.steps[0].status, "in_progress")
        self.assertEqual(goal_from_dict(completed.to_dict()), completed)

    def test_goal_accepts_limits_and_normalizes_whitespace(self):
        arguments = {
            "objective": " " + "x" * MAX_GOAL_OBJECTIVE_CHARS + " ",
            "steps": [
                {"description": str(i).ljust(MAX_GOAL_STEP_CHARS, "x"), "status": "pending"}
                for i in range(MAX_GOAL_STEPS)
            ],
        }
        goal = update_goal(None, arguments)
        self.assertEqual(len(goal.objective), MAX_GOAL_OBJECTIVE_CHARS)
        self.assertEqual(len(goal.steps), MAX_GOAL_STEPS)
        self.assertEqual(goal.status, "active")
        self.assertEqual(goal_from_dict(goal.to_dict()), goal)
        self.assertEqual(update_goal(goal, arguments).goal_id, goal.goal_id)

    def test_tool_schema_advertises_the_runtime_limits(self):
        properties = UpdateGoalTool.parameters["properties"]
        self.assertEqual(properties["objective"]["maxLength"], MAX_GOAL_OBJECTIVE_CHARS)
        self.assertEqual(properties["steps"]["maxItems"], MAX_GOAL_STEPS)
        step_schema = properties["steps"]["items"]
        self.assertEqual(step_schema["properties"]["description"]["maxLength"], MAX_GOAL_STEP_CHARS)
        self.assertFalse(step_schema["additionalProperties"])
        self.assertFalse(UpdateGoalTool.parameters["additionalProperties"])

    def test_persisted_goal_rejects_invalid_fields_and_inconsistent_status(self):
        snapshot = update_goal(None, goal_arguments()).to_dict()
        invalid = [[], "goal", {key: value for key, value in snapshot.items() if key != "status"}]
        for key, value in (
            ("goal_id", "  "), ("updated_at", "  "),
            ("objective", "x" * (MAX_GOAL_OBJECTIVE_CHARS + 1)),
            ("status", []), ("status", {}), ("status", "completed"),
            ("steps", [{"description": "Inspect", "status": []}]),
        ):
            invalid.append({**snapshot, key: value})
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                goal_from_dict(data)
        self.assertIsNone(goal_from_dict(None))


class GoalLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        self.store = SessionStore(root / "sessions")
        self.state = self.store.create(self.workspace)
        self.state.goal = update_goal(None, goal_arguments())

    def make_loop(self, provider):
        return AgentLoop(
            workspace=Workspace(self.workspace),
            model=ModelInvoker(provider, RetryPolicy(max_attempts=1)),
            tools=build_tool_registry(),
            permissions=PermissionPolicy(),
            trace=TraceRecorder(self.state.trace_path),
        )

    def test_invalid_goal_can_be_repaired_in_the_same_turn(self):
        original = self.state.goal
        provider = ScriptedProviderAdapter([
            model_response(tool_calls=[
                ToolCall("bad-goal", "update_goal", goal_arguments([])),
            ], stop_reason=StopReason.TOOL_CALLS),
            model_response(tool_calls=[
                ToolCall("fixed-goal", "update_goal", goal_arguments("pending")),
            ], stop_reason=StopReason.TOOL_CALLS),
            model_response(text="Which module should I inspect first?"),
        ])
        events = []

        def on_event(event):
            events.append(event)
            if event.kind == "tool_finished" and event.data["call_id"] == "bad-goal":
                self.assertIs(self.state.goal, original)

        result = self.make_loop(provider).run_turn(self.state, "Plan the next steps", on_event)
        self.assertEqual(result.stopped_by, "final_answer")
        self.assertEqual(len(provider.requests), 3)
        observations = [message for message in self.state.messages if isinstance(message, ToolResultMessage)]
        self.assertEqual(json.loads(observations[0].content)["metadata"]["status"], "invalid_goal")
        self.assertTrue(json.loads(observations[1].content)["ok"])
        self.assertEqual(self.state.goal.goal_id, original.goal_id)
        self.assertEqual(self.state.goal.status, "active")
        self.assertIsNone(self.state.pending_turn)
        self.assertEqual(sum(event.kind == "goal_updated" for event in events), 1)
        traces = read_jsonl_trace(self.state.trace_path)
        self.assertEqual(sum(event["event"] == "goal_updated" for event in traces), 1)
        self.assertFalse(any(event["event"] == "turn_error" for event in traces))

    def test_final_answer_and_resume_preserve_an_unfinished_goal(self):
        original = self.state.goal
        loop = self.make_loop(ScriptedProviderAdapter([
            model_response(text="I need the expected behavior first."),
        ]))
        loop.run_turn(self.state, "Discuss the task")
        self.assertIs(self.state.goal, original)
        self.store.save(self.state)
        loaded = self.store.load(self.state.session_id)
        self.assertEqual(loaded.goal, original)
        self.assertIsNone(loaded.pending_turn)
        provider = ScriptedProviderAdapter([model_response(text="Still discussing.")])
        self.make_loop(provider).run_turn(loaded, "Continue discussing")
        self.assertIn(original.goal_id, provider.requests[0].system_prompt)
        self.assertEqual(loaded.goal, original)

    def test_failure_and_retry_preserve_the_current_goal(self):
        original = self.state.goal
        loop = self.make_loop(ScriptedProviderAdapter([
            RuntimeError("test provider failure"),
            model_response(text="Still need more information."),
        ]))
        with self.assertRaisesRegex(RuntimeError, "test provider failure"):
            loop.run_turn(self.state, "Discuss the task")
        self.assertIs(self.state.goal, original)
        self.assertIsNotNone(self.state.pending_turn)
        loop.retry_turn(self.state)
        self.assertIs(self.state.goal, original)
        self.assertIsNone(self.state.pending_turn)

    def test_abort_does_not_roll_back_a_goal_update(self):
        loop = self.make_loop(ScriptedProviderAdapter([
            model_response(tool_calls=[
                ToolCall("goal-update", "update_goal", goal_arguments("pending")),
            ], stop_reason=StopReason.TOOL_CALLS),
            RuntimeError("test provider failure"),
        ]))
        with self.assertRaisesRegex(RuntimeError, "test provider failure"):
            loop.run_turn(self.state, "Update the goal")
        updated = self.state.goal
        self.assertEqual(updated.steps[0].status, "pending")
        loop.abort_turn(self.state)
        self.assertIs(self.state.goal, updated)
        self.assertIsNone(self.state.pending_turn)
        self.store.save(self.state)
        self.assertEqual(self.store.load(self.state.session_id).goal, updated)

    def test_plan_mode_cannot_update_goal_even_if_model_requests_it(self):
        original = self.state.goal
        provider = ScriptedProviderAdapter([
            model_response(tool_calls=[
                ToolCall("forbidden-goal", "update_goal", goal_arguments("completed")),
            ], stop_reason=StopReason.TOOL_CALLS),
            model_response(text="Which module should the plan cover?"),
        ])
        loop = self.make_loop(provider)
        loop.switch_mode(PLAN_MODE, build_tool_registry(PLAN_MODE))
        self.state.mode_id = PLAN_MODE.mode_id
        loop.run_turn(self.state, "Discuss a plan")
        self.assertIs(self.state.goal, original)
        observation = next(message for message in self.state.messages if isinstance(message, ToolResultMessage))
        self.assertFalse(json.loads(observation.content)["ok"])
        self.assertIn("unknown tool", json.loads(observation.content)["stderr"])

    def test_clear_is_rejected_while_running_or_pending_and_abort_preserves_goal(self):
        class BlockingProvider(ScriptedProviderAdapter):
            def __init__(self):
                super().__init__([])
                self.started = threading.Event()

            def stream_once(self, request, cancel_event=None):
                self.started.set()
                if not cancel_event.wait(5):
                    raise RuntimeError("test did not interrupt provider")
                raise_if_cancelled(cancel_event)
                yield

        original = self.state.goal
        provider = BlockingProvider()
        loop = self.make_loop(provider)
        runtime = AgentRuntime(loop, self.state, interactive_approvals=False)
        try:
            runtime.send(StartTurn(user_input="Discuss the task"))
            self.assertTrue(provider.started.wait(2))
            with self.assertRaisesRegex(RuntimeError, "turn is running"):
                runtime.clear_goal()
        finally:
            runtime.close()
        with self.assertRaisesRegex(RuntimeError, "turn is pending"):
            runtime.clear_goal()
        self.assertIs(self.state.goal, original)
        loop.abort_turn(self.state)
        self.assertIs(self.state.goal, original)
        self.assertEqual(runtime.clear_goal(), original.goal_id)
        self.assertIsNone(runtime.clear_goal())
        self.store.save(self.state)
        self.assertIsNone(self.store.load(self.state.session_id).goal)
        traces = read_jsonl_trace(self.state.trace_path)
        self.assertEqual(sum(event["event"] == "goal_cleared" for event in traces), 1)
