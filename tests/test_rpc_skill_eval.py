import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from my_agent.agent_loop import AgentLoop
from my_agent.messages import ToolCall, ToolResultMessage, UserMessage
from my_agent.model_invoker import ModelInvoker
from my_agent.permissions import PermissionPolicy
from my_agent.provider import ModelMetadata, ModelResponse, ScriptedProviderAdapter, StopReason
from my_agent.session import SessionStore
from my_agent.skills import SkillSettings, discover_skills
from my_agent.tools import LoadSkillTool, ToolRegistry
from my_agent.trace import TraceRecorder
from my_agent.workspace import Workspace
from scripts import rpc_skill_eval as ev
from scripts.rpc_eval_sandbox import IsolationError, clean_environment, profile, run_python


class RpcSkillEvalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.workspace, self.scratch, self.vendor = [self.root / name for name in ("workspace", "scratch", "vendor")]
        self.workspace.mkdir()
        self.vendor.mkdir()

    def test_environment_is_an_allowlist_not_a_copy(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "never-copy", "UNEXPECTED_SECRET": "never-copy",
                                       "PYTHONSTARTUP": "/tmp/host-script", "DYLD_INSERT_LIBRARIES": "/tmp/host.dylib"}):
            env = clean_environment(self.scratch, self.workspace, self.vendor)
        self.assertNotIn("never-copy", str(env))
        self.assertNotIn("PYTHONSTARTUP", env)
        self.assertNotIn("DYLD_INSERT_LIBRARIES", env)
        self.assertEqual(env["HOME"], str(self.scratch / "home"))

    def test_policy_never_grants_recursive_root_or_global_temp(self):
        candidate = profile(self.workspace, self.scratch, self.vendor)
        self.assertIn('(allow file-read-data (literal "/"))', candidate)
        self.assertNotIn('(subpath "/")', candidate)
        self.assertNotIn('(subpath "/private/tmp")', candidate)
        self.assertNotIn("(allow network", candidate)
        self.assertIn("(deny process-info*)", candidate)
        self.assertNotIn("(allow sysctl-read)", candidate)
        self.assertNotIn("(allow mach-lookup)", candidate)
        trusted = self.root / "trusted"
        self.assertNotIn(str(trusted), candidate)
        self.assertIn(str(trusted), profile(self.workspace, self.scratch, self.vendor, trusted))
        for broad in (Path("/"), Path.home(), Path("/tmp"), ev.ROOT):
            with self.assertRaises(IsolationError):
                profile(broad, self.scratch, self.vendor)
        for overlap in (self.workspace, self.workspace / "grader"):
            with self.assertRaises(IsolationError):
                profile(self.workspace, self.scratch, self.vendor, overlap)

    def test_no_sandbox_means_no_execution(self):
        with patch("scripts.rpc_eval_sandbox.platform.system", return_value="Linux"), \
             patch("scripts.rpc_eval_sandbox.subprocess.Popen") as popen:
            with self.assertRaises(IsolationError):
                run_python(self.workspace, self.scratch, self.vendor, "print(1)")
            popen.assert_not_called()

    def test_export_is_frozen_and_has_no_git_or_solution(self):
        base = self.root / "base"
        ev.export_revision(ev.BASE, base)
        self.assertFalse((base / ".git").exists())
        self.assertFalse((base / "docs").exists())
        self.assertFalse((base / "my_agent/request_store.py").exists())
        self.assertFalse((base / "evals").exists())
        self.assertEqual((base / "my_agent/app_server.py").read_bytes(),
                         ev.subprocess.check_output(["git", "show", f"{ev.BASE}:my_agent/app_server.py"], cwd=ev.ROOT))
        with self.assertRaises(ValueError):
            ev.export_revision("HEAD", self.root / "unfrozen")

    def test_existing_output_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            ev.calibrate(self.workspace)

    def test_calibration_rejects_errors_and_missing_results(self):
        self.assertFalse(ev.calibration_ok({}))
        reference = {"successful": True, "tests": []}
        self.assertFalse(ev.calibration_ok({name: reference for name in
                                           ("base", "reference", "raw-params", "replay-unknown")}))

    def test_real_skill_manual_and_automatic_payloads(self):
        skill_name = "durable-rpc-change"
        settings = SkillSettings(paths=(str(ev.SPEC / "catalog"),))
        catalog = discover_skills(self.workspace, settings=settings)
        self.assertEqual(catalog.diagnostics, ())
        metadata = next(s for s in catalog.skills if s.name == skill_name)
        self.assertTrue(metadata.automatic and metadata.user_invocable)
        meta = ModelMetadata(provider_id="scripted", model_id="offline")
        final = lambda: ModelResponse.from_parts(text="offline transport check", metadata=meta)
        for manual in (False, True):
            with self.subTest(manual=manual):
                # Both issue a duplicate request: manual must acknowledge; auto includes body once.
                load = ModelResponse.from_parts(tool_calls=[ToolCall("first", "load_skill", {"name": skill_name}),
                                                           ToolCall("second", "load_skill", {"name": skill_name})],
                                                stop_reason=StopReason.TOOL_CALLS, metadata=meta)
                provider = ScriptedProviderAdapter([load, final(), final()])
                store = SessionStore(self.root / str(manual))
                state = store.create(self.workspace)
                loop = AgentLoop(workspace=Workspace(self.workspace), model=ModelInvoker(provider),
                    tools=ToolRegistry([LoadSkillTool()]), permissions=PermissionPolicy(),
                    trace=TraceRecorder(state.trace_path), skill_settings=settings)
                loop.run_turn(state, "RPC 请求重试后重复执行，请修复", skills=(skill_name,) if manual else ())
                body = state.current_skills().loaded[0].body
                requests = provider.requests
                def contains_body(message):
                    if isinstance(message, ToolResultMessage):
                        return json.loads(message.content).get("stdout", "").endswith(body)
                    if message.content.startswith("Manually selected skill guidance"):
                        return any(item.get("instructions") == body
                                   for item in json.loads(message.content.split("\n", 1)[1]))
                    return False
                self.assertEqual(sum(map(contains_body, requests[0].messages)), int(manual))
                self.assertEqual(sum(map(contains_body, requests[1].messages)), 1)
                bodies = [m for m in requests[1].messages if contains_body(m)]
                self.assertIsInstance(bodies[0], UserMessage if manual else ToolResultMessage)
                results = [json.loads(m.content) for m in requests[1].messages if isinstance(m, ToolResultMessage)]
                self.assertEqual([r["metadata"]["body_included"] for r in results],
                                 [False, False] if manual else [True, False])
                loop.run_turn(state, "无关的新任务")
                self.assertFalse(any(map(contains_body, provider.requests[-1].messages)))


if __name__ == "__main__":
    unittest.main()
