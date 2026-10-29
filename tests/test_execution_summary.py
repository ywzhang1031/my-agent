import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from my_agent.trajectory import make_trajectory


def turn_started(turn_id="session-1:1"):
    return {
        "event": "turn_started",
        "session_id": "session-1",
        "turn_id": turn_id,
        "workspace": "/test/workspace",
        "task": "Update and inspect the repository",
        "ts": 1.0,
    }


def tool_result(name, call_id, result, *, arguments=None, turn_id="session-1:1"):
    return {
        "event": "tool_result",
        "session_id": "session-1",
        "turn_id": turn_id,
        "step": 1,
        "ts": 2.0,
        "call": {"name": name, "call_id": call_id, "arguments": arguments or {}},
        "result": result,
    }


def applied_patch(call_id, path, operation="update"):
    return tool_result("apply_patch", call_id, {
        "ok": True,
        "metadata": {"applied": True, "operation": operation, "changed_files": [path]},
    })


def summary_for(*results):
    return make_trajectory([turn_started(), *results])["turns"][0]["execution_summary"]


class ExecutionSummaryTests(unittest.TestCase):
    def test_empty_turn_and_model_claims_do_not_create_execution_evidence(self):
        summary = summary_for(
            {"event": "final_answer", "turn_id": "session-1:1",
             "answer": "I edited app.py and all tests passed."},
            tool_result("read_file", "read-1", {"ok": True, "stdout": "app.py"}),
        )
        self.assertEqual(summary, {"changed_files": [], "evidence": []})

    def test_only_successfully_applied_patches_contribute_files(self):
        failed = [
            {"ok": False, "metadata": {"applied": False}},
            {"ok": False, "metadata": {"status": "denied"}},
            {"ok": False, "metadata": {"applied": True, "changed_files": ["bad.py"]}},
            {"ok": True, "metadata": {"changed_files": ["unconfirmed.py"]}},
        ]
        summary = summary_for(
            applied_patch("add", "app.py", "add"),
            *[tool_result("apply_patch", str(i), result) for i, result in enumerate(failed)],
            applied_patch("update", "tests.py"),
            applied_patch("delete", "app.py", "delete"),
        )
        self.assertEqual(summary["changed_files"], ["app.py", "tests.py"])
        self.assertEqual([entry["call_id"] for entry in summary["evidence"]],
                         ["add", "update", "delete"])
        self.assertEqual(summary["evidence"][2]["operation"], "delete")
        self.assertEqual(summary["evidence"][0]["changed_files"], ["app.py"])

    def test_proposed_or_approved_calls_without_results_are_not_execution_evidence(self):
        summary = summary_for(
            {"event": "tool_call", "turn_id": "session-1:1", "step": 1,
             "call": {"name": "exec_command", "call_id": "proposal",
                      "arguments": {"argv": ["pytest"]}}},
            {"event": "approval_resolved", "turn_id": "session-1:1",
             "call_id": "proposal", "decision": "allow_once"},
        )
        self.assertEqual(summary, {"changed_files": [], "evidence": []})

    def test_command_evidence_preserves_reported_fields_without_a_task_verdict(self):
        command = tool_result("exec_command", "check", {
            "ok": True, "exit_code": 0, "truncated": True,
            "stdout": "large output is kept in the original observation",
            "metadata": {
                "status": "success", "category": "test",
                "normalized_argv": ["python3", "-m", "unittest"], "cwd": "tests",
                "timed_out": False, "interrupted": False, "duration_ms": 125,
            },
        }, arguments={"argv": ["python3", "-m", "unittest"], "cwd": "./tests"})
        summary = summary_for(command)
        self.assertEqual(summary["evidence"], [{
            "tool_name": "exec_command", "call_id": "check", "step": 1, "ts": 2.0,
            "requested_argv": ["python3", "-m", "unittest"], "requested_cwd": "./tests",
            "normalized_argv": ["python3", "-m", "unittest"], "cwd": "tests",
            "category": "test", "status": "success", "ok": True, "exit_code": 0,
            "timed_out": False, "interrupted": False, "truncated": True,
            "duration_ms": 125,
        }])
        self.assertEqual(set(summary), {"changed_files", "evidence"})

    def test_failed_denied_and_unstarted_commands_keep_distinct_statuses(self):
        cases = [
            ("nonzero_exit", 1), ("timed_out", -9), ("interrupted", -9),
            ("interrupted", None), ("denied", None), ("invalid_request", None),
            ("spawn_error", None), ("sandbox_error", None), ("repeated_call", None),
        ]
        for status, exit_code in cases:
            with self.subTest(status=status, exit_code=exit_code):
                summary = summary_for(tool_result("exec_command", "call", {
                    "ok": False, "exit_code": exit_code, "metadata": {"status": status},
                }, arguments={"argv": ["pytest"]}))
                entry = summary["evidence"][0]
                self.assertEqual(entry["status"], status)
                self.assertFalse(entry["ok"])
                self.assertEqual(entry["exit_code"], exit_code)
                self.assertIsNone(entry["normalized_argv"])
                self.assertIsNone(entry["cwd"])
                self.assertIsNone(entry["duration_ms"])
                self.assertEqual(entry["requested_argv"], ["pytest"])

    def test_missing_process_metadata_remains_unknown(self):
        entry = summary_for(tool_result("exec_command", "partial", {"ok": True}))[
            "evidence"
        ][0]
        self.assertEqual(entry["status"], "unknown")
        for field in ("exit_code", "category", "timed_out", "interrupted", "truncated"):
            self.assertIsNone(entry[field])

    def test_result_order_survives_same_step_retry_and_reused_call_ids(self):
        check = tool_result("exec_command", "reused", {
            "ok": False, "exit_code": 1, "metadata": {"status": "nonzero_exit"},
        })
        check["ts"] = 8.0
        patch = applied_patch("patch", "app.py")
        patch["ts"] = 7.0
        retry = tool_result("exec_command", "reused", {
            "ok": True, "exit_code": 0, "metadata": {"status": "success"},
        })
        summary = summary_for(check, patch,
                              {"event": "turn_resumed", "turn_id": "session-1:1", "step": 1},
                              retry)
        self.assertEqual([entry["call_id"] for entry in summary["evidence"]],
                         ["reused", "patch", "reused"])
        self.assertEqual([entry["ts"] for entry in summary["evidence"]], [8.0, 7.0, 2.0])
        self.assertEqual([entry["step"] for entry in summary["evidence"]], [1, 1, 1])

    def test_evidence_is_scoped_to_its_turn_even_after_abort(self):
        events = [
            turn_started(), applied_patch("first-patch", "app.py"),
            {"event": "turn_aborted", "turn_id": "session-1:1"},
            turn_started("session-1:2"),
            tool_result("exec_command", "second-check", {
                "ok": True, "exit_code": 0, "metadata": {"status": "success"},
            }, turn_id="session-1:2"),
        ]
        before = copy.deepcopy(events)
        trajectory = make_trajectory(events)
        first, second = trajectory["turns"]
        self.assertEqual(first["execution_summary"]["changed_files"], ["app.py"])
        self.assertEqual(first["outcome"]["status"], "aborted")
        self.assertEqual(second["execution_summary"]["changed_files"], [])
        self.assertEqual(second["execution_summary"]["evidence"][0]["call_id"], "second-check")
        self.assertEqual(second["outcome"]["status"], "incomplete")
        self.assertEqual(events, before)
        self.assertEqual(make_trajectory(events), trajectory)

    def test_export_script_emits_v8_summary_without_rewriting_raw_trace(self):
        repo = Path(__file__).resolve().parents[1]
        events = [turn_started(), applied_patch("patch-1", "app.py")]
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp, "trace.jsonl")
            output = Path(tmp, "trajectory.json")
            raw = "".join(json.dumps(event) + "\n" for event in events)
            trace.write_text(raw, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "scripts/export_trajectory.py", "--trace", str(trace),
                 "--output", str(output)],
                cwd=repo, env={**os.environ, "PYTHONPATH": str(repo)},
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            trajectory = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(trajectory["schema_version"], "my-agent.trajectory.v8")
            self.assertEqual(trajectory["turns"][0]["execution_summary"]["changed_files"], ["app.py"])
            self.assertEqual(trace.read_text(encoding="utf-8"), raw)
