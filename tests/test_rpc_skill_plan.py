from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from my_agent.messages import AssistantMessage, ToolCall, ToolResultMessage
from my_agent.skills import SkillSettings, discover_skills, load_skill
from my_agent.tools import skill_load_result
from scripts import rpc_skill_plan as plan


class RpcSkillPlanTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.rows = plan.cases()

    def test_design_has_twelve_cases_and_four_distinct_distractors(self):
        self.assertEqual(len(self.rows), 12)
        near = {s for row in self.rows if row["group"] == "near" for allowed in row["allowed_sets"] for s in allowed}
        self.assertEqual(len(near), 4)
        catalog = discover_skills(self.root, settings=SkillSettings(paths=(str(plan.CATALOG),)))
        self.assertEqual(catalog.diagnostics, ())
        self.assertEqual({s.name for s in catalog.skills}, near | {plan.TARGET})
        self.assertTrue(all(s.automatic and s.user_invocable for s in catalog.skills))

    def test_oracles_and_skill_names_do_not_leak_into_user_prompts(self):
        for row in self.rows:
            payload = plan.routing_input(row)
            self.assertEqual(set(payload), {"prompt"})
            self.assertEqual(payload["prompt"], row["prompt"])
            self.assertNotIn("allowed_sets", json.dumps(payload))
            self.assertNotIn("rationale", json.dumps(payload))
        bad = deepcopy(self.rows)
        bad[0]["allowed_sets"] = [["invented-skill"]]
        with patch("scripts.rpc_skill_plan.json.loads", return_value=bad):
            with self.assertRaises(ValueError):
                plan.cases()

    def test_task_order_is_balanced_and_manual_retains_auto_catalog(self):
        trials = plan.task_trials()
        self.assertEqual(len(trials), 9)
        self.assertEqual(len({r["id"] for r in trials}), 9)
        for arm in plan.ARMS:
            self.assertEqual({r["position"] for r in trials if r["arm"] == arm}, {1, 2, 3})
            self.assertEqual({r["repeat"] for r in trials if r["arm"] == arm}, {1, 2, 3})
        self.assertEqual(plan.arm_settings("off"), {
            "catalog_enabled": False, "load_skill_enabled": False, "selected_skills": []})
        manual = plan.arm_settings("manual")
        self.assertEqual(manual["selected_skills"], [plan.TARGET])
        self.assertTrue(manual["catalog_enabled"] and manual["load_skill_enabled"])
        self.assertEqual(plan.arm_settings("auto"), {**manual, "selected_skills": []})
        with self.assertRaises(ValueError):
            plan.arm_settings("surprise")

    def test_budget_arithmetic_counts_repeated_input_and_all_output(self):
        budget = plan.budget()
        self.assertEqual(budget["estimates"]["low"], {
            "api_calls": 110, "input_tokens": 1_500_000, "output_tokens": 190_000,
            "cny_all_cache_miss": "2.26"})
        self.assertEqual(budget["estimates"]["high"], {
            "api_calls": 198, "input_tokens": 2_700_000, "output_tokens": 342_000,
            "cny_all_cache_miss": "4.068"})
        ceiling = budget["aggregate_ceiling"]
        self.assertEqual(ceiling, {"api_calls": 252, "input_tokens": 3_888_000,
            "output_tokens": 672_000, "cny_all_cache_miss": "6.576"})
        self.assertLess(Decimal(ceiling["cny_all_cache_miss"]), Decimal(budget["proposed_cny_guardrail"]))
        self.assertFalse(budget["enforced_by_live_runner"])
        self.assertEqual(Decimal(plan.cost(1_000_000, 0, 1_000_000)), Decimal("0.02"))
        self.assertEqual(Decimal(plan.cost(1_000_000, 1_000_000)), Decimal("5"))
        with self.assertRaises(ValueError):
            plan.cost(1, 0, 2)

    def test_routing_requires_correct_choice_and_delivered_body(self):
        case = self.rows[0]
        name = plan.TARGET
        self.assertTrue(plan.routing_score(case, [name], [name], completed=True)["exact_and_delivered"])
        missing = plan.routing_score(case, [name], [], completed=True)
        self.assertTrue(missing["exact_selection"])
        self.assertFalse(missing["exact_and_delivered"])
        extra = plan.routing_score(case, [name, "provider-request-retry"], [name], completed=True)
        self.assertFalse(extra["exact_and_delivered"])
        duplicate = plan.routing_score(case, [name, name], [name], completed=True)
        self.assertTrue(duplicate["exact_and_delivered"])
        self.assertEqual(duplicate["duplicate_load_calls"], 1)
        self.assertFalse(plan.routing_score(case, [], [], completed=True)["exact_selection"])
        self.assertIsNone(plan.routing_score(case, [], [], completed=False)["exact_and_delivered"])

    def test_near_cases_are_not_blanket_negative_skills(self):
        near = next(row for row in self.rows if row["group"] == "near")
        correct = near["allowed_sets"][0]
        self.assertTrue(plan.routing_score(near, correct, correct, completed=True)["exact_and_delivered"])
        self.assertFalse(plan.routing_score(near, [plan.TARGET], [plan.TARGET], completed=True)["exact_selection"])
        none = next(row for row in self.rows if row["group"] == "none")
        self.assertTrue(plan.routing_score(none, [], [], completed=True)["exact_and_delivered"])
        self.assertFalse(plan.routing_score(none, correct, correct, completed=True)["exact_and_delivered"])

    def evidence_fixture(self):
        catalog = discover_skills(self.root, settings=SkillSettings(paths=(str(plan.CATALOG),)))
        skill = next(s for s in catalog.skills if s.name == plan.TARGET)
        loaded = load_skill(skill, "model")
        call = ToolCall("load-1", "load_skill", {"name": plan.TARGET})
        assistant = AssistantMessage(content="", tool_calls=[call])
        result = ToolResultMessage(tool_call_id=call.call_id, tool_name=call.name,
            content=json.dumps(skill_load_result(loaded, include_body=True).to_dict()))
        return catalog, [assistant, result]

    def test_body_must_reach_next_request_not_only_session_log(self):
        catalog, messages = self.evidence_fixture()
        attempts, delivered = plan.routing_evidence(messages, [], catalog)
        self.assertEqual(attempts, [plan.TARGET])
        self.assertEqual(delivered, [])
        attempts, delivered = plan.routing_evidence(messages, [SimpleNamespace(messages=messages)], catalog)
        self.assertEqual(delivered, [plan.TARGET])

    def test_wrong_body_or_failed_result_cannot_claim_success(self):
        catalog, messages = self.evidence_fixture()
        payload = json.loads(messages[1].content)
        for changed in ({**payload, "stdout": "already loaded"}, {**payload, "ok": False}):
            altered = [messages[0], ToolResultMessage(tool_call_id="load-1", tool_name="load_skill",
                                                     content=json.dumps(changed))]
            self.assertEqual(plan.routing_evidence(altered, [SimpleNamespace(messages=altered)], catalog)[1], [])
        with self.assertRaises(ValueError):
            plan.routing_evidence([messages[0], messages[0]], [], catalog)

    def test_all_twelve_scripted_cases_use_real_loop_and_isolated_catalog(self):
        with patch("socket.create_connection", side_effect=AssertionError("no network allowed")):
            result = plan.check_wiring(self.rows)
        self.assertEqual(result["passed_transport_checks"], 12)
        self.assertEqual(result["real_model_calls"], 0)
        self.assertIsNone(result["model_selection_accuracy"])

    def test_plan_has_no_live_authorization_and_refuses_overwrite(self):
        out = self.root / "plan"
        with patch("scripts.rpc_skill_plan.check_wiring", return_value={"passed_transport_checks": 12}):
            manifest = plan.prepare(out)
        self.assertFalse(manifest["live_authorized"])
        self.assertFalse(manifest["live_runner_ready"])
        self.assertEqual(manifest["model_calls"], 0)
        self.assertEqual(len(json.loads((out / "routing-inputs.json").read_text())), 12)
        self.assertIn("routing-cases.json", manifest["spec_hashes"])
        self.assertIn("rpc_skill_plan.py", manifest["runner_hashes"])
        original = (out / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            plan.prepare(out)
        self.assertEqual((out / "manifest.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
