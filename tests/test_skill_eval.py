from datetime import date, datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from my_agent.provider import ModelCompleted, ModelMetadata, ModelRequest, ModelResponse, ModelUsage
from scripts import skill_eval as ev


class SkillEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.day = date(2026, 11, 6)
        self.meter = ev.Metered("offline-dummy", self.day, self.out)
        self.request = ModelRequest(messages=(), tools=())

    def test_window_uses_shanghai_day_and_hour(self):
        for hour, minute, expected in [(10, 4, False), (10, 5, True), (15, 54, True), (15, 55, False)]:
            self.assertEqual(ev.window(self.day, datetime(2026,11,6,hour,minute,tzinfo=timezone.utc)), expected)
        self.assertFalse(ev.window(self.day, datetime(2026,11,7,19,tzinfo=ev.TZ)))

    def test_reversed_paired_design_and_strict_grading(self):
        cases = json.loads((ev.DATA/"cases.json").read_text())
        plan = ev.design(cases)
        self.assertEqual(plan, ev.design(cases))
        self.assertEqual(len(plan), 32)
        for case in cases:
            orders = [[arm for rep,c,arm in plan if c == case and rep == i] for i in (1,2)]
            self.assertEqual(orders[0], orders[1][::-1])
            self.assertEqual("manual" in orders[0], bool(case["skill"]))
        self.assertTrue(ev.grade('["C","A"]', ["A","C"]))
        for invalid in ('["A","C","A"]','["A","B"]','{"answer":["A","C"]}', 'answer: A and C'):
            self.assertFalse(ev.grade(invalid, ["A","C"]))

    def test_guards_run_before_transport(self):
        with patch.object(ev.DeepSeekAdapter, "stream_once") as transport:
            with patch.object(ev, "window", return_value=False), self.assertRaisesRegex(ev.StopRun, "window"):
                list(self.meter.stream_once(self.request))
            with patch.object(ev, "window", return_value=True):
                for count, spent in [(0, ev.YUAN), (ev.CALLS, 0)]:
                    self.meter.calls = [{}] * count
                    self.meter.spent = spent
                    with self.assertRaisesRegex(ev.StopRun, "budget"):
                        list(self.meter.stream_once(self.request))
            transport.assert_not_called()
            self.assertFalse((self.out/"calls.jsonl").exists())

    def test_accounting_uses_provider_usage_and_never_counts_reasoning_twice(self):
        meta = ModelMetadata(provider_id="deepseek", model_id=ev.MODEL)
        usage = ModelUsage(input_tokens=100, output_tokens=40, cache_read_tokens=60, reasoning_tokens=30)
        event = ModelCompleted(ModelResponse.from_parts(text="OK", metadata=meta, usage=usage))
        with patch.object(ev, "window", return_value=True), patch.object(ev.DeepSeekAdapter,"stream_once",return_value=[event]):
            self.assertEqual(list(self.meter.stream_once(self.request)), [event])
        self.assertEqual(self.meter.spent, 60+40*50+40*200)
        rows = [json.loads(line) for line in (self.out/"calls.jsonl").read_text().splitlines()]
        self.assertEqual([r["phase"] for r in rows], ["reserved","settled"])

    def test_missing_usage_aborts_and_preserves_unknown_call(self):
        event = ModelCompleted(ModelResponse.from_parts(metadata=ModelMetadata(provider_id="deepseek",model_id=ev.MODEL)))
        with patch.object(ev, "window", return_value=True), patch.object(ev.DeepSeekAdapter,"stream_once",return_value=[event]):
            with self.assertRaisesRegex(ev.StopRun,"missing_usage"):
                list(self.meter.stream_once(self.request))
        ev.report(self.out, [], self.meter, "stopped", 32)
        self.assertIn("用量未知 1 次", (self.out/"report.md").read_text())

    def test_scripted_paths_distinguish_routing_from_answer_quality(self):
        case = json.loads((ev.DATA/"cases.json").read_text())[0]
        for arm in ("manual", "auto", "off"):
            result = ev.run_case(self.out/arm, case, arm, ev.scripted(case, arm))
            self.assertTrue(result["passed"])
            self.assertEqual(result["attempted"], [case["skill"]] if arm == "auto" else [])
            self.assertEqual(result["loaded"], [] if arm == "off" else [case["skill"]])
        # A correct route is not rewarded for a wrong answer.
        result = ev.run_case(self.out/"wrong", case, "auto", ev.scripted({**case,"gold":["WRONG"]},"auto"))
        self.assertFalse(result["passed"])
        self.assertEqual(result["loaded"], [case["skill"]])
