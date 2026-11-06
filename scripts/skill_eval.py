"""One-night, bounded evaluation of Skill routing using the real read-only loop."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, time
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import tempfile
import time as clock
from zoneinfo import ZoneInfo

from my_agent.agent_loop import AgentLoop
from my_agent.config import load_runtime_config
from my_agent.messages import AssistantMessage, ToolCall, ToolResultMessage
from my_agent.model_invoker import ModelInvoker, RetryPolicy
from my_agent.modes import CollaborationMode
from my_agent.permissions import PermissionPolicy
from my_agent.provider import ModelCompleted, ModelError, ModelMetadata, ModelResponse, ScriptedProviderAdapter, StopReason
from my_agent.providers.deepseek import DeepSeekAdapter
from my_agent.session import SessionState
from my_agent.skills import SkillSettings
from my_agent.tools import LoadSkillTool, ReadFileTool, ToolRegistry
from my_agent.trace import TraceRecorder
from my_agent.workspace import Workspace


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evals/skills"
TZ = ZoneInfo("Asia/Shanghai")
MODEL = "deepseek-flash"
ENDPOINT = "https://api.deepseek.com"
SEED, STEPS, CALLS, OUTPUT = 42, 4, 128, 1536
# Integer pricing units: one unit = 1 / 50,000,000 CNY (off-peak Flash, 2026-11-06).
YUAN = 50_000_000
MODE = CollaborationMode("skill-eval", "Read-only evaluation",
    "You are a read-only assistant. Inspect the supplied evidence with read_file. "
    "Answer the user's question using only that evidence and the tools actually provided.", ())


def window(day, now):
    local = now.astimezone(TZ)
    return local.date() == day and time(18, 5) <= local.time() < time(23, 55)


def design(cases):
    rng = random.Random(SEED)
    blocks = []
    for case in cases:
        arms = ["off", "auto"] + (["manual"] if case["skill"] else [])
        rng.shuffle(arms)
        blocks.append((case, arms))
    result = []
    for repeat in (1, 2):
        rng.shuffle(blocks)
        for case, arms in blocks:
            result.extend((repeat, case, arm) for arm in (arms if repeat == 1 else reversed(arms)))
    return result


def grade(answer, gold):
    try:
        items = json.loads(answer)
    except (TypeError, ValueError):
        return False
    return (isinstance(items, list) and all(isinstance(x, str) for x in items)
            and sorted(items) == sorted(gold))


def record(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def usage_units(inputs, hits, outputs):
    return hits + 50 * (inputs - hits) + 200 * outputs


class StopRun(Exception):
    pass


class Metered(DeepSeekAdapter):
    def __init__(self, key, day, directory):
        super().__init__(api_key=key, model=MODEL, base_url=ENDPOINT, thinking="enabled",
                         context_window_tokens=16384, max_output_tokens=OUTPUT)
        self.day, self.directory = day, directory
        self.calls = []
        self.spent = 0
        self.trial = ""

    def stream_once(self, request, cancel_event=None):
        if not window(self.day, datetime.now(TZ)):
            raise StopRun("outside_authorized_window")
        size = len(json.dumps(self.build_payload(request), ensure_ascii=False).encode())
        reserve = usage_units(size + 1024, 0, OUTPUT)
        if len(self.calls) >= CALLS or self.spent + reserve > YUAN:
            raise StopRun("request_or_cost_budget")
        item = {"call": len(self.calls) + 1, "trial": self.trial, "started": datetime.now(TZ).isoformat()}
        self.calls.append(item)
        record(self.directory / "calls.jsonl", {**item, "phase": "reserved", "reserve_units": reserve})
        try:
            for event in super().stream_once(request, cancel_event):
                if isinstance(event, ModelCompleted):
                    u = event.response.usage
                    if u is None or u.input_tokens is None or u.output_tokens is None:
                        raise StopRun("missing_usage_no_more_calls")
                    hits = u.cache_read_tokens
                    if not (0 <= (hits or 0) <= u.input_tokens and u.output_tokens >= 0):
                        raise StopRun("invalid_usage_no_more_calls")
                    units = usage_units(u.input_tokens, hits or 0, u.output_tokens)
                    item.update(input=u.input_tokens, output=u.output_tokens, hits=hits,
                                reasoning=u.reasoning_tokens, units=units, model=event.response.metadata.response_model_id)
                    self.spent += units
                    record(self.directory / "calls.jsonl", {**item, "phase": "settled"})
                    if units > reserve or self.spent > YUAN:
                        raise StopRun("usage_exceeded_reservation")
                yield event
        except ModelError as exc:
            # Do not put vendor error bodies, request headers or secrets in reports/traces.
            raise StopRun(f"provider_{exc.kind.value}_{exc.status_code or 'unknown'}") from None


def scripted(case, arm):
    calls = [ToolCall("evidence", "read_file", {"path": "evidence.txt"})]
    if arm == "auto" and case["skill"]:
        calls.insert(0, ToolCall("skill", "load_skill", {"name": case["skill"]}))
    meta = ModelMetadata(provider_id="scripted", model_id="scripted-model")
    return ScriptedProviderAdapter([
        ModelResponse.from_parts(tool_calls=calls, stop_reason=StopReason.TOOL_CALLS, metadata=meta),
        ModelResponse.from_parts(text=json.dumps(case["gold"]), metadata=meta)])


def run_case(directory, case, arm, adapter):
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="skill-eval-") as temp:
        base = Path(temp)
        workspace = base / "workspace"
        workspace.mkdir()
        evidence = workspace / "evidence.txt"
        evidence.write_text(case["evidence"], encoding="utf-8")
        catalog = base / "catalog"
        tools = [ReadFileTool()]
        if arm != "off":
            shutil.copytree(DATA / "catalog", catalog)
            tools.append(LoadSkillTool())
        state = SessionState("trial", str(workspace.resolve()), directory, "", "")
        loop = AgentLoop(workspace=Workspace(workspace), model=ModelInvoker(adapter, RetryPolicy(max_attempts=1)),
                         tools=ToolRegistry(tools), permissions=PermissionPolicy(), trace=TraceRecorder(directory / "trace.jsonl"),
                         mode=MODE, max_steps=STEPS, skill_settings=SkillSettings(paths=(str(catalog),) if arm != "off" else ()))
        start = clock.monotonic()
        result = loop.run_turn(state, case["task"] + " 请读取 evidence.txt，只返回答案字符串组成的 JSON 数组。",
                               skills=(case["skill"],) if arm == "manual" else ())
        results = [(m.tool_name, json.loads(m.content)) for m in state.messages if isinstance(m, ToolResultMessage)]
        attempts = [call.arguments.get("name") for m in state.messages if isinstance(m, AssistantMessage)
                    for call in m.tool_calls if call.name == "load_skill"]
        loaded = [entry.skill.name for entry in state.current_skills().loaded] if state.current_skills() else []
        read = any(name == "read_file" and v.get("ok") and v.get("path") == str(evidence.resolve())
                   and v.get("stdout") == case["evidence"] for name, v in results)
        return {"seconds": round(clock.monotonic()-start, 3), "answer": result.answer, "stopped_by": result.stopped_by,
                "passed": result.stopped_by == "final_answer" and read and grade(result.answer, case["gold"]),
                "evidence_read": read, "attempted": attempts, "loaded": loaded, "steps": result.steps,
                "tools": dict(Counter(name for name, _ in results))}


def report(out, rows, meter, status, planned):
    lines = ["# Skill 实测报告", "", f"状态：{status}；已记录 {len(rows)}/{planned} 个 turn。每格显示通过数/已完成数。",
             "", "| 题型 | 组 | 任务通过 | API/工具次数 | 输入/输出 tokens | 缓存命中 tokens | turn 耗时中位数 | 估算元 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for positive in (True, False):
        for arm in (("off", "manual", "auto") if positive else ("off", "auto")):
            group = [r for r in rows if bool(r["expected"]) == positive and r["arm"] == arm]
            if not group:
                continue
            ids = {r["id"] for r in group}
            calls = [c for c in meter.calls if c["trial"] in ids]
            known = [c for c in calls if "units" in c]
            hits = sum(c["hits"] for c in known) if all(c["hits"] is not None for c in known) else "未知"
            lines.append(f"| {'适用' if positive else '不适用'} | {arm} | {sum(r['passed'] for r in group)}/{len(group)} | "
                         f"{len(calls)}/{sum(sum(r['tools'].values()) for r in group)} | "
                         f"{sum(c['input'] for c in known)}/{sum(c['output'] for c in known)} | {hits} | "
                         f"{statistics.median(r['seconds'] for r in group):.2f}s | {sum(c['units'] for c in known)/YUAN:.5f} |")
    pos = [r for r in rows if r["arm"] == "auto" and r["expected"]]
    neg = [r for r in rows if r["arm"] == "auto" and not r["expected"]]
    right = sum(bool(r["attempted"]) and all(s == r["expected"] for s in r["attempted"])
                and r["loaded"] == [r["expected"]] for r in pos)
    lines += ["", f"自动精确选用率：{right}/{len(pos)}；负例误触发率：{sum(bool(r['attempted']) for r in neg)}/{len(neg)}。",
              "分母为 0 表示无可用样本。任务通过不依赖是否选 Skill。"]
    matched = {(r["repeat"], r["case"], r["arm"]): r for r in rows if r["expected"]}
    for a,b in (("manual","off"), ("auto","off"), ("auto","manual")):
        pairs = [(r, matched[(rep,case,b)]) for (rep,case,arm),r in matched.items() if arm == a and (rep,case,b) in matched]
        delta = [int(x["passed"])-int(y["passed"]) for x,y in pairs]
        lines.append(f"- {a} 对 {b} 配对胜/平/负：{delta.count(1)}/{delta.count(0)}/{delta.count(-1)}。")
    unknown = sum("units" not in c for c in meter.calls)
    lines += ["", f"全部请求：{len(meter.calls)}；已知用量估算 ¥{meter.spent/YUAN:.5f}；用量未知 {unknown} 次，不按免费处理。",
              "缓存拆分缺失时按全部未命中估价。小型只读合成任务，每题两次不是独立任务；不是显著性、通用能力或账单证明。",
              "", "逐题答案/选用/工具数在 trials.jsonl；逐次 usage、实际返回模型在 calls.jsonl；各 trial 的 trace 可回溯。"]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-date", type=date.fromisoformat, help="Run once after 18:05 Shanghai on this date; default is offline.")
    args = parser.parse_args()
    cases = json.loads((DATA/"cases.json").read_text())
    plan = design(cases)
    if not args.live_date:
        with tempfile.TemporaryDirectory() as temp:
            for index, (_, case, arm) in enumerate(plan):
                row = run_case(Path(temp)/str(index), case, arm, scripted(case,arm))
                assert row["passed"] and row["loaded"] == ([case["skill"]] if arm != "off" and case["skill"] else [])
        print(f"Offline wiring checks: {len(plan)}; real API calls: 0. No model-quality score.")
        return
    if not window(args.live_date, datetime.now(TZ)):
        parser.exit(2, "Outside the authorized evening; no request sent.\n")
    os.umask(0o077)
    resolved = load_runtime_config().resolve_model(provider_id="deepseek")
    if not resolved.api_key or resolved.base_url.rstrip("/") != ENDPOINT:
        parser.exit(2, "Missing DeepSeek credential or endpoint is not official; no request sent.\n")
    out = ROOT / ".my-agent/evals" / f"skills-{args.live_date}-live"
    out.mkdir(parents=True, exist_ok=False)  # A partial or completed paid run is never automatically repeated.
    meter = Metered(resolved.api_key, args.live_date, out)
    files = [Path(__file__), DATA/"cases.json", *sorted((DATA/"catalog").glob("*/SKILL.md"))]
    manifest = {"model":MODEL, "thinking":"enabled", "seed":SEED, "steps":STEPS, "max_calls":CALLS,
                "output_per_call":OUTPUT, "budget_cny":1, "timezone":"Asia/Shanghai", "tariff":"Flash off-peak CNY 2026-11-06",
                "revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                "hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                "plan":[[rep, c["id"], arm] for rep,c,arm in plan]}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    rows, status = [], "partial"
    try:
        for rep, case, arm in plan:
            meter.trial = f"{rep}-{case['id']}-{arm}"
            row = run_case(out/meter.trial, case, arm, meter)
            row.update(id=meter.trial, case=case["id"], arm=arm, expected=case["skill"], repeat=rep)
            rows.append(row)
            record(out/"trials.jsonl", row)
            report(out, rows, meter, status, len(plan))
        status = "complete"
    except StopRun as exc:
        status = f"stopped: {exc}"
    finally:
        report(out, rows, meter, status, len(plan))
    print(out/"report.md")
    if status != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
