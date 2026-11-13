"""Prepare and check the RPC Skill study offline; never load credentials or call a model API."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from my_agent.agent_loop import AgentLoop
from my_agent.messages import AssistantMessage, ToolCall, ToolResultMessage
from my_agent.model_invoker import ModelInvoker, RetryPolicy
from my_agent.modes import CollaborationMode
from my_agent.permissions import PermissionPolicy
from my_agent.provider import ModelMetadata, ModelResponse, ScriptedProviderAdapter, StopReason
from my_agent.session import SessionState
from my_agent.skills import SkillSettings, discover_skills, load_skill
from my_agent.tools import LoadSkillTool, ToolRegistry
from my_agent.trace import TraceRecorder
from my_agent.workspace import Workspace
from scripts.rpc_skill_eval import BASE, ROOT, SPEC, hashes


TARGET = "durable-rpc-change"
ARMS = ("off", "manual", "auto")
CATALOG = SPEC / "catalog"
LIMITS = {
    "routing": {"trials": 12, "api_calls": 3, "input_tokens": 24_000, "output_tokens": 8_000,
                "max_output_per_call": 4_096},
    "task": {"trials": 9, "api_calls": 24, "input_tokens": 400_000, "output_tokens": 64_000,
             "max_output_per_call": 8_192},
}
# Planning assumptions, not measurements. Output includes thinking and tool arguments.
ESTIMATES = {
    "routing": {"calls": (20, 36), "mean_input": 3_000, "mean_output": 500},
    "task": {"calls": (90, 162), "mean_input": 16_000, "mean_output": 2_000},
}
PRICING = {"checked_on": "2026-11-13", "currency": "CNY", "per_tokens": 1_000_000,
           "input_miss": "1", "input_hit": "0.02", "output": "4",
           "period": "off_peak", "url": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"}
PROBE_MODE = CollaborationMode("rpc-routing-probe", "Skill routing probe",
    "Assess the user's request with the available tools, then briefly describe the approach. "
    "Do not modify files or claim that a repair or test has been executed.", ())
TASK_MODE = CollaborationMode("rpc-repair-eval", "Isolated RPC repair",
    "Implement the user's task in this workspace. Inspect the code before editing. "
    "Use only the tools provided. Test with the installed Python and project dependencies; "
    "do not access the network or install packages. Preserve existing tests and public interfaces. "
    "Inspect edited files and run relevant tests before explaining the result. "
    "Do not claim that unrun tests passed.",
    ("list_files", "read_file", "search", "apply_patch", "exec_command", "load_skill"))


def cases() -> list[dict]:
    rows = json.loads((SPEC / "routing-cases.json").read_text())
    names = {p.parent.name for p in CATALOG.glob("*/SKILL.md")}
    if (len(names) != 5 or TARGET not in names or len(rows) != 12
            or len({row["id"] for row in rows}) != 12
            or Counter(row["group"] for row in rows) != {"target": 4, "near": 4, "none": 4}):
        raise ValueError("routing design must contain five skills and 4/4/4 unique cases")
    for row in rows:
        allowed = row["allowed_sets"]
        if not allowed or not row["prompt"].strip() or not row["rationale"].strip():
            raise ValueError("incomplete routing case")
        for selection in allowed:
            if len(selection) != len(set(selection)) or not set(selection) <= names:
                raise ValueError("invalid allowed skill set")
        if (row["group"] == "target" and allowed != [[TARGET]]
                or row["group"] == "none" and allowed != [[]]
                or row["group"] == "near" and any(len(s) != 1 or TARGET in s for s in allowed)):
            raise ValueError("group and oracle disagree")
        if any(name in row["prompt"] for name in names):
            raise ValueError("automatic routing prompts must not name the answer skill")
    return rows


def routing_input(case: dict) -> dict:
    # Gold labels and rationales are trusted evaluator data, never model input.
    return {"prompt": case["prompt"]}


def arm_settings(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return {"catalog_enabled": arm != "off", "load_skill_enabled": arm != "off",
            "selected_skills": [TARGET] if arm == "manual" else []}


def task_trials() -> list[dict]:
    # Each arm occurs once in each order position; never adapt order after seeing scores.
    return [{"id": f"task-{repeat + 1}-{arm}", "repeat": repeat + 1, "arm": arm,
             "position": position + 1, **arm_settings(arm)}
            for repeat in range(3)
            for position, arm in enumerate(ARMS[repeat:] + ARMS[:repeat])]


def cost(inputs: int, outputs: int, hits: int = 0) -> str:
    if inputs < 0 or outputs < 0 or not 0 <= hits <= inputs:
        raise ValueError("invalid token counts")
    amount = ((inputs - hits) * Decimal(PRICING["input_miss"])
              + hits * Decimal(PRICING["input_hit"]) + outputs * Decimal(PRICING["output"]))
    return str(amount / PRICING["per_tokens"])


def budget() -> dict:
    estimates = {}
    for index, name in enumerate(("low", "high")):
        calls = sum(stage["calls"][index] for stage in ESTIMATES.values())
        inputs = sum(stage["calls"][index] * stage["mean_input"] for stage in ESTIMATES.values())
        outputs = sum(stage["calls"][index] * stage["mean_output"] for stage in ESTIMATES.values())
        estimates[name] = {"api_calls": calls, "input_tokens": inputs, "output_tokens": outputs,
                           "cny_all_cache_miss": cost(inputs, outputs)}
    ceiling = {key: sum(stage["trials"] * stage[key] for stage in LIMITS.values())
               for key in ("api_calls", "input_tokens", "output_tokens")}
    ceiling["cny_all_cache_miss"] = cost(ceiling["input_tokens"], ceiling["output_tokens"])
    return {"estimates": estimates, "per_trial_limits": LIMITS, "aggregate_ceiling": ceiling,
            "proposed_cny_guardrail": "7", "enforced_by_live_runner": False}


def routing_score(case: dict, attempted: list[str], delivered: list[str], *, completed: bool) -> dict:
    allowed = [set(s) for s in case["allowed_sets"]]
    chosen, bodies = set(attempted), set(delivered)
    return {"completed": completed,
            "exact_selection": chosen in allowed if completed else None,
            "exact_and_delivered": chosen in allowed and bodies == chosen if completed else None,
            "target_attempted": TARGET in chosen,
            "any_attempted": bool(attempted),
            "attempted": attempted, "delivered": sorted(bodies),
            "duplicate_load_calls": len(attempted) - len(chosen)}


def routing_evidence(messages, requests, catalog) -> tuple[list[str], list[str]]:
    raw_calls = [call for message in messages if isinstance(message, AssistantMessage)
                 for call in message.tool_calls if call.name == "load_skill"]
    calls = {call.call_id: call for call in raw_calls}
    if len(calls) != len(raw_calls):
        raise ValueError("ambiguous duplicate tool call IDs")
    attempted = [call.arguments.get("name") if isinstance(call.arguments.get("name"), str)
                 else "<invalid>" for call in raw_calls]
    snapshots = {s.name: load_skill(s, "model") for s in catalog.skills}
    delivered = set()
    # A successful tool execution is not sufficient: the body must reach a subsequent request.
    for request in requests:
        for message in request.messages:
            if not isinstance(message, ToolResultMessage) or message.tool_call_id not in calls:
                continue
            payload = json.loads(message.content)
            meta = payload.get("metadata", {})
            name = calls[message.tool_call_id].arguments.get("name")
            snapshot = snapshots.get(name) if isinstance(name, str) else None
            if (snapshot and payload.get("ok") is True and meta.get("status") == "skill_loaded"
                    and meta.get("body_included") is True
                    and meta.get("skill") == snapshot.manifest()
                    and payload.get("stdout", "").endswith("\n" + snapshot.body)):
                delivered.add(name)
    return attempted, sorted(delivered)


def check_wiring(rows: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix="rpc-routing-wiring-") as temp:
        root = Path(temp).resolve()
        for case in rows:
            workspace = root / case["id"] / "workspace"
            workspace.mkdir(parents=True)
            settings = SkillSettings(paths=(str(CATALOG),), discover_parents=False, discover_personal=False)
            catalog = discover_skills(workspace, settings=settings)
            if catalog.diagnostics or len(catalog.skills) != 5:
                raise ValueError("isolated catalog discovery failed")
            meta = ModelMetadata(provider_id="scripted", model_id="offline")
            responses = []
            names = case["allowed_sets"][0]
            if names:
                responses.append(ModelResponse.from_parts(
                    tool_calls=[ToolCall(f"skill-{i}", "load_skill", {"name": name}) for i, name in enumerate(names)],
                    stop_reason=StopReason.TOOL_CALLS, metadata=meta))
            responses.append(ModelResponse.from_parts(text="offline transport check", metadata=meta))
            adapter = ScriptedProviderAdapter(responses)
            state = SessionState(case["id"], str(workspace), root / case["id"], "", "")
            loop = AgentLoop(workspace=Workspace(workspace), model=ModelInvoker(adapter, RetryPolicy(max_attempts=1)),
                tools=ToolRegistry([LoadSkillTool()]), permissions=PermissionPolicy(), mode=PROBE_MODE,
                trace=TraceRecorder(state.trace_path), max_steps=LIMITS["routing"]["api_calls"], skill_settings=settings)
            result = loop.run_turn(state, routing_input(case)["prompt"])
            attempted, delivered = routing_evidence(state.messages, adapter.requests, catalog)
            score = routing_score(case, attempted, delivered, completed=result.stopped_by == "final_answer")
            if score["exact_and_delivered"] is not True:
                raise ValueError(f"offline transport failed: {case['id']}")
    return {"scripted_turns": len(rows), "passed_transport_checks": len(rows), "real_model_calls": 0,
            "model_selection_accuracy": None}


def prepare(out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    rows = cases()
    wiring = check_wiring(rows)
    source_files = [ROOT / "scripts" / name for name in
                    ("rpc_skill_plan.py", "rpc_skill_eval.py", "rpc_eval_sandbox.py")]
    manifest = {"kind": "rpc-skill-study-preparation", "status": "awaiting_user_confirmation",
        "created_at": datetime.now(timezone.utc).isoformat(), "live_authorized": False, "live_runner_ready": False,
        "model_calls": 0, "model": {"provider": "deepseek", "id": "deepseek-flash",
            "endpoint": "https://api.deepseek.com", "thinking": "enabled",
            "documented_version": "DeepSeek-V4.1-Flash-1017",
            "reasoning_effort": "provider_default_high"},
        "historical_base": BASE, "harness_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "spec_hashes": hashes(SPEC), "harness_hashes": hashes(ROOT / "my_agent"),
        "runner_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "pricing": PRICING, "budget": budget(), "offline_wiring": wiring,
        "task_mode": {"instructions": TASK_MODE.instructions, "tools": list(TASK_MODE.tool_names)},
        "routing_order": [row["id"] for row in rows], "task_trials": task_trials(),
        "execution_rules": {"retries": 0, "fallback_model": False, "concurrency": 1,
            "time_window": "Unscheduled; require a newly approved Asia/Shanghai date, 18:05-23:55",
            "personal_or_parent_skills": False, "missing_usage": "stop_run_and_mark_unknown",
            "budget_exhausted": "stop_trial_and_keep_partial_artifacts_no_top_up",
            "provider_or_isolation_error": "stop_run_no_automatic_retry",
            "context_window_tokens": 131_072, "model_summarization": False,
            "max_tool_seconds": 60, "max_tool_output_chars": 16_000,
            "task_feedback": "visible_baseline_tests_only; hidden_grader_once_after_completion"}}
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (out / "routing-inputs.json").write_text(json.dumps(
        [routing_input(row) for row in rows], ensure_ascii=False, indent=2) + "\n")
    lines = ["# RPC Skill 评测待确认方案", "", "真实模型调用：0。当前仅准备，不存在付费运行入口。",
        f"离线消息接线：{wiring['passed_transport_checks']}/12；不是模型选用率。",
        "", "## 选用测试", "", "| ID | 类型 | 允许的选用集合 |", "|---|---|---|"]
    lines += [f"| {r['id']} | {r['group']} | {json.dumps(r['allowed_sets'], ensure_ascii=False)} |" for r in rows]
    lines += ["", "## 任务对照", "", "同一历史起点、同一需求、同一验收；每次全新隔离工作区和上下文。"]
    lines += [f"- 第 {r} 轮：" + " -> ".join(t["arm"] for t in task_trials() if t["repeat"] == r) for r in (1, 2, 3)]
    lines += ["", "## 预计用量与拟定限制", "", "| 情景 | API 次数 | 输入 tokens | 输出 tokens | 全部未命中估价 CNY |",
              "|---|---:|---:|---:|---:|"]
    for label, value in {**budget()["estimates"], "aggregate_ceiling": budget()["aggregate_ceiling"]}.items():
        lines.append(f"| {label} | {value['api_calls']} | {value['input_tokens']} | {value['output_tokens']} | {value['cny_all_cache_miss']} |")
    lines += ["", "预留预算 7 元；这是待实现的本地停止策略，不是已经生效的厂商账单硬上限。",
              "输入是所有 API 调用累计值，重复发送历史也计入；输出包含 thinking，不把缓存命中从输入里扣掉。",
              "预计区间来自 calls x 假设平均 tokens，不是实测、置信区间或完成保证。",
              "确认后先补齐计量/隔离执行器并完成无付费 preflight，再按新批准日期运行；不用旧授权。",
              "", "完整协议：docs/rpc-skill-evaluation.md；输入、代码与预算快照：manifest.json。"]
    (out / "plan.md").write_text("\n".join(lines) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New offline plan directory; never overwrite")
    args = parser.parse_args()
    os.umask(0o077)
    out = args.output or ROOT / ".my-agent/evals" / ("rpc-skills-plan-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    prepare(out.resolve())
    print(out.resolve() / "plan.md")
    print("Offline preparation: 12 scripted routing checks, 9 planned task trials, real model calls: 0")


if __name__ == "__main__":
    main()
