from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "my-agent.trajectory.v3"


def read_jsonl_trace(path: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(path)
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        events.append(event)
    return events


def make_trajectory(events: list[dict[str, Any]], source_path: str | Path | None = None) -> dict[str, Any]:
    if not events:
        raise ValueError("trace contains no events")
    if not _first_event(events, "turn_started"):
        raise ValueError("trace does not contain a turn_started event")
    return _make_session_trajectory(events, source_path)


def _make_session_trajectory(
    events: list[dict[str, Any]],
    source_path: str | Path | None,
) -> dict[str, Any]:
    started_events = [event for event in events if event.get("event") == "turn_started"]
    turns = []
    for started in started_events:
        turn_id = started.get("turn_id")
        turn_events = [event for event in events if event.get("turn_id") == turn_id]
        final = _first_event(turn_events, "final_answer")
        aborted = _first_event(turn_events, "turn_aborted")
        errors = [event for event in turn_events if event.get("event") == "turn_error"]
        turns.append(
            {
                "turn_id": turn_id,
                "task": started.get("task", ""),
                "started_at": _first_ts(turn_events),
                "ended_at": _last_ts(turn_events),
                "max_steps": started.get("max_steps"),
                "steps": [
                    _make_step(step, step_events)
                    for step, step_events in _group_step_events(turn_events)
                ],
                "errors": [
                    {
                        "step": error.get("step"),
                        "error": error.get("error", ""),
                        "retryable": error.get("retryable", False),
                        "ts": error.get("ts"),
                    }
                    for error in errors
                ],
                "final_answer": final.get("answer", "") if final else "",
                "outcome": {
                    "status": (
                        "completed"
                        if final
                        else "aborted"
                        if aborted
                        else "failed"
                        if errors
                        else "incomplete"
                    ),
                    "final_step": final.get("step") if final else None,
                },
                "metrics": _make_metrics(
                    turn_events,
                    _first_ts(turn_events),
                    _last_ts(turn_events),
                ),
            }
        )

    first_started = started_events[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": _trajectory_id(events),
        "source_path": str(source_path) if source_path is not None else None,
        "session_id": first_started.get("session_id", ""),
        "workspace": first_started.get("workspace", ""),
        "started_at": _first_ts(events),
        "ended_at": _last_ts(events),
        "turns": turns,
        "outcome": {
            "status": "completed"
            if turns and all(turn["outcome"]["status"] == "completed" for turn in turns)
            else "failed"
            if any(turn["outcome"]["status"] == "failed" for turn in turns)
            else "aborted"
            if any(turn["outcome"]["status"] == "aborted" for turn in turns)
            else "incomplete",
            "turn_count": len(turns),
        },
        "metrics": _make_metrics(events, _first_ts(events), _last_ts(events)),
    }


def write_trajectory_json(trajectory: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _group_step_events(events: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        step = event.get("step")
        if isinstance(step, int):
            grouped.setdefault(step, []).append(event)
    return sorted(grouped.items())


def _make_step(step: int, events: list[dict[str, Any]]) -> dict[str, Any]:
    model_request = _first_event(events, "model_request") or {}
    model_response = _first_event(events, "model_response") or {}
    final = _first_event(events, "final_answer")
    return {
        "step": step,
        "model_request": {
            "messages": model_request.get("messages"),
            "tools": model_request.get("tools", []),
            "context": model_request.get("context", {}),
            "ts": model_request.get("ts"),
        },
        "model_response": {
            "content": model_response.get("content", ""),
            "tool_calls": model_response.get("tool_calls", []),
            "finish_reason": model_response.get("finish_reason"),
            "usage": model_response.get("usage", {}),
            "ts": model_response.get("ts"),
        },
        "actions": [_make_action(event) for event in events if event.get("event") == "tool_call"],
        "observations": [
            _make_observation(event) for event in events if event.get("event") == "tool_result"
        ],
        "final_answer": final.get("answer") if final else None,
    }


def _make_action(event: dict[str, Any]) -> dict[str, Any]:
    call = event.get("call", {})
    return {
        "action_id": call.get("call_id", ""),
        "tool_name": call.get("name", ""),
        "arguments": call.get("arguments", {}),
        "ts": event.get("ts"),
    }


def _make_observation(event: dict[str, Any]) -> dict[str, Any]:
    call = event.get("call", {})
    result = event.get("result", {})
    return {
        "action_id": call.get("call_id", ""),
        "tool_name": call.get("name", ""),
        "ts": event.get("ts"),
        "output": {
            "ok": result.get("ok"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "exit_code": result.get("exit_code"),
            "truncated": result.get("truncated", False),
            "path": result.get("path"),
            "metadata": result.get("metadata", {}),
        },
    }


def _make_metrics(events: list[dict[str, Any]], started_at: float | None, ended_at: float | None) -> dict[str, Any]:
    tool_results = [event for event in events if event.get("event") == "tool_result"]
    return {
        "events": len(events),
        "model_requests": sum(1 for event in events if event.get("event") == "model_request"),
        "model_responses": sum(1 for event in events if event.get("event") == "model_response"),
        "provider_retries": sum(1 for event in events if event.get("event") == "provider_retry"),
        "turn_errors": sum(1 for event in events if event.get("event") == "turn_error"),
        "turn_aborts": sum(1 for event in events if event.get("event") == "turn_aborted"),
        "context_compactions": sum(
            1 for event in events if event.get("event") == "context_compacted"
        ),
        "tool_calls": sum(1 for event in events if event.get("event") == "tool_call"),
        "failed_tool_calls": sum(
            1 for event in tool_results if not event.get("result", {}).get("ok", False)
        ),
        "truncated_observations": sum(
            1 for event in tool_results if event.get("result", {}).get("truncated", False)
        ),
        "duration_seconds": round(ended_at - started_at, 3)
        if started_at is not None and ended_at is not None
        else None,
    }


def _first_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    for event in events:
        if event.get("event") == event_name:
            return event
    return None


def _first_ts(events: list[dict[str, Any]]) -> float | None:
    for event in events:
        ts = event.get("ts")
        if isinstance(ts, int | float):
            return float(ts)
    return None


def _last_ts(events: list[dict[str, Any]]) -> float | None:
    for event in reversed(events):
        ts = event.get("ts")
        if isinstance(ts, int | float):
            return float(ts)
    return None


def _trajectory_id(events: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(event, sort_keys=True, ensure_ascii=False) for event in events)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
