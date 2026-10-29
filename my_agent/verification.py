from __future__ import annotations

import json
from typing import Any

from .database import Database


NOT_RUN = {"denied", "invalid_request", "repeated_call", "spawn_error", "unavailable"}


def read_verification(database: Database, session_id: str, turn_id: str | None = None) -> dict[str, Any]:
    """A read-only projection, not a task-completion verdict or workspace fingerprint."""
    with database.transaction() as connection:
        row = connection.execute("SELECT data FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(f"session not found: {session_id}")
        session = json.loads(row[0])
        if turn_id is None:
            pending = session.get("pending_turn")
            turn_id = pending["turn_id"] if pending else (
                f"{session_id}:{session['turn_counter']}" if session.get("turn_counter") else None
            )
        if turn_id is not None and (
            not turn_id.startswith(session_id + ":") or not turn_id[len(session_id) + 1:].isdigit()
        ):
            raise ValueError("turn_id must belong to the requested session")
        rows = connection.execute(
            "SELECT rowid,attempt_id,call_id,tool_name,arguments,status,result,started_at,finished_at "
            "FROM tool_attempts WHERE session_id=? AND turn_id=? ORDER BY rowid", (session_id, turn_id),
        ).fetchall()

    changed: dict[str, dict] = {}
    checks: list[dict] = []
    unknown: list[dict] = []
    last_edit = 0
    for order, attempt_id, call_id, name, arguments, status, result, started, finished in rows:
        if name not in {"apply_patch", "exec_command"}:
            continue
        args = json.loads(arguments)
        result = json.loads(result) if result is not None else {}
        metadata = result.get("metadata", {})
        base = {"attempt_id": attempt_id, "call_id": call_id, "tool_name": name,
                "started_at": started, "finished_at": finished}
        outcome = metadata.get("status", "unknown")
        if status != "completed":
            unknown.append({**base, "reason": "result_not_committed"})
        elif name == "apply_patch":
            if result.get("ok") is True and metadata.get("applied") is True:
                last_edit = order
                for path in metadata.get("changed_files", []):
                    changed[path] = {**base, "path": path, "operation": metadata.get("operation")}
            elif metadata.get("applied") is not False and outcome not in NOT_RUN:
                unknown.append({**base, "reason": "patch_outcome_unknown"})
        if name != "exec_command":
            continue
        exit_code = result.get("exit_code")
        if outcome in NOT_RUN:
            check_status = "not_run"
        elif (status != "completed" or metadata.get("timed_out") or metadata.get("interrupted")
              or outcome in {"interrupted", "timeout", "timed_out"} or type(exit_code) is not int):
            check_status = "unknown"
        elif exit_code == 0 and result.get("ok") is True and outcome == "success":
            check_status = "passed"
        else:
            check_status = "failed"
        argv = metadata.get("normalized_argv", args.get("argv", []))
        if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
            argv = []
        checks.append({**base, "order": order, "status": check_status,
                       "argv": argv,
                       "cwd": metadata.get("cwd", args.get("cwd", ".")),
                       "category": metadata.get("category"), "exit_code": exit_code,
                       "duration_ms": metadata.get("duration_ms")})

    # One turn executes tools serially. Only edits observed in THIS turn establish ordering.
    latest: dict[str, dict] = {}
    for check in checks:
        check["freshness"] = "stale" if check["order"] < last_edit else "no_later_tracked_edit"
        identity = json.dumps([check["argv"], check["cwd"]], sort_keys=True)
        latest[identity] = check
        del check["order"]
    current = list(latest.values())
    if unknown or any(check["status"] == "unknown" for check in current):
        summary = "unknown"
    elif any(check["freshness"] != "stale" and check["status"] == "failed" for check in current):
        summary = "failed"
    elif any(check["freshness"] == "stale" and check["status"] != "not_run" for check in current):
        summary = "stale"
    elif any(check["status"] == "not_run" for check in current):
        summary = "unverified"
    elif any(check["status"] == "passed" and check["category"] in {"test", "check", "build"} for check in current):
        summary = "checks_passed"
    elif changed:
        summary = "unverified"
    else:
        summary = "no_evidence"
    return {"session_id": session_id, "turn_id": turn_id, "status": summary,
            "scope": "turn_tracked_operations", "external_changes": "not_observed",
            "changed_files": list(changed.values()), "checks": checks, "unknown_attempts": unknown}
