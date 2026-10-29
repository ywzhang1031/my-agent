from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


GOAL_STEP_STATUSES = {"pending", "in_progress", "completed"}
MAX_GOAL_STEPS = 12
MAX_GOAL_OBJECTIVE_CHARS = 500
MAX_GOAL_STEP_CHARS = 240


@dataclass(frozen=True)
class GoalStep:
    description: str
    status: str

    def to_dict(self) -> dict[str, str]:
        return {
            "description": self.description,
            "status": self.status,
        }


@dataclass(frozen=True)
class GoalState:
    goal_id: str
    objective: str
    steps: tuple[GoalStep, ...]
    status: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "objective": self.objective,
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status,
            "updated_at": self.updated_at,
        }


def update_goal(
    current: GoalState | None,
    arguments: dict[str, Any],
) -> GoalState:
    objective = arguments.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("goal objective must be a non-empty string")
    objective = objective.strip()
    if len(objective) > MAX_GOAL_OBJECTIVE_CHARS:
        raise ValueError(
            f"goal objective exceeds {MAX_GOAL_OBJECTIVE_CHARS} characters"
        )

    steps = _parse_steps(arguments.get("steps"))
    status = "completed" if all(step.status == "completed" for step in steps) else "active"
    same_goal = current is not None and current.objective == objective
    goal_id = current.goal_id if same_goal else f"goal-{secrets.token_hex(6)}"
    return GoalState(
        goal_id=goal_id,
        objective=objective,
        steps=steps,
        status=status,
        updated_at=_now(),
    )


def goal_from_dict(data: Any) -> GoalState | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("goal must be an object")
    goal_id = data.get("goal_id")
    objective = data.get("objective")
    updated_at = data.get("updated_at")
    if not isinstance(goal_id, str) or not goal_id:
        raise ValueError("goal_id must be a non-empty string")
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("goal objective must be a non-empty string")
    objective = objective.strip()
    if len(objective) > MAX_GOAL_OBJECTIVE_CHARS:
        raise ValueError(
            f"goal objective exceeds {MAX_GOAL_OBJECTIVE_CHARS} characters"
        )
    if not isinstance(updated_at, str) or not updated_at:
        raise ValueError("goal updated_at must be a non-empty string")
    steps = _parse_steps(data.get("steps"))
    expected_status = (
        "completed" if all(step.status == "completed" for step in steps) else "active"
    )
    status = str(data.get("status", expected_status))
    if status != expected_status:
        raise ValueError(
            f"goal status {status!r} does not match step state {expected_status!r}"
        )
    return GoalState(
        goal_id=goal_id,
        objective=objective,
        steps=steps,
        status=status,
        updated_at=updated_at,
    )


def format_goal_context(goal: GoalState | None) -> str:
    if goal is None:
        return ""
    return (
        "<current_goal>\n"
        f"{json.dumps(goal.to_dict(), ensure_ascii=False, indent=2)}\n"
        "</current_goal>"
    )


def _parse_steps(value: Any) -> tuple[GoalStep, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("goal steps must be a non-empty array")
    if len(value) > MAX_GOAL_STEPS:
        raise ValueError(f"goal cannot contain more than {MAX_GOAL_STEPS} steps")
    steps: list[GoalStep] = []
    descriptions: set[str] = set()
    in_progress = 0
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"goal step {index} must be an object")
        description = item.get("description")
        status = item.get("status")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"goal step {index} description must be a non-empty string")
        description = description.strip()
        if len(description) > MAX_GOAL_STEP_CHARS:
            raise ValueError(
                f"goal step {index} exceeds {MAX_GOAL_STEP_CHARS} characters"
            )
        if description in descriptions:
            raise ValueError(f"goal step descriptions must be unique: {description}")
        if status not in GOAL_STEP_STATUSES:
            raise ValueError(f"invalid goal step status: {status}")
        descriptions.add(description)
        in_progress += int(status == "in_progress")
        steps.append(GoalStep(description=description, status=str(status)))
    if in_progress > 1:
        raise ValueError("goal cannot have more than one in_progress step")
    return tuple(steps)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
