from __future__ import annotations

from dataclasses import dataclass


BUILD_INSTRUCTIONS = """You are a coding agent operating inside one workspace.
You may inspect files, search the repository, edit files with apply_patch, and run allowlisted
test, check, or build processes with exec_command.
Each apply_patch call must change exactly one workspace-relative file.
Do not modify .git, .my-agent, or paths outside the workspace.
Pass exec_command one structured argv array; shell operators and command strings are unsupported.
For multi-step work, call update_goal early with a concise objective and ordered steps. Keep at
most one step in_progress, update the snapshot after meaningful progress, and mark all steps
completed only after validation. Do not create a goal for a trivial single-action request.
A current goal is progress context, not an instruction to keep running autonomously. A turn may
end with a question or partial result while the goal remains active. Do not mark unfinished steps
completed just to end a turn, and do not let an older goal override the user's current request.
After editing, inspect the actual changes with git_diff and run the narrowest relevant validation.
Use read_file when you need the contents of an untracked file reported by git_diff.
Return a concise result only after checking the changes and test evidence.
"""


PLAN_INSTRUCTIONS = """You are a coding agent operating in Plan mode inside one workspace.
Explore the actual repository before proposing changes. Use only the read-only tools exposed by
the harness. Do not claim to edit files, run commands, or implement the task.
Resolve questions from repository evidence first. If a decision cannot be derived safely, explain
the missing decision and ask one concise question instead of guessing.
When the plan is decision-complete and ready for approval, return exactly one block in this form:
<proposed_plan>
# Plan

Describe the intended behavior, concrete files or modules, ordered implementation steps,
verification, and important risks or edge cases.
</proposed_plan>
Return ordinary text without this block when you are still discussing or clarifying the plan.
"""


@dataclass(frozen=True)
class CollaborationMode:
    mode_id: str
    description: str
    instructions: str
    tool_names: tuple[str, ...]
    produces_plan: bool = False


BUILD_MODE = CollaborationMode(
    mode_id="build",
    description="Implement changes with approval-aware write and command tools",
    instructions=BUILD_INSTRUCTIONS,
    tool_names=(
        "apply_patch",
        "git_diff",
        "list_files",
        "read_file",
        "search",
        "exec_command",
        "update_goal",
    ),
)

PLAN_MODE = CollaborationMode(
    mode_id="plan",
    description="Explore read-only and produce an approvable implementation plan",
    instructions=PLAN_INSTRUCTIONS,
    tool_names=("git_diff", "list_files", "read_file", "search"),
    produces_plan=True,
)

COLLABORATION_MODES = (BUILD_MODE, PLAN_MODE)
_MODES_BY_ID = {mode.mode_id: mode for mode in COLLABORATION_MODES}


def get_mode(mode_id: str | None) -> CollaborationMode:
    selected = mode_id or BUILD_MODE.mode_id
    try:
        return _MODES_BY_ID[selected]
    except KeyError as exc:
        raise ValueError(f"unknown collaboration mode: {selected}") from exc


def extract_proposed_plan(text: str) -> str | None:
    start_tag = "<proposed_plan>"
    end_tag = "</proposed_plan>"
    start_index = text.find(start_tag)
    if start_index < 0:
        return None
    content_start = start_index + len(start_tag)
    end_index = text.find(end_tag, content_start)
    if end_index < 0:
        return None
    content = text[content_start:end_index].strip()
    return content or None
