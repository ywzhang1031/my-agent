from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .permissions import PermissionPolicy
from .workspace import Workspace


PatchKind = Literal["add", "update", "delete"]
MAX_PATCH_CHARS = 100_000
MAX_PATCHABLE_FILE_BYTES = 1_000_000


class PatchError(ValueError):
    pass


@dataclass(frozen=True)
class PatchHunk:
    old_lines: tuple[str, ...]
    new_lines: tuple[str, ...]


@dataclass(frozen=True)
class PatchOperation:
    kind: PatchKind
    path: str
    content: str = ""
    hunks: tuple[PatchHunk, ...] = ()


@dataclass(frozen=True)
class AppliedPatch:
    operation: PatchKind
    path: str
    absolute_path: Path


_FILE_HEADERS: dict[str, PatchKind] = {
    "*** Add File: ": "add",
    "*** Update File: ": "update",
    "*** Delete File: ": "delete",
}


def parse_patch(patch: str) -> PatchOperation:
    if not isinstance(patch, str) or not patch.strip():
        raise PatchError("patch must be a non-empty string")
    if len(patch) > MAX_PATCH_CHARS:
        raise PatchError(f"patch exceeds {MAX_PATCH_CHARS} characters")
    lines = patch.splitlines()
    if len(lines) < 3 or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise PatchError("patch must start with '*** Begin Patch' and end with '*** End Patch'")

    operation_headers = [
        (index, line)
        for index, line in enumerate(lines[1:-1], start=1)
        if any(line.startswith(prefix) for prefix in _FILE_HEADERS)
    ]
    if len(operation_headers) != 1 or operation_headers[0][0] != 1:
        raise PatchError("patch must contain exactly one file operation")

    header = operation_headers[0][1]
    kind, path = _parse_file_header(header)
    body = lines[2:-1]
    if kind == "add":
        if any(not line.startswith("+") for line in body):
            raise PatchError("every Add File line must start with '+'")
        content = "\n".join(line[1:] for line in body)
        if body:
            content += "\n"
        return PatchOperation(kind=kind, path=path, content=content)
    if kind == "delete":
        if body:
            raise PatchError("Delete File does not accept patch body lines")
        return PatchOperation(kind=kind, path=path)
    return PatchOperation(kind=kind, path=path, hunks=_parse_hunks(body))


def apply_patch(
    patch: str,
    workspace: Workspace,
    permissions: PermissionPolicy,
) -> AppliedPatch:
    operation = parse_patch(patch)
    target, relative_path = workspace.resolve_patch_path(operation.path)
    allowed, reason = permissions.allow_file_change(operation.kind, relative_path)
    if not allowed:
        raise PatchError(reason)

    if operation.kind == "add":
        if target.exists():
            raise PatchError(f"cannot add existing path: {relative_path}")
        _atomic_write(target, operation.content)
    elif operation.kind == "update":
        original = _read_patchable_file(target, relative_path)
        updated = _apply_hunks(original, operation.hunks)
        if updated == original:
            raise PatchError(f"patch made no changes: {relative_path}")
        _atomic_write(target, updated, mode=target.stat().st_mode & 0o777)
    else:
        if not target.exists():
            raise PatchError(f"cannot delete missing path: {relative_path}")
        if not target.is_file():
            raise PatchError(f"cannot delete non-file path: {relative_path}")
        target.unlink()

    return AppliedPatch(
        operation=operation.kind,
        path=relative_path,
        absolute_path=target,
    )


def _parse_file_header(header: str) -> tuple[PatchKind, str]:
    for prefix, kind in _FILE_HEADERS.items():
        if header.startswith(prefix):
            path = header[len(prefix) :].strip()
            if not path:
                raise PatchError("file operation path is required")
            return kind, path
    raise PatchError(f"unknown file operation: {header}")


def _parse_hunks(lines: list[str]) -> tuple[PatchHunk, ...]:
    if not lines:
        raise PatchError("Update File requires at least one hunk")
    hunks: list[PatchHunk] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("@@"):
            raise PatchError("each update hunk must start with '@@'")
        index += 1
        old_lines: list[str] = []
        new_lines: list[str] = []
        changed = False
        while index < len(lines) and not lines[index].startswith("@@"):
            line = lines[index]
            if not line or line[0] not in {" ", "+", "-"}:
                raise PatchError("hunk lines must start with ' ', '+', or '-'")
            marker, content = line[0], line[1:]
            if marker in {" ", "-"}:
                old_lines.append(content)
            if marker in {" ", "+"}:
                new_lines.append(content)
            changed = changed or marker in {"+", "-"}
            index += 1
        if not old_lines:
            raise PatchError("update hunks require exact existing context")
        if not changed:
            raise PatchError("update hunk contains no changes")
        hunks.append(PatchHunk(tuple(old_lines), tuple(new_lines)))
    return tuple(hunks)


def _read_patchable_file(path: Path, relative_path: str) -> str:
    if not path.exists():
        raise PatchError(f"cannot update missing path: {relative_path}")
    if not path.is_file():
        raise PatchError(f"cannot update non-file path: {relative_path}")
    raw = path.read_bytes()
    if len(raw) > MAX_PATCHABLE_FILE_BYTES:
        raise PatchError(
            f"file exceeds {MAX_PATCHABLE_FILE_BYTES} patchable bytes: {relative_path}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchError(f"cannot patch non-UTF-8 file: {relative_path}") from exc
    if "\r" in text:
        raise PatchError(f"cannot patch file with non-LF line endings: {relative_path}")
    return text


def _apply_hunks(original: str, hunks: tuple[PatchHunk, ...]) -> str:
    original_lines = original.splitlines()
    output: list[str] = []
    cursor = 0
    for hunk in hunks:
        matches = [
            index
            for index in range(cursor, len(original_lines) - len(hunk.old_lines) + 1)
            if tuple(original_lines[index : index + len(hunk.old_lines)]) == hunk.old_lines
        ]
        if not matches:
            raise PatchError("patch context did not match target file")
        if len(matches) > 1:
            raise PatchError("patch context is ambiguous; include more unchanged lines")
        start = matches[0]
        output.extend(original_lines[cursor:start])
        output.extend(hunk.new_lines)
        cursor = start + len(hunk.old_lines)
    output.extend(original_lines[cursor:])
    updated = "\n".join(output)
    if original.endswith("\n"):
        updated += "\n"
    return updated


def _atomic_write(path: Path, content: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".my-agent-patch-",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode if mode is not None else 0o644)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
