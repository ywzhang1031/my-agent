from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


class WorkspaceError(Exception):
    pass


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise WorkspaceError(f"workspace does not exist: {self.root}")
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace is not a directory: {self.root}")

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=False)
        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            raise WorkspaceError(f"path is outside workspace: {path}")
        return resolved

    def list_files(self, path: str = ".", max_files: int = 200) -> tuple[list[str], bool]:
        start = self.resolve(path)
        if not start.exists():
            raise WorkspaceError(f"path does not exist: {path}")
        if start.is_file():
            return [str(start.relative_to(self.root))], False
        files: list[str] = []
        for current_root, dirnames, filenames in os.walk(start):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in {".git", ".readonly-agent", "__pycache__", "node_modules"}
            ]
            for filename in filenames:
                full_path = Path(current_root) / filename
                files.append(str(full_path.relative_to(self.root)))
                if len(files) >= max_files:
                    return sorted(files), True
        return sorted(files), False

    def read_file(self, path: str, max_chars: int = 12000) -> tuple[str, bool, Path]:
        resolved = self.resolve(path)
        if not resolved.exists():
            raise WorkspaceError(f"file does not exist: {path}")
        if not resolved.is_file():
            raise WorkspaceError(f"path is not a file: {path}")
        text = resolved.read_text(encoding="utf-8")
        if len(text) <= max_chars:
            return text, False, resolved
        return text[:max_chars] + "\n...[truncated]", True, resolved

    def search(
        self,
        pattern: str,
        path: str = ".",
        literal: bool = False,
        max_results: int = 100,
    ) -> tuple[list[dict[str, Any]], bool]:
        if not pattern:
            raise WorkspaceError("pattern is required")
        start = self.resolve(path)
        if shutil.which("rg"):
            return self._search_with_rg(pattern, start, literal, max_results)
        return self._search_with_python(pattern, start, literal, max_results)

    def _search_with_rg(
        self,
        pattern: str,
        start: Path,
        literal: bool,
        max_results: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        args = ["rg", "--line-number", "--no-heading", "--color", "never"]
        if literal:
            args.append("--fixed-strings")
        args.extend([pattern, str(start)])
        completed = subprocess.run(args, text=True, capture_output=True, check=False)
        if completed.returncode not in {0, 1}:
            raise WorkspaceError(completed.stderr.strip() or "rg failed")
        matches: list[dict[str, Any]] = []
        for line in completed.stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            file_path, line_number, text = parts
            matches.append(
                {
                    "path": str(Path(file_path).resolve().relative_to(self.root)),
                    "line": int(line_number),
                    "text": text,
                }
            )
            if len(matches) >= max_results:
                return matches, True
        return matches, False

    def _search_with_python(
        self,
        pattern: str,
        start: Path,
        literal: bool,
        max_results: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        regex = re.compile(re.escape(pattern) if literal else pattern)
        files, _ = self.list_files(start.relative_to(self.root), max_files=5000)
        matches: list[dict[str, Any]] = []
        for rel_path in files:
            full_path = self.root / rel_path
            try:
                lines = full_path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for index, line in enumerate(lines, start=1):
                if regex.search(line):
                    matches.append({"path": rel_path, "line": index, "text": line})
                    if len(matches) >= max_results:
                        return matches, True
        return matches, False
