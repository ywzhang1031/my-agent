from __future__ import annotations

import shlex


class PermissionPolicy:
    blocked_shell_tokens = {";", "&&", "||", "|", ">", ">>", "<", "`", "$("}

    def allow_test_command(self, command: str) -> tuple[bool, str, list[str]]:
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return False, f"could not parse command: {exc}", []
        if not argv:
            return False, "empty command is not allowed", []
        if any(token in command for token in self.blocked_shell_tokens):
            return False, "shell chaining or redirection is not allowed", argv
        if self._is_allowed_test_command(argv):
            return True, "", argv
        return False, f"command is not allowed in read-only diagnostic mode: {command}", argv

    def _is_allowed_test_command(self, argv: list[str]) -> bool:
        if argv[0] in {"pytest", "py.test"}:
            return True
        if argv[0] in {"python", "python3"}:
            return len(argv) >= 3 and argv[1] == "-m" and argv[2] in {"pytest", "unittest"}
        if argv[0] == "npm":
            return len(argv) >= 2 and argv[1] == "test"
        if argv[0] in {"pnpm", "yarn"}:
            return len(argv) >= 2 and argv[1] == "test"
        if argv[0] == "go":
            return len(argv) >= 2 and argv[1] == "test"
        if argv[0] == "cargo":
            return len(argv) >= 2 and argv[1] == "test"
        if argv[0] == "make":
            return len(argv) >= 2 and argv[1] in {"test", "tests", "check"}
        return False
