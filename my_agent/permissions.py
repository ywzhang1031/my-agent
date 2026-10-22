from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Any


MAX_COMMAND_ARGUMENTS = 128
MAX_COMMAND_ARGUMENT_CHARS = 4096


class PermissionEffect(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionRule:
    action: str
    resource_pattern: str
    effect: PermissionEffect

    def matches(self, action: str, resource: str) -> bool:
        return fnmatchcase(action, self.action) and fnmatchcase(
            resource,
            self.resource_pattern,
        )


@dataclass(frozen=True)
class PermissionProfile:
    profile_id: str
    description: str
    rules: tuple[PermissionRule, ...]
    default_effect: PermissionEffect = PermissionEffect.DENY

    def decide(self, action: str, resource: str) -> PermissionEffect:
        effect = self.default_effect
        for rule in self.rules:
            if rule.matches(action, resource):
                effect = rule.effect
        return effect


SUPERVISED_PROFILE = PermissionProfile(
    profile_id="supervised",
    description="Ask before workspace edits and allowlisted commands",
    rules=(
        PermissionRule("apply_patch", "*", PermissionEffect.ASK),
        PermissionRule("exec_command", "*", PermissionEffect.ASK),
    ),
)

ACCEPT_EDITS_PROFILE = PermissionProfile(
    profile_id="accept_edits",
    description="Allow workspace edits; ask before allowlisted commands",
    rules=(
        PermissionRule("apply_patch", "*", PermissionEffect.ALLOW),
        PermissionRule("exec_command", "*", PermissionEffect.ASK),
    ),
)

TRUSTED_PROFILE = PermissionProfile(
    profile_id="trusted",
    description="Allow edits and commands that pass the hard safety policy",
    rules=(
        PermissionRule("apply_patch", "*", PermissionEffect.ALLOW),
        PermissionRule("exec_command", "*", PermissionEffect.ALLOW),
    ),
)

PERMISSION_PROFILES = (
    SUPERVISED_PROFILE,
    ACCEPT_EDITS_PROFILE,
    TRUSTED_PROFILE,
)
DEFAULT_PERMISSION_PROFILE_ID = SUPERVISED_PROFILE.profile_id
_PROFILES_BY_ID = {profile.profile_id: profile for profile in PERMISSION_PROFILES}


def get_permission_profile(profile_id: str | None) -> PermissionProfile:
    selected = profile_id or DEFAULT_PERMISSION_PROFILE_ID
    try:
        return _PROFILES_BY_ID[selected]
    except KeyError as exc:
        raise ValueError(f"unknown permission profile: {selected}") from exc


@dataclass(frozen=True)
class CommandDecision:
    outcome: str
    reason: str
    argv: tuple[str, ...] = ()
    category: str | None = None

    @property
    def allowed(self) -> bool:
        return self.outcome == "allow"


class PermissionPolicy:
    blocked_shell_tokens = {";", "&&", "||", "|", "|&", "&", ">", ">>", "<"}
    protected_path_parts = {".git", ".my-agent"}

    def __init__(self, profile_id: str | None = None) -> None:
        self._profile = get_permission_profile(profile_id)

    @property
    def profile(self) -> PermissionProfile:
        return self._profile

    def switch_profile(self, profile_id: str) -> PermissionProfile:
        self._profile = get_permission_profile(profile_id)
        return self._profile

    def decide_permission(self, action: str, resource: str) -> PermissionEffect:
        return self._profile.decide(action, resource)

    def allow_file_change(self, operation: str, path: str) -> tuple[bool, str]:
        if operation not in {"add", "update", "delete"}:
            return False, f"unsupported file operation: {operation}"
        if any(part.casefold() in self.protected_path_parts for part in PurePosixPath(path).parts):
            return False, f"file operation targets a protected path: {path}"
        return True, ""

    def decide_command(self, value: Any) -> CommandDecision:
        invalid_reason = self._validate_argv(value)
        if invalid_reason:
            return CommandDecision(outcome="invalid", reason=invalid_reason)

        argv = tuple(value)
        if any(argument in self.blocked_shell_tokens or "\n" in argument for argument in argv):
            return CommandDecision(
                outcome="deny",
                reason="shell operators and newlines are not supported; pass one process argv",
                argv=argv,
            )
        if "/" in argv[0] or "\\" in argv[0]:
            return CommandDecision(
                outcome="deny",
                reason="executable paths are not allowed; use an allowlisted executable name",
                argv=argv,
            )

        category = self._command_category(argv)
        if category is None:
            return CommandDecision(
                outcome="deny",
                reason=f"command is not allowed by the exec-command policy: {list(argv)!r}",
                argv=argv,
            )
        return CommandDecision(
            outcome="allow",
            reason="",
            argv=argv,
            category=category,
        )

    def _validate_argv(self, value: Any) -> str:
        if not isinstance(value, list):
            return "argv must be an array of strings"
        if not value:
            return "argv must contain at least one argument"
        if len(value) > MAX_COMMAND_ARGUMENTS:
            return f"argv exceeds the {MAX_COMMAND_ARGUMENTS}-argument limit"
        for argument in value:
            if not isinstance(argument, str) or not argument:
                return "every argv item must be a non-empty string"
            if "\x00" in argument:
                return "argv must not contain NUL bytes"
            if len(argument) > MAX_COMMAND_ARGUMENT_CHARS:
                return (
                    "argv item exceeds the "
                    f"{MAX_COMMAND_ARGUMENT_CHARS}-character limit"
                )
        return ""

    def _command_category(self, argv: tuple[str, ...]) -> str | None:
        executable = argv[0]
        if executable in {"pytest", "py.test"}:
            return "test"
        if executable in {"python", "python3"}:
            return self._python_module_category(argv)
        if executable in {"npm", "pnpm", "yarn"}:
            return self._package_script_category(argv)
        if executable == "go" and len(argv) >= 2:
            return {"test": "test", "vet": "check", "build": "build"}.get(argv[1])
        if executable == "cargo" and len(argv) >= 2:
            return {
                "test": "test",
                "check": "check",
                "clippy": "check",
                "build": "build",
            }.get(argv[1])
        if executable == "make" and len(argv) >= 2:
            return {
                "test": "test",
                "tests": "test",
                "check": "check",
            }.get(argv[1])
        if executable == "ruff" and len(argv) >= 2:
            if argv[1] == "check" and not _contains_flag(argv, "--fix", "--fix-only"):
                return "check"
            if argv[1] == "format" and "--check" in argv:
                return "check"
        if executable == "black" and "--check" in argv:
            return "check"
        if executable in {"mypy", "pyright"}:
            return "check"
        if executable == "eslint" and not _contains_flag(argv, "--fix"):
            return "check"
        if executable == "tsc" and "--noEmit" in argv:
            return "check"
        return None

    def _python_module_category(self, argv: tuple[str, ...]) -> str | None:
        if len(argv) < 3 or argv[1] != "-m":
            return None
        module = argv[2]
        if module in {"pytest", "unittest"}:
            return "test"
        if module == "compileall":
            return "check"
        if module in {"mypy", "pyright"}:
            return "check"
        if module == "black" and "--check" in argv[3:]:
            return "check"
        if module == "ruff" and len(argv) >= 4:
            if argv[3] == "check" and not _contains_flag(argv, "--fix", "--fix-only"):
                return "check"
            if argv[3] == "format" and "--check" in argv[4:]:
                return "check"
        return None

    def _package_script_category(self, argv: tuple[str, ...]) -> str | None:
        if len(argv) >= 2 and argv[1] == "test":
            return "test"
        if len(argv) < 3 or argv[1] != "run":
            return None
        return {
            "test": "test",
            "lint": "check",
            "typecheck": "check",
            "check": "check",
            "build": "build",
        }.get(argv[2])


def _contains_flag(argv: tuple[str, ...], *flags: str) -> bool:
    return any(
        argument == flag or argument.startswith(f"{flag}=")
        for argument in argv
        for flag in flags
    )
