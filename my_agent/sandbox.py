from __future__ import annotations

import platform as platform_module
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


WORKSPACE_WRITE = "workspace-write"
DANGER_FULL_ACCESS = "danger-full-access"
DEFAULT_SANDBOX_MODE = WORKSPACE_WRITE

SEATBELT_EXECUTABLE = Path("/usr/bin/sandbox-exec")
SEATBELT_PROFILE = """(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))
(allow file-read*)
(allow file-write*
  (subpath (param "WORKSPACE"))
  (subpath (param "USER_TEMP"))
  (subpath "/private/tmp"))
(deny file-write*
  (subpath (param "GIT_DIR"))
  (subpath (param "AGENT_DIR")))
(allow file-write-data (literal "/dev/null"))
(allow file-ioctl)
(allow sysctl-read)
(allow mach*)
(allow ipc*)
(allow iokit*)
(allow pseudo-tty)
"""


class SandboxUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxProfile:
    mode: str
    description: str
    filesystem_read: str
    filesystem_write: str
    network: str
    dangerous: bool


@dataclass(frozen=True)
class SandboxAvailability:
    available: bool
    backend: str | None
    unavailable_reason: str | None = None


SANDBOX_PROFILES = (
    SandboxProfile(
        mode=WORKSPACE_WRITE,
        description=(
            "Host files are readable; writes are limited to the workspace and temp "
            "directories, with network access denied."
        ),
        filesystem_read="host-readable",
        filesystem_write="workspace-and-temp",
        network="denied",
        dangerous=False,
    ),
    SandboxProfile(
        mode=DANGER_FULL_ACCESS,
        description=(
            "No operating-system sandbox; commands inherit the agent process access."
        ),
        filesystem_read="unrestricted",
        filesystem_write="unrestricted",
        network="allowed",
        dangerous=True,
    ),
)
SANDBOX_MODES = tuple(profile.mode for profile in SANDBOX_PROFILES)


def get_sandbox_profile(mode: str) -> SandboxProfile:
    for profile in SANDBOX_PROFILES:
        if profile.mode == mode:
            return profile
    raise ValueError(f"unknown sandbox mode: {mode}")


def sandbox_availability(
    mode: str,
    *,
    platform: str | None = None,
) -> SandboxAvailability:
    get_sandbox_profile(mode)
    if mode == DANGER_FULL_ACCESS:
        return SandboxAvailability(available=True, backend="none")
    selected_platform = platform or platform_module.system()
    if selected_platform != "Darwin":
        return SandboxAvailability(
            available=False,
            backend=None,
            unavailable_reason=f"not supported on {selected_platform}",
        )
    if not SEATBELT_EXECUTABLE.is_file():
        return SandboxAvailability(
            available=False,
            backend=None,
            unavailable_reason=f"sandbox executable not found: {SEATBELT_EXECUTABLE}",
        )
    return SandboxAvailability(available=True, backend="macos-seatbelt")


@dataclass(frozen=True)
class SandboxPolicy:
    mode: str
    workspace: Path

    def __post_init__(self) -> None:
        get_sandbox_profile(self.mode)
        object.__setattr__(self, "workspace", self.workspace.expanduser().resolve())


@dataclass(frozen=True)
class SandboxCommand:
    argv: tuple[str, ...]
    mode: str
    backend: str


class SandboxBackend(Protocol):
    def prepare(
        self,
        argv: Sequence[str],
        policy: SandboxPolicy,
    ) -> SandboxCommand:
        ...


class DangerFullAccessBackend:
    def prepare(
        self,
        argv: Sequence[str],
        policy: SandboxPolicy,
    ) -> SandboxCommand:
        return SandboxCommand(
            argv=tuple(argv),
            mode=policy.mode,
            backend="none",
        )


class MacOSSeatbeltBackend:
    def __init__(self, executable: str | Path = SEATBELT_EXECUTABLE) -> None:
        self.executable = Path(executable)

    def prepare(
        self,
        argv: Sequence[str],
        policy: SandboxPolicy,
    ) -> SandboxCommand:
        if not self.executable.is_file():
            raise SandboxUnavailableError(
                f"macOS sandbox executable not found: {self.executable}"
            )
        workspace = policy.workspace
        parameters = (
            f"WORKSPACE={workspace}",
            f"USER_TEMP={Path(tempfile.gettempdir()).resolve()}",
            f"GIT_DIR={(workspace / '.git').resolve()}",
            f"AGENT_DIR={(workspace / '.my-agent').resolve()}",
        )
        wrapped = [str(self.executable)]
        for parameter in parameters:
            wrapped.extend(("-D", parameter))
        wrapped.extend(("-p", SEATBELT_PROFILE, "--", *argv))
        return SandboxCommand(
            argv=tuple(wrapped),
            mode=policy.mode,
            backend="macos-seatbelt",
        )


def prepare_sandbox_command(
    argv: Sequence[str],
    policy: SandboxPolicy,
    *,
    platform: str | None = None,
) -> SandboxCommand:
    if policy.mode == DANGER_FULL_ACCESS:
        return DangerFullAccessBackend().prepare(argv, policy)
    selected_platform = platform or platform_module.system()
    if selected_platform == "Darwin":
        return MacOSSeatbeltBackend().prepare(argv, policy)
    raise SandboxUnavailableError(
        f"sandbox mode {policy.mode!r} is not supported on {selected_platform}"
    )
