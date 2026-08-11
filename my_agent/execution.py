from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


PASSTHROUGH_ENVIRONMENT_KEYS = {
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "WINDIR",
}


@dataclass(frozen=True)
class ProcessResult:
    status: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    truncated: bool = False
    timed_out: bool = False
    duration_ms: int = 0
    environment_keys: tuple[str, ...] = ()


class ProcessRunner:
    def __init__(self, max_output_bytes: int) -> None:
        self.max_output_bytes = max_output_bytes

    def run(
        self,
        argv: list[str],
        cwd: Path,
        timeout_seconds: int,
    ) -> ProcessResult:
        environment = _subprocess_environment()
        started = time.monotonic()

        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=os.name == "posix",
                )
            except OSError as exc:
                return ProcessResult(
                    status="spawn_error",
                    stderr=str(exc),
                    duration_ms=_duration_ms(started),
                    environment_keys=tuple(sorted(environment)),
                )

            timed_out = False
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_process_group(process)
                exit_code = process.wait()

            stdout, stdout_truncated = _read_capture(stdout_file, self.max_output_bytes)
            stderr, stderr_truncated = _read_capture(stderr_file, self.max_output_bytes)
            if timed_out and not stderr:
                stderr = f"command timed out after {timeout_seconds}s"

        return ProcessResult(
            status=(
                "timed_out"
                if timed_out
                else "success"
                if exit_code == 0
                else "nonzero_exit"
            ),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            truncated=stdout_truncated or stderr_truncated,
            timed_out=timed_out,
            duration_ms=_duration_ms(started),
            environment_keys=tuple(sorted(environment)),
        )


def _subprocess_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in PASSTHROUGH_ENVIRONMENT_KEYS
    }
    environment["PATH"] = _absolute_path_entries(environment.get("PATH", os.defpath))
    environment.update(
        {
            "CI": "1",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "NO_COLOR": "1",
            "PAGER": "cat",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _absolute_path_entries(value: str) -> str:
    entries = [entry for entry in value.split(os.pathsep) if entry and Path(entry).is_absolute()]
    return os.pathsep.join(entries) or os.defpath


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _read_capture(stream: BinaryIO, max_bytes: int) -> tuple[str, bool]:
    stream.seek(0)
    data = stream.read(max_bytes + 1)
    truncated = len(data) > max_bytes
    text = data[:max_bytes].decode("utf-8", errors="replace")
    if truncated:
        text += "\n...[truncated]"
    return text, truncated


def _duration_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
