"""Fail-closed, offline macOS execution for the RPC Skill evaluation only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys


class IsolationError(RuntimeError):
    pass


def clean_environment(scratch: Path, workspace: Path, vendor: Path) -> dict[str, str]:
    home, temp = scratch / "home", scratch / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home), "TMPDIR": str(temp),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "LANG": "en_US.UTF-8", "TZ": "Asia/Shanghai",
        "PYTHONPATH": os.pathsep.join((str(workspace), str(vendor))),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    }


def profile(workspace: Path, scratch: Path, vendor: Path, trusted: Path | None = None) -> str:
    # No host-readable fallback. Python runs without site initialization (-I -S).
    roots = [p.resolve() for p in (workspace, scratch, vendor, trusted) if p is not None]
    broad = {Path("/"), Path("/Users"), Path.home().resolve(), Path("/tmp").resolve(),
             Path("/private/var"), Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]}
    if any(p in broad for p in roots):
        raise IsolationError("broad filesystem roots are not allowed")
    if len(set(roots)) != len(roots) or any(a != b and (a.is_relative_to(b) or b.is_relative_to(a)) for a in roots for b in roots):
        raise IsolationError("workspace, scratch, vendor and grader must be separate siblings")
    reads = [workspace, scratch, vendor, Path(sys.base_prefix).resolve(),
             Path("/System/Library"), Path("/System/Volumes/Preboot/Cryptexes/OS"),
             Path("/private/var/db/dyld"), Path("/usr/lib"), Path("/usr/share/zoneinfo"),
             Path("/opt/homebrew/Cellar"), Path("/opt/homebrew/opt")]
    if trusted is not None:
        reads.append(trusted)
    if any(p.resolve() == Path("/") for p in reads):
        raise IsolationError("recursive root access is forbidden")
    # On this macOS version, deny-default alone did not block KERN_PROCARGS2.
    clauses = ["(version 1)", "(deny default)", "(deny process-info*)", "(allow process-exec)",
               "(allow process-fork)", "(allow signal (target same-sandbox))",
               "(allow file-read-metadata)"]
    for name in ("hw.ncpu", "hw.activecpu", "hw.machine", "hw.model", "hw.memsize", "hw.pagesize", "hw.pagesize_compat",
                 "kern.osrelease", "kern.osversion", "kern.ostype", "kern.osproductversion", "kern.hostname", "kern.version",
                 "kern.bootargs", "kern.iossupportversion", "kern.argmax", "security.mac.lockdown_mode_state"):
        clauses.append(f"(allow sysctl-read (sysctl-name {json.dumps(name)}))")
    for path in reads:
        clauses.append(f"(allow file-read* (subpath {json.dumps(str(path.resolve()))}))")
    for path in (workspace, scratch):
        clauses.append(f"(allow file-write* (subpath {json.dumps(str(path.resolve()))}))")
    # dyld opens the root directory; literal '/' does not authorize its descendants.
    for path in ("/", "/dev/null", "/dev/random", "/dev/urandom", "/private/etc/localtime"):
        clauses.append(f"(allow file-read-data (literal {json.dumps(path)}))")
    clauses.append('(allow file-write-data (literal "/dev/null"))')
    return "\n".join(clauses)


def run_python(workspace: Path, scratch: Path, vendor: Path, code: str, *,
               args: tuple[str, ...] = (), trusted: Path | None = None,
               timeout: float = 90) -> subprocess.CompletedProcess[str]:
    if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise IsolationError("macOS Seatbelt is required; unsandboxed fallback is forbidden")
    env = clean_environment(scratch, workspace, vendor)
    command = ["/usr/bin/sandbox-exec", "-p", profile(workspace, scratch, vendor, trusted),
               str(Path(sys.executable).resolve()), "-I", "-S", "-c", code, *args]
    with subprocess.Popen(command, cwd=workspace, env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          close_fds=True, start_new_session=True) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise IsolationError("sandboxed worker timed out; process group killed") from None
        finally:
            # Also reap descendants left behind by an exited worker.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if "sandbox_apply:" in stderr:
        raise IsolationError("Seatbelt could not start (nested sandbox or host restriction)")
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


PROBE = r'''
import ctypes, errno, json, os, pathlib, socket, subprocess, sys
workspace, vendor, forbidden = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), json.loads(sys.argv[3])
checks = {}
def blocked(path, write=False):
    try:
        with open(path, "ab" if write else "rb") as f:
            if not write:
                f.read(1)
    except OSError as exc:
        return exc.errno in (errno.EPERM, errno.EACCES)
    return False
checks["workspace_write"] = (workspace / "probe.txt").write_text("probe") == 5
checks["workspace_read"] = (workspace / "probe.txt").read_text() == "probe"
checks["empty_credentials_environment"] = not any(k in os.environ for k in
    ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "MY_AGENT_CONFIG", "MY_AGENT_CREDENTIALS", "RPC_EVAL_SECRET"))
for name, path in forbidden.items():
    checks["deny_read_" + name] = blocked(path)
    checks["deny_write_" + name] = blocked(path, True)
link = workspace / "escape-link"
link.symlink_to(forbidden["host_sentinel"])
checks["deny_symlink_escape"] = blocked(link)
checks["readonly_vendor"] = blocked(vendor / "sentinel", True)
for family, address, label in (
    (socket.AF_INET, ("127.0.0.1", 9), "loopback"),
    (socket.AF_INET6, ("::1", 9), "loopback_v6"),
    (socket.AF_INET, ("192.0.2.1", 443), "external"),
):
    try:
        with socket.socket(family) as s:
            s.settimeout(0.2)
            s.connect(address)
    except OSError as exc:
        checks["deny_network_" + label] = exc.errno in (errno.EPERM, errno.EACCES)
    else:
        checks["deny_network_" + label] = False
child = subprocess.run([sys.executable, "-I", "-S", "-c",
    "import sys; open(sys.argv[1]).read()", forbidden["host_sentinel"]], capture_output=True, text=True)
checks["child_inherits_sandbox"] = child.returncode != 0 and "PermissionError" in child.stderr
# Darwin CTL_KERN/KERN_PROCARGS2: a clean environment must not be bypassed via the parent.
libc = ctypes.CDLL(None, use_errno=True)
mib = (ctypes.c_int * 3)(1, 49, os.getppid())
size = ctypes.c_size_t(262144)
buf = ctypes.create_string_buffer(size.value)
rc = libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0)
checks["deny_parent_process_environment"] = rc == -1 and ctypes.get_errno() in (errno.EPERM, errno.EACCES)
if not checks["deny_parent_process_environment"]:
    print("process environment probe: rc=%s errno=%s" % (rc, ctypes.get_errno()), file=sys.stderr)
print(json.dumps(checks))
sys.exit(0 if all(checks.values()) else 1)
'''


def check_isolation(workspace: Path, scratch: Path, vendor: Path,
                    forbidden: dict[str, Path]) -> dict[str, bool]:
    for path in forbidden.values():
        if not path.is_file():
            raise IsolationError("isolation probe requires existing, non-secret sentinel files")
    previous = os.environ.get("RPC_EVAL_SECRET")
    os.environ["RPC_EVAL_SECRET"] = "synthetic-environment-canary"
    try:
        result = run_python(workspace, scratch, vendor, PROBE,
                            args=(str(workspace), str(vendor), json.dumps({k: str(v) for k, v in forbidden.items()})))
    finally:
        if previous is None:
            os.environ.pop("RPC_EVAL_SECRET", None)
        else:
            os.environ["RPC_EVAL_SECRET"] = previous
    try:
        checks = json.loads(result.stdout)
    except ValueError:
        raise IsolationError(f"isolation probe did not complete (exit={result.returncode}): {result.stderr[-1500:]}") from None
    if result.returncode or not checks or not all(value is True for value in checks.values()):
        raise IsolationError(f"isolation probe failed: {checks}; {result.stderr[-500:]}")
    return checks
