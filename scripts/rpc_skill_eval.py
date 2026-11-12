"""Offline calibration only. No provider, credential loader, or paid-run entrypoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

from scripts.rpc_eval_sandbox import IsolationError, check_isolation, run_python


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "evals/rpc-skills"
BASE = "af9a86442a0f93e74356e9f321eefadd20d97ac2"
REFERENCE = "b71b2afa77e1f8ea7d9b187ece60bf9f751e80d8"
EXPORT_PATHS = ("my_agent", "tests", "cli.py", "requirements.txt")
REGRESSIONS = ("test_app_server.py", "test_persistence.py", "test_recovery.py", "test_service.py")


def hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def export_revision(revision: str, destination: Path) -> None:
    if revision not in (BASE, REFERENCE):
        raise ValueError("only the frozen historical revisions may be exported")
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", revision, *EXPORT_PATHS], cwd=ROOT)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for entry in stream:
            path = PurePosixPath(entry.name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] not in EXPORT_PATHS:
                raise ValueError("unsafe historical archive path")
            if entry.isdir():
                continue
            if not entry.isfile():
                raise ValueError("historical snapshot contains a link or special file")
            output = destination.joinpath(*path.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(stream.extractfile(entry).read())


def copy_vendor(destination: Path) -> None:
    destination.mkdir()
    for name in ("prompt_toolkit", "wcwidth", "yaml"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise IsolationError(f"install the project's existing dependencies first: missing {name}")
        source = Path(spec.origin).parent
        if any(p.is_symlink() for p in source.rglob("*")):
            raise IsolationError("dependency package contains symlinks")
        shutil.copytree(source, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "sentinel").write_text("read-only dependencies\n")


def grade(workspace: Path, scratch: Path, vendor: Path, trusted: Path) -> dict:
    paths = [str(workspace), str(vendor), str(trusted / "regressions")]
    code = ("import runpy,sys; sys.path[:0] = " + repr(paths) + "; "
            "sys.argv = ['acceptance.py', 'grade']; runpy.run_path(" +
            repr(str(trusted / "acceptance.py")) + ", run_name='__main__')")
    result = run_python(workspace, scratch, vendor, code, trusted=trusted, timeout=120)
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        raise IsolationError(f"grader did not produce a result: {result.stderr[-2000:]}") from None
    if result.returncode not in (0, 1) or payload.get("run") != len(payload.get("tests", [])):
        raise IsolationError("grader exited abnormally or omitted test results")
    payload["exit_code"] = result.returncode
    return payload


def acceptance(result: dict) -> list[dict]:
    return [test for test in result["tests"] if ".RpcAcceptance." in test["test"]]


def mutate(root: Path, kind: str) -> None:
    path = root / "my_agent/request_store.py"
    source = path.read_text()
    if kind == "raw-params":
        old = "fingerprint = hashlib.sha256(canonical.encode()).hexdigest()"
        new = "fingerprint = canonical"
    elif kind == "replay-unknown":
        old = '''return ResponseEnvelope.failure(request.request_id, code="request_outcome_unknown",
                message="Request was already admitted but its outcome is unavailable. Inspect the session; do not resubmit with a new ID blindly.")'''
        new = "return None"
    else:
        raise ValueError(kind)
    if source.count(old) != 1:
        raise ValueError("frozen reference mutation no longer matches")
    path.write_text(source.replace(old, new))


def calibration_ok(results: dict) -> bool:
    if set(results) != {"base", "reference", "raw-params", "replay-unknown"}:
        return False
    reference = results["reference"]
    expected = {row["test"] for row in reference["tests"]}
    if (len(acceptance(reference)) != 15 or not reference["successful"]
            or any(row["status"] != "passed" for row in reference["tests"])):
        return False
    for result in results.values():
        if {row["test"] for row in result["tests"]} != expected:
            return False
        if any(row["status"] in ("error", "skipped") for row in result["tests"]):
            return False
    if any(row["status"] != "passed" for row in results["base"]["tests"]
           if ".RpcAcceptance." not in row["test"]):
        return False
    failures = lambda name: {row["test"].rsplit(".", 1)[-1]
                             for row in acceptance(results[name]) if row["status"] == "failed"}
    base_required = {"test_duplicate_create", "test_restart_replays_response",
                     "test_crash_before_dispatch", "test_crash_after_side_effect"}
    return (base_required <= failures("base")
            and "test_secret_not_persisted_or_returned" in failures("raw-params")
            and {"test_crash_before_dispatch", "test_crash_after_side_effect"} <= failures("replay-unknown"))


def write_report(out: Path, manifest: dict, results: dict, status: str) -> None:
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    lines = ["# RPC Skill 离线校准", "", f"状态：{status}",
             "模型调用：0。以下是验收器/隔离的工程验证，不是 Skill 的模型成绩。", "",
             "| 实现 | 行为验收通过 | 原有回归通过 | error/skip |", "|---|---:|---:|---:|"]
    for name, result in results.items():
        checks = acceptance(result)
        original = [r for r in result["tests"] if r not in checks]
        score = lambda rows: f"{sum(r['status'] == 'passed' for r in rows)}/{len(rows)}"
        lines.append(f"| {name} | {score(checks)} | {score(original)} | "
                     f"{sum(r['status'] in ('error', 'skipped') for r in result['tests'])} |")
    if "reference" in results:
        lines += ["", "| 行为验收 | " + " | ".join(results) + " |",
                  "|---|" + "---|" * len(results)]
        for row in acceptance(results["reference"]):
            key = row["test"]
            statuses = [next((r["status"] for r in result["tests"] if r["test"] == key), "missing")
                        for result in results.values()]
            lines.append("| " + key.rsplit(".", 1)[-1] + " | " + " | ".join(statuses) + " |")
    lines += ["", "隔离探针：", *[f"- {name}: {passed}" for name, passed in manifest.get("isolation", {}).items()],
              "", "范围：历史 AppServer/存储/恢复/Service 回归，不包含浏览器、HTTP 服务或模型收益评测。",
              "隔离是 macOS Seatbelt 的 OS 权限边界，不是 VM；未验证恶意内核漏洞或恶意代码伪造评分。",
              "校准版本、依赖/输入哈希在 manifest.json；逐项失败证据在 results.json。"]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def calibrate(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = {"kind": "offline-calibration", "model_calls": 0, "base": BASE, "reference": REFERENCE,
                "started_at": datetime.now(timezone.utc).isoformat(), "spec_hashes": hashes(SPEC),
                "runner_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), ROOT / "scripts/rpc_eval_sandbox.py")}}
    results, status = {}, "blocked"
    try:
        with tempfile.TemporaryDirectory(prefix="rpc-skill-eval-") as temporary:
            root = Path(temporary).resolve()
            base, reference = root / "base", root / "reference"
            export_revision(BASE, base)
            export_revision(REFERENCE, reference)
            # Both variants get identical visible tests. The added historical solution tests stay out.
            shutil.rmtree(reference / "tests")
            shutil.copytree(base / "tests", reference / "tests")
            for workspace in (base, reference):
                shutil.copyfile(SPEC / "task.md", workspace / "TASK.md")
            manifest["snapshot_hashes"] = {"base": hashes(base), "reference": hashes(reference)}
            trusted = root / "trusted"
            trusted.mkdir()
            shutil.copyfile(SPEC / "acceptance.py", trusted / "acceptance.py")
            (trusted / "regressions").mkdir()
            for name in REGRESSIONS:
                shutil.copyfile(base / "tests" / name, trusted / "regressions" / name)
            vendor = root / "vendor"
            copy_vendor(vendor)
            manifest["dependency_hashes"] = hashes(vendor)
            sentinel = root / "host-sentinel"
            sentinel.write_text("not a real secret\n")
            probe_workspace = root / "probe"
            probe_workspace.mkdir()
            manifest["isolation"] = check_isolation(probe_workspace, root / "probe-scratch", vendor, {
                "host_sentinel": sentinel,
                "current_source": ROOT / "my_agent/request_store.py",
                "git_history": ROOT / ".git/HEAD",
                "grader": trusted / "acceptance.py",
                "reference": reference / "my_agent/request_store.py",
            })
            manifest["trusted_hashes"] = hashes(trusted)
            for name in ("base", "reference", "raw-params", "replay-unknown"):
                workspace = root / name
                if name not in ("base", "reference"):
                    shutil.copytree(reference, workspace)
                    mutate(workspace, name)
                results[name] = grade(workspace, root / f"{name}-scratch", vendor, trusted)
                if hashes(trusted) != manifest["trusted_hashes"] or hashes(vendor) != manifest["dependency_hashes"]:
                    raise IsolationError("trusted grading inputs were modified")
                write_report(out, manifest, results, "running")
            status = "passed" if calibration_ok(results) else "failed"
    except (IsolationError, OSError, ValueError, subprocess.SubprocessError) as exc:
        manifest["blocked_reason"] = str(exc)
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = status
        write_report(out, manifest, results, status)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New report directory; never overwrite previous evidence")
    args = parser.parse_args()
    os.umask(0o077)
    out = args.output or ROOT / ".my-agent/evals" / ("rpc-skills-offline-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    result = calibrate(out.resolve())
    print(out.resolve() / "report.md")
    if result["status"] != "passed":
        print(result.get("blocked_reason", "calibration checks failed"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
