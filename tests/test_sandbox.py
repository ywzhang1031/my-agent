from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from my_agent.execution import _process_status
from my_agent.sandbox import (
    DANGER_FULL_ACCESS,
    SANDBOX_PROFILES,
    WORKSPACE_WRITE,
    DangerFullAccessBackend,
    MacOSSeatbeltBackend,
    SandboxPolicy,
    SandboxUnavailableError,
    get_sandbox_profile,
    prepare_sandbox_command,
    sandbox_availability,
)


class SandboxTests(unittest.TestCase):
    def test_profile_catalog_describes_effective_security_boundaries(self) -> None:
        self.assertEqual(
            [profile.mode for profile in SANDBOX_PROFILES],
            [WORKSPACE_WRITE, DANGER_FULL_ACCESS],
        )

        workspace_write = get_sandbox_profile(WORKSPACE_WRITE)
        self.assertEqual(workspace_write.filesystem_read, "host-readable")
        self.assertEqual(workspace_write.filesystem_write, "workspace-and-temp")
        self.assertEqual(workspace_write.network, "denied")
        self.assertFalse(workspace_write.dangerous)

        full_access = get_sandbox_profile(DANGER_FULL_ACCESS)
        self.assertEqual(full_access.filesystem_write, "unrestricted")
        self.assertEqual(full_access.network, "allowed")
        self.assertTrue(full_access.dangerous)

        with self.assertRaisesRegex(ValueError, "unknown sandbox mode"):
            get_sandbox_profile("unknown")

    def test_availability_reports_backend_without_silent_fallback(self) -> None:
        unsupported = sandbox_availability(WORKSPACE_WRITE, platform="Linux")
        full_access = sandbox_availability(DANGER_FULL_ACCESS, platform="Linux")

        self.assertFalse(unsupported.available)
        self.assertIsNone(unsupported.backend)
        self.assertIn("not supported", unsupported.unavailable_reason)
        self.assertTrue(full_access.available)
        self.assertEqual(full_access.backend, "none")

    def test_danger_full_access_leaves_command_unchanged(self) -> None:
        policy = SandboxPolicy(mode=DANGER_FULL_ACCESS, workspace=Path("/tmp/workspace"))

        command = DangerFullAccessBackend().prepare(["pytest", "-q"], policy)

        self.assertEqual(command.argv, ("pytest", "-q"))
        self.assertEqual(command.mode, DANGER_FULL_ACCESS)
        self.assertEqual(command.backend, "none")

    def test_macos_seatbelt_wraps_command_with_workspace_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir, "workspace")
            workspace.mkdir()
            executable = Path(tmpdir, "sandbox-exec")
            executable.touch()
            policy = SandboxPolicy(mode=WORKSPACE_WRITE, workspace=workspace)

            command = MacOSSeatbeltBackend(executable).prepare(
                ["pytest", "-q"],
                policy,
            )

        self.assertEqual(command.argv[0], str(executable))
        self.assertIn("-p", command.argv)
        self.assertIn(f"WORKSPACE={workspace.resolve()}", command.argv)
        self.assertIn(f"GIT_DIR={(workspace / '.git').resolve()}", command.argv)
        self.assertIn(f"AGENT_DIR={(workspace / '.my-agent').resolve()}", command.argv)
        self.assertEqual(command.argv[-3:], ("--", "pytest", "-q"))
        self.assertEqual(command.mode, WORKSPACE_WRITE)
        self.assertEqual(command.backend, "macos-seatbelt")

    def test_workspace_write_fails_closed_on_unsupported_platform(self) -> None:
        policy = SandboxPolicy(mode=WORKSPACE_WRITE, workspace=Path("/tmp/workspace"))

        with self.assertRaisesRegex(SandboxUnavailableError, "not supported"):
            prepare_sandbox_command(["pytest", "-q"], policy, platform="Linux")

    def test_nested_seatbelt_failure_is_classified_as_sandbox_error(self) -> None:
        status = _process_status(
            exit_code=1,
            stderr="sandbox-exec: sandbox_apply: Operation not permitted",
            timed_out=False,
            interrupted=False,
            sandbox_backend="macos-seatbelt",
        )

        self.assertEqual(status, "sandbox_error")


if __name__ == "__main__":
    unittest.main()
