import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from my_agent.agent_loop import AgentLoop
from my_agent.cli import SLASH_COMMANDS, _print_skill_catalog, build_tool_registry
from my_agent.model_invoker import ModelInvoker
from my_agent.permissions import PermissionPolicy
from my_agent.provider import ModelMetadata, ModelResponse, ScriptedProviderAdapter
from my_agent.session import SessionStore
from my_agent.skills import MAX_CATALOG_ENTRIES, MAX_FRONTMATTER_BYTES, discover_skills
from my_agent.trace import TraceRecorder
from my_agent.workspace import Workspace


class SkillDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog = self.root / ".agents" / "skills"

    def skill(self, name="test-review", header=None, body="Private workflow body"):
        directory = self.catalog / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "SKILL.md"
        header = header if header is not None else f"name: {name}\ndescription: Review tests for missing cases."
        path.write_text(f"---\n{header}\n---\n{body}\n", encoding="utf-8")
        return path

    def test_missing_empty_and_ignores_non_directory_entries(self):
        self.assertEqual(discover_skills(self.root).status, "missing")
        self.catalog.mkdir(parents=True)
        (self.catalog / "README.md").write_text("not a skill")
        catalog = discover_skills(self.root)
        self.assertEqual(catalog.status, "ready")
        self.assertEqual(catalog.skills, ())
        self.assertEqual(catalog.diagnostics, ())

    def test_valid_sorted_catalog_only_exposes_metadata(self):
        self.skill("z-last")
        path = self.skill("a-first", "name: a-first\ndescription: >\n  Review tests.\n  Check edge cases.")
        catalog = discover_skills(self.root)
        self.assertEqual([skill.name for skill in catalog.skills], ["a-first", "z-last"])
        self.assertIn("Review tests. Check edge cases.", catalog.skills[0].description)
        self.assertEqual(catalog.skills[0].path, str(path.resolve()))
        self.assertNotIn("loaded", catalog.to_dict())
        self.assertNotIn("Private workflow body", json.dumps(catalog.to_dict()))
        digest = catalog.skills[0].frontmatter_sha256
        with path.open("ab") as source:
            source.write(b"\xff" * 100_000)
        self.assertEqual(discover_skills(self.root).skills[0].frontmatter_sha256, digest)
        self.skill("a-first", "name: a-first\ndescription: Changed description")
        self.assertNotEqual(discover_skills(self.root).skills[0].frontmatter_sha256, digest)

    def test_invalid_fields_are_isolated_and_reported(self):
        self.skill("valid")
        cases = {
            "missing-description": "name: missing-description",
            "wrong-name": "name: different\ndescription: explanation",
            "empty-description": "name: empty-description\ndescription: ' '",
            "numeric-description": "name: numeric-description\ndescription: 123",
            "long-description": "name: long-description\ndescription: " + "x" * 1025,
            "control-description": 'name: control-description\ndescription: "bad\\e[31m"',
            "duplicate": "name: duplicate\nname: duplicate\ndescription: explanation",
            "bad-yaml": "name: [\ndescription: explanation",
            "non-mapping": "- scalar",
            "Uppercase": "name: Uppercase\ndescription: explanation",
            "bad--name": "name: bad--name\ndescription: explanation",
            "n" * 65: "name: " + "n" * 65 + "\ndescription: explanation",
        }
        for name, header in cases.items():
            self.skill(name, header)
        catalog = discover_skills(self.root)
        self.assertEqual([skill.name for skill in catalog.skills], ["valid"])
        self.assertEqual(len(catalog.diagnostics), len(cases))
        self.assertTrue(all(issue.severity == "error" for issue in catalog.diagnostics))

    def test_description_utf8_and_name_string_types(self):
        self.skill("unicode-description", "name: unicode-description\ndescription: \u6392\u67e5\u6d4b\u8bd5\u5931\u8d25")
        self.skill("true", "name: 'true'\ndescription: 'Valid quoted string'")
        self.assertEqual(len(discover_skills(self.root).skills), 2)
        self.skill("true", "name: true\ndescription: valid")
        self.assertEqual(len(discover_skills(self.root).skills), 1)

    def test_optional_fields_are_not_permissions(self):
        self.skill(header="name: test-review\ndescription: Review tests\nallowed-tools: Bash(*)\nmetadata:\n  author: example")
        catalog = discover_skills(self.root)
        self.assertEqual(len(catalog.skills), 1)
        self.assertEqual(catalog.diagnostics[0].severity, "warning")
        self.assertNotIn("Bash(*)", json.dumps(catalog.to_dict()))
        self.assertIn("do not grant permissions", catalog.diagnostics[0].message)

    def test_unsafe_yaml_never_executes_and_rejects_aliases_or_deep_structures(self):
        sentinel = self.root / "must-not-exist"
        self.skill("unsafe", f"!!python/object/apply:os.system ['touch {sentinel}']")
        self.skill("aliases", "name: aliases\ndescription: &text example\nmetadata: *text")
        self.skill("recursive", "name: recursive\ndescription: ok\nmetadata: &x [*x]")
        self.skill("deep", "name: deep\ndescription: ok\nmetadata: " + "[" * 50 + "x" + "]" * 50)
        self.skill("wide", "name: wide\ndescription: ok\nmetadata: [" + ",".join("x" for _ in range(1200)) + "]")
        self.skill("nested-duplicate", "name: nested-duplicate\ndescription: ok\nmetadata: {a: x, a: y}")
        self.skill("wrong-tag", "!!map [name, description]")
        self.skill("bad-scalar", "name: bad-scalar\ndescription: !!int PRIVATE_SCALAR_CONTENT")
        catalog = discover_skills(self.root)
        self.assertEqual(catalog.skills, ())
        self.assertEqual(len(catalog.diagnostics), 8)
        self.assertNotIn("PRIVATE_SCALAR_CONTENT", json.dumps(catalog.to_dict()))
        self.assertFalse(sentinel.exists())

    def test_frontmatter_delimiters_encoding_and_size_limits(self):
        path = self.skill()
        for raw in (b"no delimiters", b"---\nname: test-review", b"---\n\xff\n---\n",
                    b"---\n" + b"x" * MAX_FRONTMATTER_BYTES + b"\n---\n"):
            with self.subTest(raw=raw[:40]):
                path.write_bytes(raw)
                catalog = discover_skills(self.root)
                self.assertEqual(catalog.skills, ())
                self.assertEqual(len(catalog.diagnostics), 1)
        path.write_bytes(b"---\r\nname: test-review\r\ndescription: Review tests\r\n---\r\nbody")
        self.assertEqual(len(discover_skills(self.root).skills), 1)

    def test_limits_catalog_without_returning_arbitrary_subset(self):
        self.catalog.mkdir(parents=True)
        for index in range(MAX_CATALOG_ENTRIES + 1):
            (self.catalog / f"entry-{index}").mkdir()
        catalog = discover_skills(self.root)
        self.assertEqual(catalog.status, "error")
        self.assertEqual(catalog.skills, ())
        self.assertIn("no partial catalog", catalog.diagnostics[0].message)

    def test_rejects_links_in_every_path_component(self):
        target = self.root / "outside"
        target.mkdir()
        for component in (".agents", ".agents/skills", ".agents/skills/example"):
            with self.subTest(component=component), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                workspace = Path(temporary)
                link = workspace / component
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(target, target_is_directory=True)
                catalog = discover_skills(workspace)
                self.assertEqual(catalog.skills, ())
                self.assertEqual(len(catalog.diagnostics), 1)
        path = self.skill()
        path.unlink()
        path.symlink_to(target / "SKILL.md")
        self.assertEqual(len(discover_skills(self.root).diagnostics), 1)

    def test_missing_or_special_skill_files_do_not_block_valid_entries(self):
        self.skill("valid")
        path = self.skill("fifo")
        path.unlink()
        os.mkfifo(path)
        directory = self.skill("directory")
        directory.unlink()
        directory.mkdir()
        missing = self.skill("missing")
        missing.unlink()
        catalog = discover_skills(self.root)
        self.assertEqual([s.name for s in catalog.skills], ["valid"])
        self.assertEqual(len(catalog.diagnostics), 3)

    def test_file_replaced_by_link_during_discovery_is_rejected(self):
        path = self.skill()
        outside = self.root / "outside.md"
        outside.write_text("secret")
        original_open = os.open
        def replace(name, *args, **kwargs):
            if name == "SKILL.md":
                path.unlink()
                path.symlink_to(outside)
            return original_open(name, *args, **kwargs)
        with patch("my_agent.skills.os.open", side_effect=replace):
            catalog = discover_skills(self.root)
        self.assertEqual(catalog.skills, ())
        self.assertEqual(len(catalog.diagnostics), 1)

    def test_unreadable_directory_is_an_explicit_diagnostic(self):
        with patch("my_agent.skills.os.open", side_effect=PermissionError(13, "Permission denied")):
            catalog = discover_skills(self.root)
        self.assertEqual(catalog.status, "error")
        self.assertIn("Permission denied", catalog.diagnostics[0].message)

    def test_isolated_discovery_includes_product_roots_but_not_parents_or_nested_projects(self):
        self.skill()
        child = self.root / "child"
        child.mkdir()
        self.assertEqual(discover_skills(child).status, "missing")
        nested = self.catalog / "container" / ".agents" / "skills" / "nested"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("---\nname: nested\ndescription: hidden\n---\n")
        for name in (".opencode", ".claude"):
            legacy = self.root / name / "skills" / "legacy"
            legacy.mkdir(parents=True)
            (legacy / "SKILL.md").write_text("---\nname: legacy\ndescription: hidden\n---\n")
        catalog = discover_skills(self.root)
        self.assertEqual([s.name for s in catalog.skills], ["legacy", "test-review"])
        self.assertTrue(any("Shadowed by" in issue.message for issue in catalog.diagnostics))

    def test_cli_lists_metadata_and_escapes_diagnostic_paths(self):
        self.skill()
        self.skill("bad\x1bname")
        catalog = discover_skills(self.root)
        output = io.StringIO()
        _print_skill_catalog(catalog, file=output)
        rendered = output.getvalue()
        self.assertIn("metadata only", rendered)
        self.assertIn("Review tests", rendered)
        self.assertIn("error:", rendered)
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("Private workflow body", rendered)
        self.assertIn("/skills", [command.name for command in SLASH_COMMANDS])

    def test_undecodable_directory_names_produce_json_safe_diagnostics(self):
        self.catalog.mkdir(parents=True)
        entry = Mock()
        entry.name = "bad\udcff"
        entry.is_symlink.return_value = False
        entry.is_dir.return_value = True
        # Do not require the host filesystem to support non-UTF-8 names.
        with patch("my_agent.skills.os.scandir") as scan:
            scan.return_value.__enter__.return_value = iter([entry])
            catalog = discover_skills(self.root)
        self.assertEqual(catalog.skills, ())
        self.assertEqual(len(catalog.diagnostics), 1)
        self.assertIn("\\udcff", catalog.diagnostics[0].path)
        json.dumps(catalog.to_dict(), ensure_ascii=False).encode("utf-8")

    def test_turn_discovery_injects_metadata_but_not_body_or_scripts(self):
        path = self.skill(header="name: test-review\ndescription: UNIQUE_SKILL_DESCRIPTION")
        sentinel = self.root / "never-created"
        script = path.parent / "scripts" / "run.sh"
        script.parent.mkdir()
        script.write_text(f"touch {sentinel}\n")
        script.chmod(0o700)
        catalog = discover_skills(self.root)
        self.assertEqual(len(catalog.skills), 1)
        store = SessionStore(self.root / ".my-agent" / "sessions")
        state = store.create(self.root)
        provider = ScriptedProviderAdapter([ModelResponse.from_parts(text="Done", metadata=ModelMetadata(provider_id="scripted", model_id="scripted-model"))])
        tools = build_tool_registry()
        before = tools.names()
        loop = AgentLoop(workspace=Workspace(self.root), model=ModelInvoker(provider), tools=tools,
                         permissions=PermissionPolicy(), trace=TraceRecorder(state.trace_path))
        loop.run_turn(state, "Hello")
        request = provider.requests[0]
        self.assertEqual(tools.names(), before)
        self.assertIn("load_skill", before)
        contents = request.system_prompt + "".join(m.content for m in request.messages)
        self.assertIn("UNIQUE_SKILL_DESCRIPTION", contents)
        self.assertNotIn("Private workflow body", contents)
        self.assertFalse(sentinel.exists())
