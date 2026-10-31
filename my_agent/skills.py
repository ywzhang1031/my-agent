from __future__ import annotations

import hashlib
import os
import re
import stat
import unicodedata
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, BinaryIO

import yaml


SKILLS_DIRECTORY = ".agents/skills"
MAX_FRONTMATTER_BYTES = 8192
MAX_CATALOG_ENTRIES = 128
MAX_YAML_EVENTS = 1024
MAX_YAML_DEPTH = 16
_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: str
    frontmatter_sha256: str


@dataclass(frozen=True)
class SkillDiagnostic:
    path: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class SkillCatalog:
    status: str
    skills: tuple[SkillMetadata, ...] = ()
    diagnostics: tuple[SkillDiagnostic, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": SKILLS_DIRECTORY,
            "status": self.status,
            "model_visible": False,
            "skills": [asdict(skill) for skill in self.skills],
            "diagnostics": [asdict(issue) for issue in self.diagnostics],
        }


class _FrontmatterLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise ValueError("YAML mapping tag requires a mapping")
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings")
            if key in result:
                raise ValueError("duplicate YAML mapping key")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _parse_frontmatter(header: str) -> dict[str, Any]:
    # Bound structural complexity before construction; aliases are not needed here.
    depth = 0
    for count, event in enumerate(yaml.parse(header), start=1):
        if count > MAX_YAML_EVENTS:
            raise ValueError("frontmatter has too many YAML nodes")
        if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None):
            raise ValueError("YAML anchors and aliases are not supported")
        if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
            depth += 1
            if depth > MAX_YAML_DEPTH:
                raise ValueError("frontmatter YAML nesting is too deep")
        elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
            depth -= 1
    try:
        data = yaml.load(header, Loader=_FrontmatterLoader)
    except (ValueError, TypeError, OverflowError):
        # SafeLoader scalar conversion errors can include raw field contents.
        raise ValueError("invalid YAML mapping or scalar value") from None
    if not isinstance(data, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return data


def _read_frontmatter(source: BinaryIO) -> tuple[dict[str, Any], str]:
    opening = source.readline(MAX_FRONTMATTER_BYTES + 1)
    if opening.rstrip(b"\r\n") != b"---":
        raise ValueError("SKILL.md must begin with a --- frontmatter delimiter")
    raw = bytearray(opening)
    lines = []
    while True:
        line = source.readline(MAX_FRONTMATTER_BYTES - len(raw) + 1)
        raw.extend(line)
        if len(raw) > MAX_FRONTMATTER_BYTES:
            raise ValueError(f"frontmatter exceeds {MAX_FRONTMATTER_BYTES} bytes")
        if not line:
            raise ValueError("frontmatter is missing the closing --- delimiter")
        if line.rstrip(b"\r\n") == b"---":
            break
        lines.append(line)
    data = _parse_frontmatter(b"".join(lines).decode("utf-8"))
    return data, hashlib.sha256(raw).hexdigest()


def _open_directory(stack: ExitStack, name: str | Path, parent: int | None = None) -> int:
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    stack.callback(os.close, fd)
    return fd


def _read_skill(catalog_fd: int, name: str) -> tuple[SkillMetadata, bool]:
    if len(name) > 64 or not _NAME.fullmatch(name):
        raise ValueError("skill directory name must be 1-64 lowercase ASCII letters, digits or single hyphens")
    with ExitStack() as stack:
        directory_fd = _open_directory(stack, name, catalog_fd)
        fd = os.open("SKILL.md", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        with os.fdopen(fd, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError("SKILL.md must be a regular file")
            data, digest = _read_frontmatter(source)
    if data.get("name") != name:
        raise ValueError("frontmatter name must be a string matching its directory name")
    description = data.get("description")
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise ValueError("description must be a non-empty string of at most 1024 characters")
    if any(unicodedata.category(char) == "Cc" and char not in "\n\r\t" for char in description):
        raise ValueError("description must not contain control characters")
    return SkillMetadata(
        name=name, description=description.strip(), path=f"{SKILLS_DIRECTORY}/{name}/SKILL.md",
        frontmatter_sha256=digest,
    ), bool(data.keys() - {"name", "description"})


def discover_skills(workspace: Path) -> SkillCatalog:
    skills = []
    diagnostics = []
    with ExitStack() as stack:
        try:
            root_fd = _open_directory(stack, workspace)
            agents_fd = _open_directory(stack, ".agents", root_fd)
            catalog_fd = _open_directory(stack, "skills", agents_fd)
        except FileNotFoundError:
            return SkillCatalog(status="missing")
        except OSError as exc:
            return SkillCatalog(status="error", diagnostics=(SkillDiagnostic(
                SKILLS_DIRECTORY, f"Cannot open skill directory without following links: {exc.strerror}",
            ),))
        try:
            with os.scandir(catalog_fd) as entries:
                candidates = list(islice(entries, MAX_CATALOG_ENTRIES + 1))
            if len(candidates) > MAX_CATALOG_ENTRIES:
                return SkillCatalog(status="error", diagnostics=(SkillDiagnostic(
                    SKILLS_DIRECTORY, f"Directory exceeds {MAX_CATALOG_ENTRIES} entries; no partial catalog returned",
                ),))
            for entry in sorted(candidates, key=lambda item: item.name):
                path = f"{SKILLS_DIRECTORY}/{entry.name}/SKILL.md"
                # Undecodable filesystem bytes must not break JSON diagnostics.
                path = path.encode("utf-8", errors="backslashreplace").decode("utf-8")
                try:
                    if entry.is_symlink():
                        raise ValueError("skill directories must not be symbolic links")
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    skill, ignored_fields = _read_skill(catalog_fd, entry.name)
                    skills.append(skill)
                    if ignored_fields:
                        diagnostics.append(SkillDiagnostic(
                            path, "Additional frontmatter fields are ignored; they do not grant permissions", "warning",
                        ))
                except UnicodeDecodeError:
                    diagnostics.append(SkillDiagnostic(path, "frontmatter must be UTF-8"))
                except yaml.YAMLError as exc:
                    mark = getattr(exc, "problem_mark", None)
                    location = f" at line {mark.line + 2}" if mark is not None else ""
                    diagnostics.append(SkillDiagnostic(path, f"invalid YAML frontmatter{location}"))
                except ValueError as exc:
                    diagnostics.append(SkillDiagnostic(path, str(exc)))
                except OSError as exc:
                    diagnostics.append(SkillDiagnostic(path, f"Cannot read skill without following links: {exc.strerror}"))
        except OSError as exc:
            return SkillCatalog(status="error", diagnostics=(SkillDiagnostic(
                SKILLS_DIRECTORY, f"Cannot scan skill directory: {exc.strerror}",
            ),))
    return SkillCatalog(status="ready", skills=tuple(skills), diagnostics=tuple(diagnostics))
