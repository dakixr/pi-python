from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from difflib import unified_diff
from fnmatch import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, ClassVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from pi.agent.models import ToolCall
from pi.agent.truncate import DEFAULT_MAX_BYTES, format_size, truncate_head, truncate_line, truncate_tail

DEFAULT_GLOB = "*"


class ReplaceEdit(BaseModel):
    old_text: str = Field(
        min_length=1,
        validation_alias=AliasChoices("oldText", "old_text"),
        serialization_alias="oldText",
    )
    new_text: str = Field(
        validation_alias=AliasChoices("newText", "new_text"),
        serialization_alias="newText",
    )


class BaseTool(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: ClassVar[str]
    description: ClassVar[str]
    _context: dict[str, object] = PrivateAttr(default_factory=dict)

    @classmethod
    def bind(cls, **context: object) -> "BaseTool":
        tool = cls.model_construct()
        tool._bind_context(**context)
        return tool

    def _bind_context(self, **context: object) -> None:
        self._context = dict(context)

    def _inherit_context(self, prototype: "BaseTool") -> None:
        self._bind_context(**prototype._context)

    def to_definition(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.__class__.model_json_schema(by_alias=True),
            },
        }

    def parse_arguments(self, arguments: dict[str, object]) -> "BaseTool":
        parsed = self.__class__.model_validate(arguments)
        parsed._inherit_context(self)
        return parsed

    @abstractmethod
    def execute(self) -> dict[str, object]:
        raise NotImplementedError


class WorkspaceTool(BaseTool, ABC):
    _root: Path = PrivateAttr()

    def _bind_context(self, **context: object) -> None:
        root = context.get("root")
        if root is None:
            raise ValueError("Workspace root is required")
        if not isinstance(root, (str, os.PathLike)):
            raise ValueError("Workspace root must be path-like")
        candidate = Path(root).resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise ValueError(f"Workspace root does not exist or is not a directory: {root}")
        self._context = {"root": candidate}
        self._root = candidate

    @property
    def root(self) -> Path:
        return self._root

    def _resolve_path(self, raw_path_value: str, *, allow_missing: bool = False) -> Path:
        if not raw_path_value.strip():
            raise ValueError("Path must be a non-empty path")
        raw_path = Path(raw_path_value)
        candidate = raw_path.resolve() if raw_path.is_absolute() else (self.root / raw_path).resolve()
        if not allow_missing and not candidate.exists():
            raise FileNotFoundError(raw_path_value)
        return candidate

    def _ensure_text_file(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {self._display_path(path)}")

    def _success(self, **payload: object) -> dict[str, object]:
        return {"ok": True, **payload}

    def _error(self, error: str, **payload: object) -> dict[str, object]:
        return {"ok": False, "error": error, **payload}

    def _relative(self, path: Path) -> str:
        return self._display_path(path)

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def _read_existing_text(self, path: Path) -> str | None:
        if not path.exists() or not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None

    def _walk(self, path: Path) -> Iterable[Path]:
        if path.is_file():
            yield path
            return
        yield path
        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(entry for entry in dirs if entry != ".git")
            current_root = Path(root)
            for name in sorted(files):
                yield current_root / name
            for name in dirs:
                yield current_root / name

    def _iter_files(self, path: Path, *, glob: str = DEFAULT_GLOB) -> Iterable[Path]:
        if path.is_file():
            if fnmatch(path.name, glob):
                yield path
            return

        git_candidates = self._iter_git_files(path, glob=glob)
        if git_candidates is not None:
            yield from git_candidates
            return

        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(entry for entry in dirs if entry != ".git")
            current_root = Path(root)
            for name in sorted(files):
                if fnmatch(name, glob):
                    yield current_root / name

    def _iter_git_files(self, path: Path, *, glob: str) -> Iterable[Path] | None:
        try:
            path.relative_to(self.root)
        except ValueError:
            return None
        try:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.root),
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "--full-name",
                    "--",
                    self._git_pathspec(path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None

        resolved_path = path.resolve()
        seen: set[Path] = set()
        candidates: list[Path] = []
        for line in completed.stdout.splitlines():
            candidate = (self.root / line).resolve()
            if not candidate.exists() or not candidate.is_file():
                continue
            try:
                candidate.relative_to(resolved_path)
            except ValueError:
                if candidate != resolved_path:
                    continue
            if candidate not in seen and fnmatch(candidate.name, glob):
                seen.add(candidate)
                candidates.append(candidate)
        return candidates

    def _git_pathspec(self, path: Path) -> str:
        relative = path.relative_to(self.root)
        if not relative.parts:
            return "."
        return str(relative)


def build_unified_diff(path: str, before: str, after: str, *, created: bool = False) -> tuple[str | None, bool]:
    diff_lines = list(
        unified_diff(
            [] if created else before.splitlines(),
            after.splitlines(),
            fromfile="/dev/null" if created else f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    if not diff_lines:
        return None, False
    diff = "\n".join(diff_lines)
    truncation = truncate_head(diff)
    return truncation.content, truncation.truncated


class ReadTool(WorkspaceTool):
    name = "read"
    description = (
        "Read a UTF-8 text file. Relative paths resolve from the configured root. "
        "Output is truncated to 2000 lines or 50KB. Use offset/limit to continue."
    )

    path: str = Field(min_length=1)
    offset: int | None = Field(default=None, ge=1)
    limit: int | None = Field(default=None, ge=1)

    def execute(self) -> dict[str, object]:
        try:
            path = self._resolve_path(self.path)
            self._ensure_text_file(path)
            content = path.read_text(encoding="utf-8")
            all_lines = content.split("\n")
            start = (self.offset or 1) - 1
            if start >= len(all_lines):
                raise ValueError(f"Offset {self.offset} is beyond end of file ({len(all_lines)} lines total)")
            selected_content = (
                "\n".join(all_lines[start : start + self.limit])
                if self.limit is not None
                else "\n".join(all_lines[start:])
            )
            truncation = truncate_head(selected_content)
            start_line = start + 1
            next_offset = None
            output_text = truncation.content
            details: dict[str, object] | None = None

            if truncation.first_line_exceeds_limit:
                first_line_size = format_size(len(all_lines[start].encode("utf-8")))
                output_text = (
                    f"[Line {start_line} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                    f"Use bash: sed -n '{start_line}p' {self.path} | head -c {DEFAULT_MAX_BYTES}]"
                )
                details = {"truncation": truncation.to_dict()}
            elif truncation.truncated:
                end_line = start_line + truncation.output_lines - 1
                next_offset = end_line + 1
                suffix = (
                    f"[Showing lines {start_line}-{end_line} of {len(all_lines)}. Use offset={next_offset} to continue.]"
                    if truncation.truncated_by == "lines"
                    else (
                        f"[Showing lines {start_line}-{end_line} of {len(all_lines)} "
                        f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                    )
                )
                output_text = f"{truncation.content}\n\n{suffix}"
                details = {"truncation": truncation.to_dict()}
            elif self.limit is not None and start + self.limit < len(all_lines):
                next_offset = start + self.limit + 1
                remaining = len(all_lines) - (start + self.limit)
                output_text = f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"

            return self._success(
                path=self.path,
                content=output_text,
                offset=self.offset or 1,
                limit=self.limit,
                next_offset=next_offset,
                details=details,
            )
        except Exception as exc:
            return self._error(str(exc), path=self.path)


class WriteTool(WorkspaceTool):
    name = "write"
    description = "Write a UTF-8 text file. Relative paths resolve from the configured root."

    path: str = Field(min_length=1)
    content: str

    def execute(self) -> dict[str, object]:
        try:
            path = self._resolve_path(self.path, allow_missing=True)
            existed = path.exists()
            previous_content = self._read_existing_text(path) if existed else ""
            if existed and not path.is_file():
                raise ValueError(f"Path is not a regular file: {self.path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.content, encoding="utf-8")
            diff = None
            diff_truncated = False
            if previous_content is not None:
                diff, diff_truncated = build_unified_diff(self._relative(path), previous_content, self.content, created=not existed)
            return self._success(
                path=self.path,
                bytes_written=len(self.content.encode("utf-8")),
                created=not existed,
                diff=diff,
                diff_truncated=diff_truncated,
            )
        except Exception as exc:
            return self._error(str(exc), path=self.path)


class EditTool(WorkspaceTool):
    name = "edit"
    description = "Replace exact text in a UTF-8 file. Use oldText/newText for one change or edits for multiple disjoint changes."

    path: str = Field(min_length=1)
    old_text: str | None = Field(default=None, validation_alias=AliasChoices("oldText", "old_text"), serialization_alias="oldText")
    new_text: str | None = Field(default=None, validation_alias=AliasChoices("newText", "new_text"), serialization_alias="newText")
    edits: list[ReplaceEdit] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_edit_mode(self) -> "EditTool":
        single_mode = self.old_text is not None or self.new_text is not None
        multi_mode = bool(self.edits)
        if single_mode and multi_mode:
            raise ValueError("Use either oldText/newText or edits, not both.")
        if not single_mode and not multi_mode:
            raise ValueError("Provide either oldText/newText or edits.")
        if single_mode and (self.old_text is None or self.new_text is None):
            raise ValueError("Single replacement mode requires both oldText and newText.")
        return self

    def normalized_edits(self) -> list[ReplaceEdit]:
        if self.edits:
            return self.edits
        assert self.old_text is not None
        assert self.new_text is not None
        return [ReplaceEdit(old_text=self.old_text, new_text=self.new_text)]

    def execute(self) -> dict[str, object]:
        try:
            path = self._resolve_path(self.path)
            self._ensure_text_file(path)
            original = path.read_text(encoding="utf-8")
            resolved_edits: list[tuple[int, int, str, str]] = []
            for edit in self.normalized_edits():
                start = original.find(edit.old_text)
                if start == -1:
                    return self._error(f"Text not found in {self.path}", path=self.path)
                if original.find(edit.old_text, start + 1) != -1:
                    return self._error(f"Text to replace is not unique in {self.path}", path=self.path)
                resolved_edits.append((start, start + len(edit.old_text), edit.old_text, edit.new_text))
            resolved_edits.sort(key=lambda item: item[0])
            for current, following in zip(resolved_edits, resolved_edits[1:]):
                if current[1] > following[0]:
                    return self._error(f"Edits overlap in {self.path}. Merge nearby changes into one edit.", path=self.path)
            updated = original
            for start, end, old_text, new_text in reversed(resolved_edits):
                if updated[start:end] != old_text:
                    return self._error(f"Edit no longer matches file content in {self.path}", path=self.path)
                updated = updated[:start] + new_text + updated[end:]
            path.write_text(updated, encoding="utf-8")
            diff, diff_truncated = build_unified_diff(self._relative(path), original, updated)
            return self._success(path=self.path, edits_applied=len(resolved_edits), diff=diff, diff_truncated=diff_truncated)
        except Exception as exc:
            return self._error(str(exc), path=self.path)


class BashTool(WorkspaceTool):
    name = "bash"
    description = (
        "Run a shell command in the workspace root. Output keeps the last 2000 lines or 50KB. "
        "When truncated, full output is saved to a temp file."
    )

    command: str = Field(min_length=1)
    timeout: int | None = Field(default=None, ge=1, validation_alias=AliasChoices("timeout", "timeout_seconds"))

    def execute(self) -> dict[str, object]:
        try:
            completed = subprocess.run(
                ["bash", "-lc", self.command],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
                timeout=self.timeout,
            )
            return self._build_result(
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            result = self._build_result(
                returncode=-1,
                stdout=_coerce_subprocess_output(exc.stdout),
                stderr=_coerce_subprocess_output(exc.stderr),
            )
            result["ok"] = False
            result["error"] = f"Command timed out after {self.timeout} seconds"
            return result
        except Exception as exc:
            return self._error(str(exc), command=self.command, timeout=self.timeout)

    def _build_result(self, *, returncode: int, stdout: str, stderr: str) -> dict[str, object]:
        combined_parts = [part.rstrip("\n") for part in (stdout, stderr) if part]
        combined_output = "\n".join(part for part in combined_parts if part)
        combined_truncation = truncate_tail(combined_output)
        stdout_truncation = truncate_tail(stdout.rstrip("\n"))
        stderr_truncation = truncate_tail(stderr.rstrip("\n"))
        full_output_path = None
        if combined_truncation.truncated:
            fd, full_output_path = tempfile.mkstemp(prefix="pi-bash-", suffix=".log")
            os.close(fd)
            Path(full_output_path).write_text(combined_output, encoding="utf-8")
        payload = {
            "ok": returncode == 0,
            "command": self.command,
            "returncode": returncode,
            "stdout": stdout_truncation.content.rstrip("\n"),
            "stderr": stderr_truncation.content.rstrip("\n"),
            "output": combined_truncation.content.rstrip("\n"),
            "stdout_truncated": stdout_truncation.truncated,
            "stderr_truncated": stderr_truncation.truncated,
            "timeout": self.timeout,
            "details": {
                "truncation": combined_truncation.to_dict(),
                "fullOutputPath": full_output_path,
            },
        }
        if full_output_path is not None:
            payload["full_output_path"] = full_output_path
        return payload


class LsTool(WorkspaceTool):
    name = "ls"
    description = "List files and directories under a path. Relative paths resolve from the configured root."

    path: str = Field(default=".", min_length=1)
    recursive: bool = False
    limit: int = Field(default=200, ge=1, le=5_000)

    def execute(self) -> dict[str, object]:
        try:
            path = self._resolve_path(self.path)
            if path.is_file():
                entries = [self._relative(path)]
            else:
                iterator = self._walk(path) if self.recursive else sorted(path.iterdir())
                entries = [self._relative(entry) for entry in iterator if entry != path][: self.limit]
            return self._success(path=self.path, entries=entries, recursive=self.recursive, truncated=len(entries) >= self.limit)
        except Exception as exc:
            return self._error(str(exc), path=self.path)


class FindTool(WorkspaceTool):
    name = "find"
    description = "Find files or directories whose displayed path matches a glob."

    path: str = Field(default=".", min_length=1)
    pattern: str = Field(min_length=1)
    limit: int = Field(default=100, ge=1, le=5_000)

    def execute(self) -> dict[str, object]:
        try:
            path = self._resolve_path(self.path)
            matches: list[str] = []
            for candidate in self._walk(path):
                relative = self._relative(candidate)
                if fnmatch(relative, self.pattern) or fnmatch(candidate.name, self.pattern):
                    matches.append(relative)
                if len(matches) >= self.limit:
                    break
            return self._success(path=self.path, pattern=self.pattern, matches=matches, truncated=len(matches) >= self.limit)
        except Exception as exc:
            return self._error(str(exc), path=self.path, pattern=self.pattern)


class GrepTool(WorkspaceTool):
    name = "grep"
    description = "Search text files with a regular expression. Relative paths resolve from the configured root."

    path: str = Field(default=".", min_length=1)
    pattern: str = Field(min_length=1)
    glob: str = Field(default=DEFAULT_GLOB, min_length=1)
    limit: int = Field(default=50, ge=1, le=5_000)

    def execute(self) -> dict[str, object]:
        try:
            root = self._resolve_path(self.path)
            regex = re.compile(self.pattern)
            matches: list[dict[str, object]] = []
            for candidate in self._iter_files(root, glob=self.glob):
                try:
                    text = candidate.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        rendered_line, line_truncated = truncate_line(line)
                        matches.append(
                            {
                                "path": self._relative(candidate),
                                "line": line_number,
                                "text": rendered_line,
                                "line_truncated": line_truncated,
                            }
                        )
                    if len(matches) >= self.limit:
                        return self._success(
                            path=self.path,
                            pattern=self.pattern,
                            glob=self.glob,
                            matches=matches,
                            truncated=True,
                        )
            return self._success(path=self.path, pattern=self.pattern, glob=self.glob, matches=matches, truncated=False)
        except Exception as exc:
            return self._error(str(exc), path=self.path, pattern=self.pattern, glob=self.glob)


def create_coding_tools(root: Path) -> list[BaseTool]:
    return [ReadTool.bind(root=root), BashTool.bind(root=root), EditTool.bind(root=root), WriteTool.bind(root=root)]


def create_read_only_tools(root: Path) -> list[BaseTool]:
    return [ReadTool.bind(root=root), GrepTool.bind(root=root), FindTool.bind(root=root), LsTool.bind(root=root)]


def create_all_tools(root: Path) -> list[BaseTool]:
    return [*create_coding_tools(root), GrepTool.bind(root=root), FindTool.bind(root=root), LsTool.bind(root=root)]


class ToolRegistry:
    def __init__(self, root: Path, tools: Sequence[BaseTool] | None = None) -> None:
        self.root = root.resolve()
        registered = list(tools) if tools is not None else create_coding_tools(self.root)
        if not registered:
            raise ValueError("Tool registry requires at least one tool")
        self._tools = {tool.name: tool for tool in registered}

    @classmethod
    def coding(cls, root: Path) -> "ToolRegistry":
        return cls(root=root, tools=create_coding_tools(root))

    @classmethod
    def read_only(cls, root: Path) -> "ToolRegistry":
        return cls(root=root, tools=create_read_only_tools(root))

    @classmethod
    def all(cls, root: Path) -> "ToolRegistry":
        return cls(root=root, tools=create_all_tools(root))

    def definitions(self) -> list[dict[str, object]]:
        return [tool.to_definition() for tool in self._tools.values()]

    def prepare(self, name: str, arguments: dict[str, object]) -> BaseTool:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        return tool.parse_arguments(arguments)

    def execute_name(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        try:
            return self.prepare(name, arguments).execute()
        except KeyError as exc:
            return {"ok": False, "error": str(exc)}
        except ValidationError as exc:
            return {"ok": False, "error": f"Invalid tool payload: {exc}"}

    def parse_arguments(self, tool_call: ToolCall) -> dict[str, object]:
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tool arguments: {exc}") from exc
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must decode to an object")
        return arguments

    def execute(self, tool_call: ToolCall) -> dict[str, object]:
        try:
            arguments = self.parse_arguments(tool_call)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return self.execute_name(tool_call.function.name, arguments)


def _coerce_subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
