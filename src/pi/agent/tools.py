from __future__ import annotations

import json
import os
import re
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, ClassVar

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from pi.agent.models import ToolCall

MAX_TEXT_FILE_BYTES = 1_000_000
MAX_BASH_OUTPUT_CHARS = 12_000
DEFAULT_GLOB = "*"


class ReadToolInput(BaseModel):
    path: str = Field(min_length=1)
    offset: int | None = Field(default=None, ge=1)
    limit: int | None = Field(default=None, ge=1)


class WriteToolInput(BaseModel):
    path: str = Field(min_length=1)
    content: str


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


class EditToolInput(BaseModel):
    path: str = Field(min_length=1)
    old_text: str | None = Field(
        default=None,
        validation_alias=AliasChoices("oldText", "old_text"),
        serialization_alias="oldText",
    )
    new_text: str | None = Field(
        default=None,
        validation_alias=AliasChoices("newText", "new_text"),
        serialization_alias="newText",
    )
    edits: list[ReplaceEdit] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_edit_mode(self) -> "EditToolInput":
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
        return [ReplaceEdit(oldText=self.old_text, newText=self.new_text)]


class BashToolInput(BaseModel):
    command: str = Field(min_length=1, max_length=4000)
    timeout: int = Field(
        default=30,
        ge=1,
        le=120,
        validation_alias=AliasChoices("timeout", "timeout_seconds"),
    )


class LsToolInput(BaseModel):
    path: str = Field(default=".", min_length=1)
    recursive: bool = False
    limit: int = Field(default=200, ge=1, le=1_000)


class FindToolInput(BaseModel):
    path: str = Field(default=".", min_length=1)
    pattern: str = Field(min_length=1)
    limit: int = Field(default=100, ge=1, le=1_000)


class GrepToolInput(BaseModel):
    path: str = Field(default=".", min_length=1)
    pattern: str = Field(min_length=1)
    glob: str = Field(default=DEFAULT_GLOB, min_length=1)
    limit: int = Field(default=50, ge=1, le=500)


class BaseTool(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    arguments_model: ClassVar[type[BaseModel]]

    def to_definition(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments_model.model_json_schema(by_alias=True),
            },
        }

    def parse_arguments(self, arguments: dict[str, object]) -> BaseModel:
        return self.arguments_model.model_validate(arguments)

    @abstractmethod
    def execute(self, arguments: BaseModel) -> dict[str, object]:
        raise NotImplementedError


class WorkspaceTool(BaseTool, ABC):
    root: Path

    @field_validator("root")
    @classmethod
    def validate_root(cls, root: Path) -> Path:
        resolved = root.resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise ValueError(f"Workspace root does not exist or is not a directory: {root}")
        return resolved

    def _resolve_path(self, relative_path: str, *, allow_missing: bool = False) -> Path:
        if not relative_path.strip():
            raise ValueError("Path must be a non-empty relative path")

        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError("Path must be relative to the workspace root")

        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"Path {relative_path!r} is outside the workspace root")
        if not allow_missing and not candidate.exists():
            raise FileNotFoundError(relative_path)
        return candidate

    def _ensure_text_file(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path.relative_to(self.root)}")
        if path.stat().st_size > MAX_TEXT_FILE_BYTES:
            raise ValueError(
                f"File is too large to process safely (> {MAX_TEXT_FILE_BYTES} bytes)"
            )

    def _trim_output(self, value: str) -> tuple[str, bool]:
        trimmed = value.rstrip("\n")
        if len(trimmed) <= MAX_BASH_OUTPUT_CHARS:
            return trimmed, False
        return trimmed[:MAX_BASH_OUTPUT_CHARS], True

    def _success(self, **payload: object) -> dict[str, object]:
        return {"ok": True, **payload}

    def _error(self, error: str, **payload: object) -> dict[str, object]:
        return {"ok": False, "error": error, **payload}

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.root))


class ReadTool(WorkspaceTool):
    arguments_model: ClassVar[type[BaseModel]] = ReadToolInput
    name: str = "read"
    description: str = (
        "Read a UTF-8 text file relative to the workspace root. "
        "Supports offset/limit for large files."
    )

    def execute(self, arguments: BaseModel) -> dict[str, object]:
        args = arguments if isinstance(arguments, ReadToolInput) else ReadToolInput.model_validate(arguments)
        try:
            path = self._resolve_path(args.path)
            self._ensure_text_file(path)
            content = path.read_text(encoding="utf-8")

            if args.offset is None and args.limit is None:
                return self._success(path=args.path, content=content)

            lines = content.splitlines()
            if content.endswith("\n"):
                lines.append("")
            start = (args.offset or 1) - 1
            if start >= len(lines):
                raise ValueError(f"Offset {args.offset} is beyond end of file")
            end = start + args.limit if args.limit is not None else len(lines)
            selected = lines[start:end]
            return self._success(
                path=args.path,
                content="\n".join(selected),
                offset=args.offset or 1,
                limit=args.limit,
                next_offset=end + 1 if end < len(lines) else None,
            )
        except Exception as exc:
            return self._error(str(exc), path=args.path)


class WriteTool(WorkspaceTool):
    arguments_model: ClassVar[type[BaseModel]] = WriteToolInput
    name: str = "write"
    description: str = "Write a UTF-8 text file relative to the workspace root."

    def execute(self, arguments: BaseModel) -> dict[str, object]:
        args = (
            arguments if isinstance(arguments, WriteToolInput) else WriteToolInput.model_validate(arguments)
        )
        try:
            path = self._resolve_path(args.path, allow_missing=True)
            encoded = args.content.encode("utf-8")
            if len(encoded) > MAX_TEXT_FILE_BYTES:
                raise ValueError(
                    f"Content is too large to write safely (> {MAX_TEXT_FILE_BYTES} bytes)"
                )
            if path.exists() and not path.is_file():
                raise ValueError(f"Path is not a regular file: {args.path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args.content, encoding="utf-8")
            return self._success(path=args.path, bytes_written=len(encoded))
        except Exception as exc:
            return self._error(str(exc), path=args.path)


class EditTool(WorkspaceTool):
    arguments_model: ClassVar[type[BaseModel]] = EditToolInput
    name: str = "edit"
    description: str = (
        "Replace exact text in a UTF-8 file. "
        "Use oldText/newText for one change or edits for multiple disjoint changes."
    )

    def execute(self, arguments: BaseModel) -> dict[str, object]:
        args = arguments if isinstance(arguments, EditToolInput) else EditToolInput.model_validate(arguments)
        try:
            path = self._resolve_path(args.path)
            self._ensure_text_file(path)
            original = path.read_text(encoding="utf-8")

            resolved_edits: list[tuple[int, int, str, str]] = []
            for edit in args.normalized_edits():
                start = original.find(edit.old_text)
                if start == -1:
                    return self._error(f"Text not found in {args.path}", path=args.path)
                second_match = original.find(edit.old_text, start + 1)
                if second_match != -1:
                    return self._error(
                        f"Text to replace is not unique in {args.path}",
                        path=args.path,
                    )
                resolved_edits.append(
                    (start, start + len(edit.old_text), edit.old_text, edit.new_text)
                )

            resolved_edits.sort(key=lambda item: item[0])
            for current, following in zip(resolved_edits, resolved_edits[1:]):
                if current[1] > following[0]:
                    return self._error(
                        f"Edits overlap in {args.path}. Merge nearby changes into one edit.",
                        path=args.path,
                    )

            updated = original
            for start, end, old_text, new_text in reversed(resolved_edits):
                if updated[start:end] != old_text:
                    return self._error(
                        f"Edit no longer matches file content in {args.path}",
                        path=args.path,
                    )
                updated = updated[:start] + new_text + updated[end:]

            path.write_text(updated, encoding="utf-8")
            return self._success(path=args.path, edits_applied=len(resolved_edits))
        except Exception as exc:
            return self._error(str(exc), path=args.path)


class BashTool(WorkspaceTool):
    arguments_model: ClassVar[type[BaseModel]] = BashToolInput
    name: str = "bash"
    description: str = "Run a shell command in the workspace root with a timeout."

    def execute(self, arguments: BaseModel) -> dict[str, object]:
        args = arguments if isinstance(arguments, BashToolInput) else BashToolInput.model_validate(arguments)
        try:
            completed = subprocess.run(
                ["bash", "-c", args.command],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
                env={"PATH": os.environ.get("PATH", "")},
            )
            stdout, stdout_truncated = self._trim_output(completed.stdout)
            stderr, stderr_truncated = self._trim_output(completed.stderr)
            return {
                "ok": completed.returncode == 0,
                "command": args.command,
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "timeout": args.timeout,
            }
        except subprocess.TimeoutExpired as exc:
            stdout, stdout_truncated = self._trim_output(exc.stdout or "")
            stderr, stderr_truncated = self._trim_output(exc.stderr or "")
            return self._error(
                f"Command timed out after {args.timeout} seconds",
                command=args.command,
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                timeout=args.timeout,
            )
        except Exception as exc:
            return self._error(str(exc), command=args.command, timeout=args.timeout)


class LsTool(WorkspaceTool):
    arguments_model: ClassVar[type[BaseModel]] = LsToolInput
    name: str = "ls"
    description: str = "List files and directories under a workspace-relative path."

    def execute(self, arguments: BaseModel) -> dict[str, object]:
        args = arguments if isinstance(arguments, LsToolInput) else LsToolInput.model_validate(arguments)
        try:
            path = self._resolve_path(args.path)
            if path.is_file():
                entries = [self._relative(path)]
            else:
                iterator = path.rglob("*") if args.recursive else path.iterdir()
                entries = sorted(self._relative(entry) for entry in iterator)[: args.limit]
            return self._success(path=args.path, entries=entries, recursive=args.recursive)
        except Exception as exc:
            return self._error(str(exc), path=args.path)


class FindTool(WorkspaceTool):
    arguments_model: ClassVar[type[BaseModel]] = FindToolInput
    name: str = "find"
    description: str = "Find workspace files or directories whose relative path matches a glob."

    def execute(self, arguments: BaseModel) -> dict[str, object]:
        args = arguments if isinstance(arguments, FindToolInput) else FindToolInput.model_validate(arguments)
        try:
            path = self._resolve_path(args.path)
            if path.is_file():
                candidates = [path]
            else:
                candidates = [path, *path.rglob("*")]

            matches: list[str] = []
            for candidate in candidates:
                relative = self._relative(candidate)
                if fnmatch(relative, args.pattern) or fnmatch(candidate.name, args.pattern):
                    matches.append(relative)
                if len(matches) >= args.limit:
                    break
            return self._success(path=args.path, pattern=args.pattern, matches=matches)
        except Exception as exc:
            return self._error(str(exc), path=args.path, pattern=args.pattern)


class GrepTool(WorkspaceTool):
    arguments_model: ClassVar[type[BaseModel]] = GrepToolInput
    name: str = "grep"
    description: str = "Search workspace text files with a regular expression."

    def execute(self, arguments: BaseModel) -> dict[str, object]:
        args = arguments if isinstance(arguments, GrepToolInput) else GrepToolInput.model_validate(arguments)
        try:
            root = self._resolve_path(args.path)
            regex = re.compile(args.pattern)
            matches: list[dict[str, object]] = []

            candidates: Iterable[Path]
            if root.is_file():
                candidates = [root]
            else:
                candidates = (
                    candidate
                    for candidate in root.rglob("*")
                    if candidate.is_file() and fnmatch(candidate.name, args.glob)
                )

            for candidate in candidates:
                if candidate.stat().st_size > MAX_TEXT_FILE_BYTES:
                    continue
                try:
                    text = candidate.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        matches.append(
                            {
                                "path": self._relative(candidate),
                                "line": line_number,
                                "text": line,
                            }
                        )
                    if len(matches) >= args.limit:
                        return self._success(
                            path=args.path,
                            pattern=args.pattern,
                            glob=args.glob,
                            matches=matches,
                        )

            return self._success(
                path=args.path,
                pattern=args.pattern,
                glob=args.glob,
                matches=matches,
            )
        except Exception as exc:
            return self._error(str(exc), path=args.path, pattern=args.pattern, glob=args.glob)


def create_coding_tools(root: Path) -> list[BaseTool]:
    return [
        ReadTool(root=root),
        BashTool(root=root),
        EditTool(root=root),
        WriteTool(root=root),
    ]


def create_read_only_tools(root: Path) -> list[BaseTool]:
    return [
        ReadTool(root=root),
        GrepTool(root=root),
        FindTool(root=root),
        LsTool(root=root),
    ]


def create_all_tools(root: Path) -> list[BaseTool]:
    return [
        ReadTool(root=root),
        BashTool(root=root),
        EditTool(root=root),
        WriteTool(root=root),
        GrepTool(root=root),
        FindTool(root=root),
        LsTool(root=root),
    ]


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

    def execute(self, tool_call: ToolCall) -> dict[str, object]:
        tool = self._tools.get(tool_call.function.name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool: {tool_call.function.name}"}

        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"Invalid tool arguments: {exc}"}

        try:
            parsed = tool.parse_arguments(arguments)
        except ValidationError as exc:
            return {"ok": False, "error": f"Invalid tool payload: {exc}"}

        return tool.execute(parsed)
