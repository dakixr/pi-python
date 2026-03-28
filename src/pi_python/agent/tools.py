from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from pi_python.agent.models import ToolCall

MAX_TEXT_FILE_BYTES = 1_000_000
MAX_BASH_OUTPUT_CHARS = 12_000


class ReadArgs(BaseModel):
    path: str = Field(min_length=1)


class WriteArgs(BaseModel):
    path: str = Field(min_length=1)
    content: str


class EditArgs(BaseModel):
    path: str = Field(min_length=1)
    old_text: str = Field(min_length=1)
    new_text: str


class BashArgs(BaseModel):
    command: str = Field(min_length=1, max_length=4000)
    timeout_seconds: int = Field(default=30, ge=1, le=120)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    arguments_model: type[BaseModel]

    def to_definition(self) -> dict[str, object]:
        schema = self.arguments_model.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class ToolRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if not self.root.exists() or not self.root.is_dir():
            raise ValueError(f"Workspace root does not exist or is not a directory: {root}")
        self._tools: dict[str, ToolSpec] = {
            "read": ToolSpec(
                name="read",
                description="Read a UTF-8 text file relative to the workspace root.",
                arguments_model=ReadArgs,
            ),
            "write": ToolSpec(
                name="write",
                description="Write a UTF-8 text file relative to the workspace root.",
                arguments_model=WriteArgs,
            ),
            "edit": ToolSpec(
                name="edit",
                description="Replace one text fragment in a UTF-8 text file.",
                arguments_model=EditArgs,
            ),
            "bash": ToolSpec(
                name="bash",
                description="Run a shell command in the workspace root with a timeout.",
                arguments_model=BashArgs,
            ),
        }

    def definitions(self) -> list[dict[str, object]]:
        return [tool.to_definition() for tool in self._tools.values()]

    def execute(self, tool_call: ToolCall) -> dict[str, object]:
        try:
            tool_spec = self._tools[tool_call.function.name]
        except KeyError:
            return {"ok": False, "error": f"Unknown tool: {tool_call.function.name}"}

        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"Invalid tool arguments: {exc}"}

        try:
            parsed = tool_spec.arguments_model.model_validate(arguments)
        except ValidationError as exc:
            return {"ok": False, "error": f"Invalid tool payload: {exc}"}
        handler = getattr(self, f"_handle_{tool_call.function.name}")
        return handler(parsed)

    def _resolve_path(self, relative_path: str) -> Path:
        if not relative_path.strip():
            raise ValueError("Path must be a non-empty relative path")

        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError("Path must be relative to the workspace root")

        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"Path {relative_path!r} is outside the workspace root")
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

    def _handle_read(self, arguments: ReadArgs) -> dict[str, object]:
        try:
            path = self._resolve_path(arguments.path)
            self._ensure_text_file(path)
            return {
                "ok": True,
                "path": arguments.path,
                "content": path.read_text(encoding="utf-8"),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _handle_write(self, arguments: WriteArgs) -> dict[str, object]:
        try:
            path = self._resolve_path(arguments.path)
            encoded = arguments.content.encode("utf-8")
            if len(encoded) > MAX_TEXT_FILE_BYTES:
                raise ValueError(
                    f"Content is too large to write safely (> {MAX_TEXT_FILE_BYTES} bytes)"
                )
            if path.exists() and not path.is_file():
                raise ValueError(f"Path is not a regular file: {arguments.path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(arguments.content, encoding="utf-8")
            return {
                "ok": True,
                "path": arguments.path,
                "bytes_written": len(encoded),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _handle_edit(self, arguments: EditArgs) -> dict[str, object]:
        try:
            path = self._resolve_path(arguments.path)
            self._ensure_text_file(path)
            content = path.read_text(encoding="utf-8")
            if arguments.old_text not in content:
                return {
                    "ok": False,
                    "error": f"Text not found in {arguments.path}",
                }
            updated = content.replace(arguments.old_text, arguments.new_text, 1)
            path.write_text(updated, encoding="utf-8")
            return {"ok": True, "path": arguments.path}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _handle_bash(self, arguments: BashArgs) -> dict[str, object]:
        try:
            stdout: str
            stderr: str
            completed = subprocess.run(
                ["bash", "-c", arguments.command],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=arguments.timeout_seconds,
                check=False,
                env={"PATH": os.environ.get("PATH", "")},
            )
            stdout, stdout_truncated = self._trim_output(completed.stdout)
            stderr, stderr_truncated = self._trim_output(completed.stderr)
            return {
                "ok": completed.returncode == 0,
                "command": arguments.command,
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }
        except subprocess.TimeoutExpired as exc:
            stdout, stdout_truncated = self._trim_output(exc.stdout or "")
            stderr, stderr_truncated = self._trim_output(exc.stderr or "")
            return {
                "ok": False,
                "command": arguments.command,
                "error": f"Command timed out after {arguments.timeout_seconds} seconds",
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }
        except Exception as exc:
            return {"ok": False, "command": arguments.command, "error": str(exc)}
