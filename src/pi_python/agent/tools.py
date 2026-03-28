from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from pi_python.agent.models import ToolCall


class ReadFileArgs(BaseModel):
    path: str


class WriteFileArgs(BaseModel):
    path: str
    content: str


class EditFileArgs(BaseModel):
    path: str
    old_text: str
    new_text: str


class BashArgs(BaseModel):
    command: str
    timeout_seconds: int = Field(default=30, ge=1, le=300)


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
        self._tools: dict[str, ToolSpec] = {
            "read_file": ToolSpec(
                name="read_file",
                description="Read a UTF-8 text file relative to the workspace root.",
                arguments_model=ReadFileArgs,
            ),
            "write_file": ToolSpec(
                name="write_file",
                description="Write a UTF-8 text file relative to the workspace root.",
                arguments_model=WriteFileArgs,
            ),
            "edit_file": ToolSpec(
                name="edit_file",
                description="Replace one text fragment in a UTF-8 text file.",
                arguments_model=EditFileArgs,
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
        candidate = (self.root / relative_path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"Path {relative_path!r} is outside the workspace root")
        return candidate

    def _handle_read_file(self, arguments: ReadFileArgs) -> dict[str, object]:
        try:
            path = self._resolve_path(arguments.path)
            return {
                "ok": True,
                "path": arguments.path,
                "content": path.read_text(encoding="utf-8"),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _handle_write_file(self, arguments: WriteFileArgs) -> dict[str, object]:
        try:
            path = self._resolve_path(arguments.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(arguments.content, encoding="utf-8")
            return {
                "ok": True,
                "path": arguments.path,
                "bytes_written": len(arguments.content.encode("utf-8")),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _handle_edit_file(self, arguments: EditFileArgs) -> dict[str, object]:
        try:
            path = self._resolve_path(arguments.path)
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
            completed = subprocess.run(
                arguments.command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=arguments.timeout_seconds,
                check=False,
            )
            return {
                "ok": completed.returncode == 0,
                "command": arguments.command,
                "returncode": completed.returncode,
                "stdout": completed.stdout.rstrip("\n"),
                "stderr": completed.stderr.rstrip("\n"),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
