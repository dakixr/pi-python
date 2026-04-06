from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from pi.agent.models import ToolCall, ToolFunction
from pi.agent.tools import EditTool, ReadTool, ToolRegistry


def build_tool_call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=f"call-{name}",
        function=ToolFunction(name=name, arguments=json.dumps(arguments)),
    )


def test_core_tools_round_trip(tmp_path: Path) -> None:
    tools = ToolRegistry(root=tmp_path)

    assert [tool["function"]["name"] for tool in tools.definitions()] == [
        "read",
        "bash",
        "edit",
        "write",
    ]

    write_result = tools.execute(
        build_tool_call("write", {"path": "notes.txt", "content": "alpha"})
    )
    assert write_result["ok"] is True
    assert write_result["created"] is True
    assert write_result["diff"] == "--- /dev/null\n+++ b/notes.txt\n@@ -0,0 +1 @@\n+alpha"

    edit_result = tools.execute(
        build_tool_call(
            "edit",
            {"path": "notes.txt", "old_text": "alpha", "new_text": "beta"},
        )
    )
    assert edit_result["ok"] is True
    assert edit_result["diff"] == "--- a/notes.txt\n+++ b/notes.txt\n@@ -1 +1 @@\n-alpha\n+beta"
    assert (tmp_path / "notes.txt").read_text() == "beta"

    read_result = tools.execute(build_tool_call("read", {"path": "notes.txt"}))
    assert read_result == {"ok": True, "content": "beta", "path": "notes.txt"}

    bash_result = tools.execute(build_tool_call("bash", {"command": "cat notes.txt"}))
    assert bash_result["ok"] is True
    assert bash_result["stdout"] == "beta"


def test_tools_are_registered_from_pydantic_tool_models(tmp_path: Path) -> None:
    registry = ToolRegistry(
        root=tmp_path,
        tools=[ReadTool.bind(root=tmp_path), EditTool.bind(root=tmp_path)],
    )

    definitions = {tool["function"]["name"]: tool["function"] for tool in registry.definitions()}

    assert sorted(definitions) == ["edit", "read"]
    assert "offset" in definitions["read"]["parameters"]["properties"]
    assert "oldText" in definitions["edit"]["parameters"]["properties"]
    assert "old_text" not in definitions["edit"]["parameters"]["properties"]


def test_tools_reject_paths_outside_root(tmp_path: Path) -> None:
    tools = ToolRegistry(root=tmp_path)

    result = tools.execute(build_tool_call("read", {"path": "../escape.txt"}))

    assert result["ok"] is False
    assert "outside" in result["error"]


def test_tools_allow_absolute_paths_within_root(tmp_path: Path) -> None:
    tools = ToolRegistry(root=tmp_path)
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    result = tools.execute(build_tool_call("read", {"path": str(path)}))

    assert result["ok"] is True
    assert result["content"] == "hello"


def test_bash_reports_timeout(tmp_path: Path) -> None:
    tools = ToolRegistry(root=tmp_path)
    command = f'{shlex.quote(sys.executable)} -c "import time; time.sleep(2)"'

    result = tools.execute(
        build_tool_call(
            "bash",
            {
                "command": command,
                "timeout_seconds": 1,
            },
        )
    )

    assert result["ok"] is False
    assert result["command"] == command
    assert "timed out" in result["error"].lower()
