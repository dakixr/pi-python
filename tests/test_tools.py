from __future__ import annotations

import json
from pathlib import Path

from pi_python.agent.models import ToolCall, ToolFunction
from pi_python.agent.tools import ToolRegistry


def build_tool_call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=f"call-{name}",
        function=ToolFunction(name=name, arguments=json.dumps(arguments)),
    )


def test_core_tools_round_trip(tmp_path: Path) -> None:
    tools = ToolRegistry(root=tmp_path)

    write_result = tools.execute(
        build_tool_call("write_file", {"path": "notes.txt", "content": "alpha"})
    )
    assert write_result["ok"] is True

    edit_result = tools.execute(
        build_tool_call(
            "edit_file",
            {"path": "notes.txt", "old_text": "alpha", "new_text": "beta"},
        )
    )
    assert edit_result["ok"] is True
    assert (tmp_path / "notes.txt").read_text() == "beta"

    read_result = tools.execute(build_tool_call("read_file", {"path": "notes.txt"}))
    assert read_result == {"ok": True, "content": "beta", "path": "notes.txt"}

    bash_result = tools.execute(build_tool_call("bash", {"command": "cat notes.txt"}))
    assert bash_result["ok"] is True
    assert bash_result["stdout"] == "beta"


def test_tools_reject_paths_outside_root(tmp_path: Path) -> None:
    tools = ToolRegistry(root=tmp_path)

    result = tools.execute(build_tool_call("read_file", {"path": "../escape.txt"}))

    assert result["ok"] is False
    assert "outside" in result["error"]
