from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys

from pi.agent.models import ToolCall, ToolFunction
from pi.agent.tools import EditTool, ReadTool, ToolRegistry


def build_tool_call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=f"call-{name}",
        function=ToolFunction(name=name, arguments=json.dumps(arguments)),
    )


def test_core_tools_round_trip(tmp_path: Path) -> None:
    tools = ToolRegistry(root=tmp_path)

    assert [tool["function"]["name"] for tool in tools.definitions()] == ["read", "bash", "edit", "write"]

    write_result = tools.execute(build_tool_call("write", {"path": "notes.txt", "content": "alpha"}))
    assert write_result["ok"] is True
    assert write_result["created"] is True

    edit_result = tools.execute(
        build_tool_call("edit", {"path": "notes.txt", "old_text": "alpha", "new_text": "beta"})
    )
    assert edit_result["ok"] is True
    assert (tmp_path / "notes.txt").read_text() == "beta"

    read_result = tools.execute(build_tool_call("read", {"path": "notes.txt"}))
    assert read_result == {
        "ok": True,
        "content": "beta",
        "path": "notes.txt",
        "offset": 1,
        "limit": None,
        "next_offset": None,
        "details": None,
    }

    bash_result = tools.execute(build_tool_call("bash", {"command": "cat notes.txt"}))
    assert bash_result["ok"] is True
    assert bash_result["stdout"] == "beta"
    assert bash_result["output"] == "beta"


def test_tools_are_registered_from_pydantic_tool_models(tmp_path: Path) -> None:
    registry = ToolRegistry(root=tmp_path, tools=[ReadTool.bind(root=tmp_path), EditTool.bind(root=tmp_path)])

    definitions = {tool["function"]["name"]: tool["function"] for tool in registry.definitions()}

    assert sorted(definitions) == ["edit", "read"]
    assert "offset" in definitions["read"]["parameters"]["properties"]
    assert "oldText" in definitions["edit"]["parameters"]["properties"]
    assert "old_text" not in definitions["edit"]["parameters"]["properties"]


def test_tools_allow_paths_outside_root(tmp_path: Path) -> None:
    tools = ToolRegistry(root=tmp_path)
    outside = tmp_path.parent / "escape.txt"
    outside.write_text("outside", encoding="utf-8")

    result = tools.execute(build_tool_call("read", {"path": "../escape.txt"}))

    assert result["ok"] is True
    assert result["content"] == "outside"


def test_write_and_edit_use_absolute_display_path_outside_root(tmp_path: Path) -> None:
    tools = ToolRegistry(root=tmp_path)
    outside = tmp_path.parent / "outside-notes.txt"

    write_result = tools.execute(
        build_tool_call("write", {"path": str(outside), "content": "alpha"})
    )
    assert write_result["ok"] is True
    assert write_result["path"] == str(outside)
    assert "--- /dev/null" in (write_result["diff"] or "")
    assert f"+++ b/{outside}" in (write_result["diff"] or "")

    edit_result = tools.execute(
        build_tool_call(
            "edit",
            {"path": str(outside), "old_text": "alpha", "new_text": "beta"},
        )
    )
    assert edit_result["ok"] is True
    assert f"--- a/{outside}" in (edit_result["diff"] or "")
    assert f"+++ b/{outside}" in (edit_result["diff"] or "")
    assert outside.read_text(encoding="utf-8") == "beta"


def test_bash_reports_timeout(tmp_path: Path) -> None:
    tools = ToolRegistry(root=tmp_path)
    command = f'{shlex.quote(sys.executable)} -c "import time; time.sleep(2)"'

    result = tools.execute(build_tool_call("bash", {"command": command, "timeout_seconds": 1}))

    assert result["ok"] is False
    assert result["command"] == command
    assert "timed out" in result["error"].lower()


def test_read_reports_truncation_details_and_next_offset(tmp_path: Path) -> None:
    tools = ToolRegistry(root=tmp_path)
    lines = "\n".join(f"line {index}" for index in range(2505))
    (tmp_path / "large.txt").write_text(lines, encoding="utf-8")

    result = tools.execute(build_tool_call("read", {"path": "large.txt"}))

    assert result["ok"] is True
    assert result["next_offset"] == 2001
    assert "Use offset=2001 to continue." in result["content"]
    assert result["details"]["truncation"]["truncatedBy"] == "lines"
    assert result["details"]["truncation"]["outputLines"] == 2000


def test_bash_saves_full_output_when_truncated(tmp_path: Path) -> None:
    tools = ToolRegistry(root=tmp_path)
    command = f"{shlex.quote(sys.executable)} -c \"print('x' * 70000)\""

    result = tools.execute(build_tool_call("bash", {"command": command}))

    assert result["ok"] is True
    assert result["details"]["truncation"]["truncated"] is True
    assert Path(result["full_output_path"]).exists()
    assert len(Path(result["full_output_path"]).read_text(encoding="utf-8")) > len(result["output"])


def test_grep_truncates_long_matched_lines(tmp_path: Path) -> None:
    tools = ToolRegistry.read_only(tmp_path)
    long_line = "needle " + ("x" * 800)
    (tmp_path / "search.txt").write_text(long_line, encoding="utf-8")

    result = tools.execute(build_tool_call("grep", {"path": ".", "pattern": "needle"}))

    assert result["ok"] is True
    assert result["matches"][0]["line_truncated"] is True
    assert result["matches"][0]["text"].endswith("[truncated]")
