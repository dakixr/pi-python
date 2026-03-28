from __future__ import annotations

from pathlib import Path

from pi_python.agent.loop import Agent
from pi_python.agent.models import Message, ToolCall, ToolFunction
from pi_python.agent.providers.base import Provider
from pi_python.agent.tools import ToolRegistry


class FakeProvider(Provider):
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> Message:
        self.calls += 1
        if self.calls == 1:
            assert tools
            return Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=ToolFunction(
                            name="write_file",
                            arguments='{"path":"hello.txt","content":"hello"}',
                        ),
                    )
                ],
            )

        assert messages[-1].role == "tool"
        assert "hello.txt" in (messages[-1].content or "")
        return Message.assistant("done")


def test_agent_loop_runs_until_final_answer(tmp_path: Path) -> None:
    agent = Agent(provider=FakeProvider(), tools=ToolRegistry(root=tmp_path))

    result = agent.run("Create a file")

    assert result.output == "done"
    assert result.iterations == 2
    assert (tmp_path / "hello.txt").read_text() == "hello"
    assert [message.role for message in result.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]

