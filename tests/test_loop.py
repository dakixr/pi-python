from __future__ import annotations

import time
from pathlib import Path

from pi.agent.context import ContextManager
from pi.agent.loop import Agent
from pi.agent.models import Message, ToolCall, ToolFunction
from pi.agent.providers.base import Provider
from pi.agent.tools import ToolRegistry


class FakeProvider(Provider):
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> Message:
        self.calls += 1
        if self.calls == 1:
            return Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=ToolFunction(
                            name="write",
                            arguments='{"path":"hello.txt","content":"hello"}',
                        ),
                    )
                ],
            )
        assert messages[-1].role == "tool"
        return Message.assistant("done")


def test_agent_loop_runs_until_final_answer(tmp_path: Path) -> None:
    agent = Agent(provider=FakeProvider(), tools=ToolRegistry(root=tmp_path))

    result = agent.run("Create a file")

    assert result.output == "done"
    assert result.iterations == 2
    assert (tmp_path / "hello.txt").read_text() == "hello"
    assert [message.role for message in result.messages] == ["user", "assistant", "tool", "assistant"]


class HistoryProvider(Provider):
    def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> Message:
        assert [message.role for message in messages] == ["system", "user", "assistant", "user"]
        assert messages[0].content == "system prompt"
        assert messages[-2].content == "previous answer"
        return Message.assistant("continued")


def test_agent_loop_can_continue_existing_history(tmp_path: Path) -> None:
    agent = Agent(
        provider=HistoryProvider(),
        tools=ToolRegistry(root=tmp_path),
        system_prompt="system prompt",
    )

    result = agent.run(
        "next task",
        messages=[Message.user("previous task"), Message.assistant("previous answer")],
    )

    assert result.output == "continued"
    assert [message.role for message in result.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


class TransformProvider(Provider):
    def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> Message:
        assert [message.role for message in messages] == ["system", "user"]
        assert messages[0].content == "transformed system"
        return Message.assistant("done")


def test_agent_loop_uses_context_manager_before_provider_boundary(tmp_path: Path) -> None:
    def transform(messages: list[Message]) -> list[Message]:
        updated = [message.model_copy(deep=True) for message in messages]
        updated[0] = Message.system("transformed system")
        return updated

    agent = Agent(
        provider=TransformProvider(),
        tools=ToolRegistry(root=tmp_path),
        context_manager=ContextManager(
            system_prompt="original system",
            transform_messages=transform,
        ),
    )

    result = agent.run("hello")

    assert result.output == "done"


class ParallelProvider(Provider):
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> Message:
        self.calls += 1
        if self.calls == 1:
            return Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(id="call-1", function=ToolFunction(name="bash", arguments='{"command":"sleep 0.2 && echo first"}')),
                    ToolCall(id="call-2", function=ToolFunction(name="bash", arguments='{"command":"sleep 0.2 && echo second"}')),
                ],
            )
        return Message.assistant("done")


def test_agent_loop_supports_parallel_tool_execution_and_hooks(tmp_path: Path) -> None:
    before_seen: list[dict[str, object]] = []
    after_seen: list[dict[str, object]] = []
    events: list[str] = []
    agent = Agent(provider=ParallelProvider(), tools=ToolRegistry(root=tmp_path))

    start = time.perf_counter()
    result = agent.run(
        "run in parallel",
        on_event=lambda event, payload: events.append(event),
        before_tool_call=lambda ctx: before_seen.append(ctx.arguments) or None,
        after_tool_call=lambda ctx: after_seen.append(ctx.result) or None,
    )
    elapsed = time.perf_counter() - start

    assert result.output == "done"
    assert len(before_seen) == 2
    assert len(after_seen) == 2
    assert "model_start" in events
    assert "tool_execution_start" in events
    assert "tool_execution_end" in events
    assert elapsed < 0.38
