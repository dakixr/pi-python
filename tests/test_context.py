from __future__ import annotations

from pi.agent.context import ContextManager
from pi.agent.models import Message, ToolCall, ToolFunction


def test_context_manager_compacts_older_messages_for_provider_only() -> None:
    manager = ContextManager(
        system_prompt="system prompt",
        max_provider_chars=4_500,
        keep_recent_chars=1_000,
    )
    messages = [
        Message.system("system prompt"),
        Message.user("first task"),
        Message.assistant("first answer " + ("alpha " * 4000)),
        Message.user("second task"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    function=ToolFunction(name="read", arguments='{"path":"README.md"}'),
                )
            ],
        ),
        Message.tool("call-1", '{"ok":true,"content":"' + ("beta " * 4000) + '"}'),
        Message.assistant("second answer"),
        Message.user("latest task"),
        Message.assistant("latest answer"),
    ]
    context = manager.initialize("follow-up", messages)
    provider_messages = manager.messages_for_provider(context)

    assert provider_messages[0].role == "system"
    assert provider_messages[0].content == "system prompt"
    assert provider_messages[1].role == "system"
    assert "Auto-compacted summary" in (provider_messages[1].content or "")
    assert "first task" in (provider_messages[1].content or "")
    assert [message.content for message in provider_messages[-4:]] == [
        "second answer",
        "latest task",
        "latest answer",
        "follow-up",
    ]
    assert context.messages[1].content == "first task"
    assert context.messages[-1].content == "follow-up"


def test_context_manager_compaction_preserves_tool_result_with_its_turn() -> None:
    manager = ContextManager(max_provider_chars=2_000, keep_recent_chars=500)
    context = manager.initialize(
        "new work",
        [
            Message.user("older"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=ToolFunction(name="bash", arguments='{"command":"echo hi"}'),
                    )
                ],
            ),
            Message.tool("call-1", '{"ok":true,"stdout":"hi"}'),
            Message.assistant("done"),
            Message.user("recent"),
            Message.assistant("keep this"),
        ],
    )

    provider_messages = manager.messages_for_provider(context)

    assert provider_messages[-3].role == "user"
    assert provider_messages[-2].role == "assistant"
    assert provider_messages[-1].role == "user"
    assert all(message.role != "tool" for message in provider_messages[-3:])
