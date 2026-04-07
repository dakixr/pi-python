from __future__ import annotations

import json
from pathlib import Path

import httpx

from pi.agent.models import Message, ToolCall, ToolFunction
from pi.agent.providers.base import Provider
from pi.ai import Context, OpenAICompatibleConfig, OpenAICompatibleProvider, complete, create_agent, run_task, stream


class StaticProvider(Provider):
    def __init__(self, message: Message) -> None:
        self.message = message
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> Message:
        self.calls.append(messages)
        return self.message


def test_ai_complete_accepts_context() -> None:
    provider = StaticProvider(Message.assistant("done"))

    result = complete(provider=provider, context=Context.from_prompt("hello", system_prompt="sys"))

    assert result.output == "done"
    assert [message.role for message in result.messages] == ["system", "user", "assistant"]


def test_ai_stream_emits_basic_message_events() -> None:
    provider = StaticProvider(Message.assistant("streamed"))

    events = list(stream(provider=provider, prompt="hello"))

    assert [event.type for event in events] == ["message_start", "message_delta", "message_end"]
    assert events[1].delta == "streamed"


def test_ai_create_agent_and_run_task(tmp_path: Path) -> None:
    provider = StaticProvider(
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call-1",
                    function=ToolFunction(name="write", arguments='{"path":"hello.txt","content":"hello"}'),
                )
            ],
        )
    )
    final_provider = StaticProvider(Message.assistant("done"))

    class TwoStepProvider(Provider):
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> Message:
            self.calls += 1
            return provider.message if self.calls == 1 else final_provider.message

    agent = create_agent(provider=TwoStepProvider(), root=tmp_path)
    result = agent.run("create file")

    assert result.output == "done"
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello"

    second = run_task("create file", provider=TwoStepProvider(), root=tmp_path)
    assert second.output == "done"


def test_openai_compatible_provider_posts_chat_completions_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "done"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        config=OpenAICompatibleConfig(api_key="test-key", model="gpt-test", base_url="https://example.com/v1"),
        http_client=client,
    )

    message = provider.complete([Message.user("hello")], [])

    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["payload"] == {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]}
    assert message.content == "done"
