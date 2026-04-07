from __future__ import annotations

import json

import httpx

from pi.agent.models import Message
from pi.agent.providers.zai import ZAIConfig, ZAIProvider


def test_zai_provider_posts_openai_style_chat_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "read",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ZAIProvider(config=ZAIConfig(api_key="test-key", model="glm-5.1"), http_client=client)

    message = provider.complete(
        messages=[Message.user("Summarize README.md")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert captured["url"] == "https://api.z.ai/api/coding/paas/v4/chat/completions"
    assert captured["payload"] == {
        "model": "glm-5.1",
        "messages": [{"role": "user", "content": "Summarize README.md"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
    }
    assert message.tool_calls[0].function.name == "read"


def test_zai_provider_normalizes_messages_before_posting() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "done"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ZAIProvider(config=ZAIConfig(api_key="test-key", model="glm-5.1"), http_client=client)

    provider.complete(
        messages=[
            Message.system("system prompt"),
            Message.system("compacted summary"),
            Message(role="assistant", content=None, tool_calls=[]),
            Message.user("hello"),
        ],
        tools=[],
    )

    assert captured["payload"] == {
        "model": "glm-5.1",
        "messages": [
            {"role": "system", "content": "system prompt\n\ncompacted summary"},
            {"role": "user", "content": "hello"},
        ],
    }


def test_zai_provider_drops_orphaned_tool_call_exchanges() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "done"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ZAIProvider(config=ZAIConfig(api_key="test-key", model="glm-5.1"), http_client=client)

    provider.complete(
        messages=[
            Message.user("older"),
            Message(
                role="assistant",
                content="tool planning",
                tool_calls=[
                    {
                        "id": "call-1",
                        "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                    }
                ],
            ),
            Message.user("newer"),
        ],
        tools=[],
    )

    assert captured["payload"] == {
        "model": "glm-5.1",
        "messages": [
            {"role": "user", "content": "older"},
            {"role": "assistant", "content": "tool planning"},
            {"role": "user", "content": "newer"},
        ],
    }


def test_zai_provider_flattens_historical_tool_exchanges_before_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "done"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ZAIProvider(config=ZAIConfig(api_key="test-key", model="glm-5.1"), http_client=client)

    provider.complete(
        messages=[
            Message.system("system prompt"),
            Message.user("first"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call-1",
                        "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                    }
                ],
            ),
            Message.tool("call-1", '{"ok": true, "content": "hello"}'),
            Message.assistant("intermediate"),
            Message.user("second"),
        ],
        tools=[],
    )

    assert captured["payload"] == {
        "model": "glm-5.1",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": '[Assistant tool calls] read({"path":"README.md"})\n[Tool result] {"ok": true, "content": "hello"}\n\nintermediate',
            },
            {"role": "user", "content": "second"},
        ],
    }
