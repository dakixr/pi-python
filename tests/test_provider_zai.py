from __future__ import annotations

import json

import httpx

from pi_python.agent.models import Message
from pi_python.agent.providers.zai import ZAIConfig, ZAIProvider


def test_zai_provider_posts_openai_style_chat_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
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
                                        "name": "read_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ZAIProvider(
        config=ZAIConfig(api_key="test-key", model="glm-5.1"),
        http_client=client,
    )

    message = provider.complete(
        messages=[Message.user("Summarize README.md")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert captured["url"] == "https://api.z.ai/api/paas/v4/chat/completions"
    assert captured["payload"] == {
        "model": "glm-5.1",
        "messages": [{"role": "user", "content": "Summarize README.md"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
    }
    assert message.tool_calls[0].function.name == "read_file"
    assert message.tool_calls[0].function.arguments == '{"path":"README.md"}'

