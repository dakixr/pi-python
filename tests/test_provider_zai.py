from __future__ import annotations

import json

import httpx
import pytest

from pi.agent.models import Message
from pi.agent.providers.base import ProviderError, ProviderRateLimitError
from pi.agent.providers.zai import ZAIConfig, ZAIProvider


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
                                        "name": "read",
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
    assert message.tool_calls[0].function.arguments == '{"path":"README.md"}'


def test_zai_provider_retries_rate_limit_using_retry_after_header() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(
                429,
                headers={"Retry-After": "2"},
                json={"error": {"message": "Too many requests"}},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ZAIProvider(
        config=ZAIConfig(api_key="test-key", model="glm-5.1"),
        http_client=client,
        sleep=sleeps.append,
    )

    message = provider.complete(messages=[Message.user("hello")], tools=[])

    assert message.content == "done"
    assert attempts == 3
    assert sleeps == [2.0, 2.0]


def test_zai_provider_retries_transient_server_errors_with_backoff() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(
                503,
                json={"error": {"message": "Upstream unavailable"}},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ZAIProvider(
        config=ZAIConfig(api_key="test-key", model="glm-5.1"),
        http_client=client,
        sleep=sleeps.append,
    )

    message = provider.complete(messages=[Message.user("hello")], tools=[])

    assert message.content == "done"
    assert attempts == 3
    assert sleeps == [1.0, 2.0]


def test_zai_provider_retries_timeout_with_backoff() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ReadTimeout("The read operation timed out", request=request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ZAIProvider(
        config=ZAIConfig(api_key="test-key", model="glm-5.1"),
        http_client=client,
        sleep=sleeps.append,
    )

    message = provider.complete(messages=[Message.user("hello")], tools=[])

    assert message.content == "done"
    assert attempts == 3
    assert sleeps == [1.0, 2.0]


def test_zai_provider_raises_clean_rate_limit_error_after_retries() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            429,
            json={"error": {"message": "Rate limit exceeded"}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ZAIProvider(
        config=ZAIConfig(api_key="test-key", model="glm-5.1"),
        http_client=client,
        sleep=sleeps.append,
    )

    with pytest.raises(ProviderRateLimitError, match="Rate limit exceeded"):
        provider.complete(messages=[Message.user("hello")], tools=[])

    assert attempts == 3
    assert sleeps == [1.0, 2.0]


def test_zai_provider_raises_clean_timeout_error_after_retries() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("The read operation timed out", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ZAIProvider(
        config=ZAIConfig(api_key="test-key", model="glm-5.1"),
        http_client=client,
        sleep=sleeps.append,
    )

    with pytest.raises(ProviderError, match="timed out after 3 attempts"):
        provider.complete(messages=[Message.user("hello")], tools=[])

    assert attempts == 3
    assert sleeps == [1.0, 2.0]


def test_zai_provider_wraps_non_retryable_http_failures() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            400,
            json={"error": {"message": "Invalid request"}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ZAIProvider(
        config=ZAIConfig(api_key="test-key", model="glm-5.1"),
        http_client=client,
    )

    with pytest.raises(ProviderError, match="Invalid request"):
        provider.complete(messages=[Message.user("hello")], tools=[])

    assert attempts == 1
