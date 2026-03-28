from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field

from pi_python.agent.models import Message, ToolCall
from pi_python.agent.providers.base import Provider


class ZAIConfig(BaseModel):
    api_key: str
    model: str = "glm-5.1"
    base_url: str = "https://api.z.ai/api/paas/v4"
    timeout_seconds: float = 60.0


class ZAIResponseMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    def to_message(self) -> Message:
        return Message(role=self.role, content=self.content, tool_calls=self.tool_calls)


class ZAIChoice(BaseModel):
    message: ZAIResponseMessage


class ZAIChatCompletionResponse(BaseModel):
    choices: list[ZAIChoice]


class ZAIProvider(Provider):
    def __init__(self, config: ZAIConfig, http_client: httpx.Client | None = None) -> None:
        self.config = config
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )

    def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> Message:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_api_dict() for message in messages],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = self._client.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        completion = ZAIChatCompletionResponse.model_validate(response.json())
        return completion.choices[0].message.to_message()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
