from __future__ import annotations

import time
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from pi.agent.models import Message, ToolCall
from pi.agent.providers.base import Provider, ProviderError, ProviderRateLimitError, ProviderServerError


class OpenAICompatibleConfig(BaseModel):
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    chat_completions_path: str = "/chat/completions"
    timeout_seconds: float | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    max_retry_backoff_seconds: float = 8.0


class OpenAICompatibleResponseMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    def to_message(self) -> Message:
        return Message(role=self.role, content=self.content, tool_calls=self.tool_calls)


class OpenAICompatibleChoice(BaseModel):
    message: OpenAICompatibleResponseMessage


class OpenAICompatibleChatCompletionResponse(BaseModel):
    choices: list[OpenAICompatibleChoice]


class OpenAICompatibleProvider(Provider):
    def __init__(
        self,
        config: OpenAICompatibleConfig,
        http_client: httpx.Client | None = None,
        sleep: Any | None = None,
    ) -> None:
        self.config = config
        self._owns_client = http_client is None
        self._sleep = sleep or time.sleep
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            **config.headers,
        }
        self._client = http_client or httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
            headers=headers,
        )

    def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> Message:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_api_dict() for message in messages],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error: ProviderError | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.post(self._chat_completions_url(), json=payload)
            except httpx.TimeoutException as exc:
                last_error = ProviderError("Provider request timed out.")
                if attempt < self.config.max_retries:
                    self._sleep(self._backoff_seconds(attempt))
                    continue
                raise last_error from exc
            except httpx.RequestError as exc:
                raise ProviderError(f"Provider request failed: {exc}") from exc

            if response.status_code == 429:
                retry_after = _retry_after_seconds(response)
                last_error = ProviderRateLimitError(
                    f"Provider request was rate limited: {_error_detail(response)}",
                    retry_after_seconds=retry_after,
                )
                if attempt < self.config.max_retries:
                    self._sleep(retry_after or self._backoff_seconds(attempt))
                    continue
                raise last_error

            if 500 <= response.status_code < 600:
                last_error = ProviderServerError(
                    f"Provider server error: {_error_detail(response)}",
                    status_code=response.status_code,
                )
                if attempt < self.config.max_retries:
                    self._sleep(self._backoff_seconds(attempt))
                    continue
                raise last_error

            if response.is_error:
                raise ProviderError(
                    f"Provider request failed: {_error_detail(response)}",
                    status_code=response.status_code,
                )

            try:
                completion = OpenAICompatibleChatCompletionResponse.model_validate(response.json())
            except (ValueError, ValidationError) as exc:
                raise ProviderError("Provider returned an invalid response payload.") from exc

            if not completion.choices:
                raise ProviderError("Provider returned no completion choices.")
            return completion.choices[0].message.to_message()

        if last_error is not None:
            raise last_error
        raise ProviderError("Provider request failed unexpectedly.")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _backoff_seconds(self, attempt: int) -> float:
        return min(self.config.retry_backoff_seconds * (2**attempt), self.config.max_retry_backoff_seconds)

    def _chat_completions_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/{self.config.chat_completions_path.lstrip('/')}"


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, dict):
                nested = value.get("message") or value.get("detail")
                if isinstance(nested, str) and nested.strip():
                    return nested
    return response.text.strip() or f"HTTP {response.status_code}"


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None
