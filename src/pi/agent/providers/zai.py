from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from pi.agent.models import Message, ToolCall
from pi.agent.providers.base import (
    Provider,
    ProviderError,
    ProviderRateLimitError,
    ProviderServerError,
)


class ZAIConfig(BaseModel):
    api_key: str
    model: str = "glm-5.1"
    base_url: str = "https://api.z.ai/api/coding/paas/v4"
    timeout_seconds: float | None = None
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    max_retry_backoff_seconds: float = 8.0


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
    def __init__(
        self,
        config: ZAIConfig,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config
        self._owns_client = http_client is None
        self._sleep = sleep or time.sleep
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

        last_error: ProviderError | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    json=payload,
                )
            except httpx.TimeoutException as exc:
                last_error = ProviderError(
                    (
                        "ZAI request timed out after "
                        f"{attempt + 1} attempts."
                        if attempt == self.config.max_retries
                        else "ZAI request timed out."
                    )
                )
                if attempt < self.config.max_retries:
                    self._sleep(self._backoff_seconds(attempt))
                    continue
                raise last_error from exc
            except httpx.RequestError as exc:
                raise ProviderError(f"ZAI request failed: {exc}") from exc

            if response.status_code == 429:
                detail = self._error_detail(response)
                retry_after = self._retry_after_seconds(response)
                last_error = ProviderRateLimitError(
                    (
                        f"ZAI request failed after {attempt + 1} attempts: {detail}"
                        if attempt == self.config.max_retries
                        else f"ZAI request was rate limited: {detail}"
                    ),
                    retry_after_seconds=retry_after,
                )
                if attempt < self.config.max_retries:
                    self._sleep(retry_after or self._backoff_seconds(attempt))
                    continue
                raise last_error

            if 500 <= response.status_code < 600:
                detail = self._error_detail(response)
                last_error = ProviderServerError(
                    (
                        f"ZAI request failed after {attempt + 1} attempts: {detail}"
                        if attempt == self.config.max_retries
                        else f"ZAI server error: {detail}"
                    ),
                    status_code=response.status_code,
                )
                if attempt < self.config.max_retries:
                    self._sleep(self._backoff_seconds(attempt))
                    continue
                raise last_error

            if response.is_error:
                raise ProviderError(
                    f"ZAI request failed: {self._error_detail(response)}",
                    status_code=response.status_code,
                )

            try:
                completion = ZAIChatCompletionResponse.model_validate(response.json())
            except (ValueError, ValidationError) as exc:
                raise ProviderError("ZAI returned an invalid response payload.") from exc

            if not completion.choices:
                raise ProviderError("ZAI returned no completion choices.")

            return completion.choices[0].message.to_message()

        if last_error is not None:
            raise last_error
        raise ProviderError("ZAI request failed unexpectedly.")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _backoff_seconds(self, attempt: int) -> float:
        delay = self.config.retry_backoff_seconds * (2**attempt)
        return min(delay, self.config.max_retry_backoff_seconds)

    def _retry_after_seconds(self, response: httpx.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if not value:
            return None

        try:
            delay = float(value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = (retry_at - datetime.now(timezone.utc)).total_seconds()

        return max(delay, 0.0)

    def _error_detail(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()
            if isinstance(error, str) and error.strip():
                return error.strip()

        if response.reason_phrase:
            return response.reason_phrase
        return f"HTTP {response.status_code}"
