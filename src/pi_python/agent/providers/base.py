from __future__ import annotations

from typing import Protocol

from pi_python.agent.models import Message


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProviderRateLimitError(ProviderError):
    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after_seconds = retry_after_seconds


class ProviderServerError(ProviderError):
    pass


class Provider(Protocol):
    def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> Message:
        ...
