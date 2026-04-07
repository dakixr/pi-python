from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from pi.agent.models import Message, ToolCall
from pi.agent.providers.base import Provider, ProviderError, ProviderRateLimitError, ProviderServerError


class ZAIConfig(BaseModel):
    api_key: str
    model: str = "glm-5.1"
    base_url: str = "https://api.z.ai/api/coding/paas/v4"
    timeout_seconds: float | None = None
    debug_log_path: str | None = None
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
        prepared_messages = self._prepare_messages(messages)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_api_dict() for message in prepared_messages],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error: ProviderError | None = None
        attempted_message_recovery = False

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
                detail = self._error_detail(response)
                if (
                    not attempted_message_recovery
                    and self._looks_like_illegal_messages_error(detail)
                ):
                    self._log_debug_event(
                        "illegal_messages_error",
                        detail=detail,
                        messages=[message.to_api_dict() for message in prepared_messages],
                    )
                    recovered_messages = self._recover_illegal_messages(prepared_messages)
                    attempted_message_recovery = True
                    if recovered_messages != prepared_messages:
                        self._log_debug_event(
                            "illegal_messages_recovery",
                            detail=detail,
                            messages=[message.to_api_dict() for message in recovered_messages],
                        )
                        prepared_messages = recovered_messages
                        payload["messages"] = [message.to_api_dict() for message in prepared_messages]
                        continue
                raise ProviderError(
                    f"ZAI request failed: {detail}",
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

    def _prepare_messages(self, messages: list[Message]) -> list[Message]:
        prepared = [message.model_copy(deep=True) for message in messages]
        prepared = self._merge_leading_system_messages(prepared)
        for message in prepared:
            if message.role == "assistant" and message.content is None:
                message.content = ""
        prepared = self._sanitize_message_sequence(prepared)
        prepared = self._project_messages_for_zai(prepared)
        prepared = self._drop_empty_assistant_messages(prepared)
        return prepared

    def _project_messages_for_zai(self, messages: list[Message]) -> list[Message]:
        system_messages: list[Message] = []
        start = 0
        while start < len(messages) and messages[start].role == "system":
            system_messages.append(messages[start].model_copy(deep=True))
            start += 1

        conversation = messages[start:]
        trailing_exchange_start = self._trailing_structured_exchange_start(conversation)
        if trailing_exchange_start is None:
            projected = self._coalesce_text_messages(self._render_tool_messages_as_text(conversation))
            return [*system_messages, *projected]

        prefix = self._render_tool_messages_as_text(conversation[:trailing_exchange_start])
        suffix = [message.model_copy(deep=True) for message in conversation[trailing_exchange_start:]]
        return [*system_messages, *self._coalesce_text_messages([*prefix, *suffix])]

    def _trailing_structured_exchange_start(self, messages: list[Message]) -> int | None:
        trailing_tool_ids: list[str] = []
        index = len(messages) - 1
        while index >= 0 and messages[index].role == "tool":
            if messages[index].tool_call_id:
                trailing_tool_ids.append(messages[index].tool_call_id)
            index -= 1
        if not trailing_tool_ids or index < 0:
            return None
        assistant_message = messages[index]
        if assistant_message.role != "assistant" or not assistant_message.tool_calls:
            return None
        expected_ids = {tool_call.id for tool_call in assistant_message.tool_calls}
        return index if expected_ids == set(trailing_tool_ids) else None

    def _merge_leading_system_messages(self, messages: list[Message]) -> list[Message]:
        leading_system_messages: list[Message] = []
        for message in messages:
            if message.role != "system":
                break
            leading_system_messages.append(message)
        if len(leading_system_messages) <= 1:
            return messages
        merged_content = "\n\n".join(
            content
            for message in leading_system_messages
            if (content := message.content) and content.strip()
        )
        return [Message.system(merged_content), *messages[len(leading_system_messages):]]

    def _sanitize_message_sequence(self, messages: list[Message]) -> list[Message]:
        sanitized: list[Message] = []
        exchange_start: int | None = None
        pending_tool_ids: set[str] = set()
        pending_assistant: Message | None = None

        def drop_pending_exchange() -> None:
            nonlocal exchange_start, pending_tool_ids, pending_assistant
            if exchange_start is None or pending_assistant is None:
                pending_tool_ids.clear()
                exchange_start = None
                pending_assistant = None
                return
            replacement: list[Message] = []
            content = pending_assistant.content or ""
            if content.strip():
                replacement.append(Message.assistant(content))
            del sanitized[exchange_start:]
            sanitized.extend(replacement)
            pending_tool_ids.clear()
            exchange_start = None
            pending_assistant = None

        for message in messages:
            if message.role == "assistant" and message.tool_calls:
                if pending_tool_ids:
                    drop_pending_exchange()
                sanitized.append(message)
                exchange_start = len(sanitized) - 1
                pending_assistant = message
                pending_tool_ids = {tool_call.id for tool_call in message.tool_calls}
                continue
            if message.role == "tool":
                if not pending_tool_ids or message.tool_call_id not in pending_tool_ids:
                    continue
                sanitized.append(message)
                pending_tool_ids.remove(message.tool_call_id)
                if not pending_tool_ids:
                    exchange_start = None
                    pending_assistant = None
                continue
            if pending_tool_ids:
                drop_pending_exchange()
            sanitized.append(message)

        if pending_tool_ids:
            drop_pending_exchange()
        return sanitized

    def _drop_empty_assistant_messages(self, messages: list[Message]) -> list[Message]:
        return [
            message
            for message in messages
            if not (
                message.role == "assistant"
                and not message.tool_calls
                and not (message.content or "").strip()
            )
        ]

    def _looks_like_illegal_messages_error(self, detail: str) -> bool:
        normalized = detail.lower()
        return "messages parameter is illegal" in normalized or "messages parameter" in normalized

    def _recover_illegal_messages(self, messages: list[Message]) -> list[Message]:
        recovered: list[Message] = []
        system_messages: list[Message] = []
        index = 0
        while index < len(messages) and messages[index].role == "system":
            system_messages.append(messages[index].model_copy(deep=True))
            index += 1
        if system_messages:
            recovered.extend(self._merge_leading_system_messages(system_messages))
        textual_messages = self._render_tool_messages_as_text(messages[index:])
        recovered.extend(self._ensure_recoverable_dialogue_shape(self._coalesce_text_messages(textual_messages)))
        return recovered

    def _render_tool_messages_as_text(self, messages: list[Message]) -> list[Message]:
        rendered: list[Message] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.role == "assistant" and message.tool_calls:
                tool_ids = [tool_call.id for tool_call in message.tool_calls]
                tool_results: list[Message] = []
                lookahead = index + 1
                while lookahead < len(messages):
                    candidate = messages[lookahead]
                    if candidate.role == "tool" and candidate.tool_call_id in tool_ids:
                        tool_results.append(candidate)
                        lookahead += 1
                        continue
                    break
                content = self._render_tool_exchange(message, tool_results)
                if content:
                    rendered.append(Message.assistant(content))
                index = lookahead
                continue
            if message.role in {"user", "assistant"} and (message.content or "").strip():
                rendered.append(Message(role=message.role, content=(message.content or "").strip()))
            index += 1
        return rendered

    def _render_tool_exchange(self, assistant_message: Message, tool_results: list[Message]) -> str:
        parts: list[str] = []
        if (assistant_content := (assistant_message.content or "").strip()):
            parts.append(assistant_content)
        calls = [f"{tool_call.function.name}({tool_call.function.arguments})" for tool_call in assistant_message.tool_calls]
        if calls:
            parts.append(f"[Assistant tool calls] {'; '.join(calls)}")
        for tool_message in tool_results:
            if (tool_content := (tool_message.content or "").strip()):
                parts.append(f"[Tool result] {self._truncate_text(tool_content, limit=1_200)}")
        return "\n".join(parts).strip()

    def _coalesce_text_messages(self, messages: list[Message]) -> list[Message]:
        coalesced: list[Message] = []
        for message in messages:
            if coalesced and coalesced[-1].role == message.role and message.role != "system":
                merged_content = "\n\n".join(
                    part for part in [coalesced[-1].content or "", message.content or ""] if part
                )
                coalesced[-1] = Message(role=message.role, content=merged_content)
                continue
            coalesced.append(message)
        return coalesced

    def _ensure_recoverable_dialogue_shape(self, messages: list[Message]) -> list[Message]:
        if not messages:
            return [Message.user(self._recovery_prompt())]
        if messages[-1].role != "user":
            return [*messages, Message.user(self._recovery_prompt())]
        return messages

    def _recovery_prompt(self) -> str:
        return "Continue from the prior context and answer the latest request using the tool results already gathered."

    def _truncate_text(self, text: str, *, limit: int) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _log_debug_event(self, event: str, **payload: object) -> None:
        if not self.config.debug_log_path:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "model": self.config.model,
            **payload,
        }
        try:
            path = Path(self.config.debug_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            return

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
            if isinstance(error, dict) and isinstance(error.get("message"), str) and error["message"].strip():
                return error["message"].strip()
            if isinstance(error, str) and error.strip():
                return error.strip()
        return response.reason_phrase or f"HTTP {response.status_code}"
