from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import BaseModel, Field

from pi.agent.models import Message

MessageTransform = Callable[[list[Message]], list[Message]]
DEFAULT_MAX_PROVIDER_CHARS = 60_000
DEFAULT_KEEP_RECENT_CHARS = 20_000
MAX_SUMMARY_CHARS = 12_000
COMPACTION_BOUNDARY_ROLES = {"user", "assistant"}


class AgentContext(BaseModel):
    system_prompt: str | None = None
    messages: list[Message] = Field(default_factory=list)


class ContextManager:
    def __init__(
        self,
        system_prompt: str | None = None,
        transform_messages: MessageTransform | None = None,
        convert_messages: MessageTransform | None = None,
        max_provider_chars: int | None = DEFAULT_MAX_PROVIDER_CHARS,
        keep_recent_chars: int = DEFAULT_KEEP_RECENT_CHARS,
    ) -> None:
        self.system_prompt = system_prompt
        self.transform_messages = transform_messages
        self.convert_messages = convert_messages
        self.max_provider_chars = max_provider_chars
        self.keep_recent_chars = keep_recent_chars

    def initialize(
        self,
        prompt: str,
        messages: list[Message] | None = None,
    ) -> AgentContext:
        conversation = [message.model_copy(deep=True) for message in messages or []]
        if self.system_prompt and not any(message.role == "system" for message in conversation):
            conversation.insert(0, Message.system(self.system_prompt))
        conversation.append(Message.user(prompt))
        return AgentContext(system_prompt=self.system_prompt, messages=conversation)

    def append_message(self, context: AgentContext, message: Message) -> None:
        context.messages.append(message)

    def append_tool_result(
        self,
        context: AgentContext,
        *,
        tool_call_id: str,
        result: dict[str, object],
    ) -> Message:
        message = Message.tool(
            tool_call_id=tool_call_id,
            content=json.dumps(result, ensure_ascii=False),
        )
        context.messages.append(message)
        return message

    def messages_for_provider(self, context: AgentContext) -> list[Message]:
        messages = [message.model_copy(deep=True) for message in context.messages]
        if self.transform_messages is not None:
            messages = self.transform_messages(messages)
        if self.max_provider_chars is not None:
            messages = self._compact_messages_for_provider(messages)
        if self.convert_messages is not None:
            messages = self.convert_messages(messages)
        return [message.model_copy(deep=True) for message in messages]

    def _compact_messages_for_provider(self, messages: list[Message]) -> list[Message]:
        if self.max_provider_chars is None or self._estimate_chars(messages) <= self.max_provider_chars:
            return messages

        system_count = 0
        for message in messages:
            if message.role != "system":
                break
            system_count += 1

        system_messages = messages[:system_count]
        conversation = messages[system_count:]
        if len(conversation) < 2:
            return messages

        keep_start = self._find_keep_start(conversation)
        if keep_start <= 0:
            return messages

        summarized = conversation[:keep_start]
        kept = conversation[keep_start:]
        compacted = self._build_compacted_messages(system_messages, summarized, kept)

        while self._estimate_chars(compacted) > self.max_provider_chars and len(kept) > 1:
            next_boundary = self._next_compaction_boundary(kept)
            if next_boundary is None:
                break
            summarized.extend(kept[:next_boundary])
            kept = kept[next_boundary:]
            compacted = self._build_compacted_messages(system_messages, summarized, kept)

        return compacted

    def _find_keep_start(self, messages: list[Message]) -> int:
        total = 0
        keep_start = len(messages) - 1
        for index in range(len(messages) - 1, -1, -1):
            total += self._estimate_chars([messages[index]])
            keep_start = index
            if total >= self.keep_recent_chars:
                break

        while keep_start > 0 and messages[keep_start].role not in COMPACTION_BOUNDARY_ROLES:
            keep_start -= 1
        return keep_start

    def _next_compaction_boundary(self, messages: list[Message]) -> int | None:
        for index in range(1, len(messages)):
            if messages[index].role in COMPACTION_BOUNDARY_ROLES:
                return index
        return None

    def _build_compacted_messages(
        self,
        system_messages: list[Message],
        summarized: list[Message],
        kept: list[Message],
    ) -> list[Message]:
        summary = Message.system(self._build_summary(summarized))
        return [*system_messages, summary, *kept]

    def _build_summary(self, messages: list[Message]) -> str:
        lines = [
            "Auto-compacted summary of earlier conversation. This is lossy; rely on later messages when conflicts exist.",
            "",
        ]
        for message in messages:
            rendered = self._render_message(message)
            if rendered:
                lines.append(rendered)
        summary = "\n".join(lines).strip()
        if len(summary) <= MAX_SUMMARY_CHARS:
            return summary
        return summary[: MAX_SUMMARY_CHARS - 3].rstrip() + "..."

    def _render_message(self, message: Message) -> str:
        parts: list[str] = []
        if message.content:
            parts.append(f"[{message.role.capitalize()}] {self._truncate_text(message.content, limit=600)}")
        if message.tool_calls:
            calls = [
                f"{tool_call.function.name}({self._truncate_text(tool_call.function.arguments, limit=200)})"
                for tool_call in message.tool_calls
            ]
            parts.append(f"[Assistant tool calls] {'; '.join(calls)}")
        if message.role == "tool" and not message.content:
            parts.append(f"[Tool {message.tool_call_id or 'result'}] (empty result)")
        return "\n".join(parts)

    def _truncate_text(self, text: str, *, limit: int) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _estimate_chars(self, messages: list[Message]) -> int:
        return len(json.dumps([message.to_api_dict() for message in messages], ensure_ascii=False))
