from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import BaseModel, Field

from pi.agent.models import Message

MessageTransform = Callable[[list[Message]], list[Message]]


class AgentContext(BaseModel):
    system_prompt: str | None = None
    messages: list[Message] = Field(default_factory=list)


class ContextManager:
    def __init__(
        self,
        system_prompt: str | None = None,
        transform_messages: MessageTransform | None = None,
        convert_messages: MessageTransform | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.transform_messages = transform_messages
        self.convert_messages = convert_messages

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
        if self.convert_messages is not None:
            messages = self.convert_messages(messages)
        return [message.model_copy(deep=True) for message in messages]
