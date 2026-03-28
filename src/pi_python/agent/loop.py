from __future__ import annotations

import json
from dataclasses import dataclass

from pi_python.agent.models import Message
from pi_python.agent.providers.base import Provider
from pi_python.agent.tools import ToolRegistry


@dataclass(slots=True)
class AgentResult:
    output: str
    messages: list[Message]
    iterations: int


class MaxIterationsExceededError(RuntimeError):
    pass


class Agent:
    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry,
        system_prompt: str | None = None,
        max_iterations: int = 8,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

    def run(
        self,
        prompt: str,
        messages: list[Message] | None = None,
    ) -> AgentResult:
        conversation = [message.model_copy(deep=True) for message in messages or []]
        if self.system_prompt and not any(
            message.role == "system" for message in conversation
        ):
            conversation.insert(0, Message.system(self.system_prompt))
        conversation.append(Message.user(prompt))

        for iteration in range(1, self.max_iterations + 1):
            assistant_message = self.provider.complete(
                conversation, self.tools.definitions()
            )
            conversation.append(assistant_message)

            if not assistant_message.tool_calls:
                return AgentResult(
                    output=assistant_message.content or "",
                    messages=conversation,
                    iterations=iteration,
                )

            for tool_call in assistant_message.tool_calls:
                result = self.tools.execute(tool_call)
                conversation.append(
                    Message.tool(
                        tool_call_id=tool_call.id,
                        content=json.dumps(result, ensure_ascii=False),
                    )
                )

        raise MaxIterationsExceededError(
            f"Agent exceeded max_iterations={self.max_iterations}"
        )
