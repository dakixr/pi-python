from __future__ import annotations

from dataclasses import dataclass

from pi.agent.context import ContextManager
from pi.agent.models import Message
from pi.agent.providers.base import Provider
from pi.agent.tools import ToolRegistry


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
        context_manager: ContextManager | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.context_manager = context_manager or ContextManager(system_prompt=system_prompt)

    def run(
        self,
        prompt: str,
        messages: list[Message] | None = None,
    ) -> AgentResult:
        context = self.context_manager.initialize(prompt, messages)

        for iteration in range(1, self.max_iterations + 1):
            assistant_message = self.provider.complete(
                self.context_manager.messages_for_provider(context),
                self.tools.definitions(),
            )
            self.context_manager.append_message(context, assistant_message)

            if not assistant_message.tool_calls:
                return AgentResult(
                    output=assistant_message.content or "",
                    messages=context.messages,
                    iterations=iteration,
                )

            for tool_call in assistant_message.tool_calls:
                result = self.tools.execute(tool_call)
                self.context_manager.append_tool_result(
                    context,
                    tool_call_id=tool_call.id,
                    result=result,
                )

        raise MaxIterationsExceededError(
            f"Agent exceeded max_iterations={self.max_iterations}"
        )
