from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pi.agent.loop import Agent, AgentResult
from pi.agent.models import Message
from pi.agent.providers.base import Provider
from pi.agent.tools import ToolRegistry


@dataclass(slots=True)
class Context:
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)

    @classmethod
    def from_prompt(cls, prompt: str, *, system_prompt: str | None = None) -> "Context":
        messages = [Message.user(prompt)]
        if system_prompt is not None:
            messages.insert(0, Message.system(system_prompt))
        return cls(system_prompt=system_prompt, messages=messages)


@dataclass(slots=True, frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, object]
    handler: Callable[[dict[str, Any]], Any] | None = None

    def to_definition(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class CompletionResult:
    message: Message
    output: str
    messages: list[Message]


@dataclass(slots=True, frozen=True)
class StreamEvent:
    type: str
    message: Message | None = None
    delta: str | None = None


def complete(
    *,
    provider: Provider,
    prompt: str | None = None,
    context: Context | None = None,
    messages: list[Message] | None = None,
    system_prompt: str | None = None,
    tools: list[Tool] | None = None,
) -> CompletionResult:
    resolved_messages = _build_messages(prompt=prompt, context=context, messages=messages, system_prompt=system_prompt)
    response = provider.complete(resolved_messages, [tool.to_definition() for tool in tools or []])
    return CompletionResult(
        message=response,
        output=response.content or "",
        messages=[*resolved_messages, response],
    )


def stream(
    *,
    provider: Provider,
    prompt: str | None = None,
    context: Context | None = None,
    messages: list[Message] | None = None,
    system_prompt: str | None = None,
    tools: list[Tool] | None = None,
) -> Iterator[StreamEvent]:
    completion = complete(
        provider=provider,
        prompt=prompt,
        context=context,
        messages=messages,
        system_prompt=system_prompt,
        tools=tools,
    )
    yield StreamEvent(type="message_start", message=completion.message)
    if completion.output:
        yield StreamEvent(type="message_delta", delta=completion.output)
    yield StreamEvent(type="message_end", message=completion.message)


def create_agent(
    *,
    provider: Provider,
    root: str | Path,
    system_prompt: str | None = None,
    max_iterations: int = 8,
    tool_mode: str = "coding",
) -> Agent:
    root_path = Path(root)
    registry = {
        "coding": ToolRegistry.coding,
        "read_only": ToolRegistry.read_only,
        "all": ToolRegistry.all,
    }.get(tool_mode)
    if registry is None:
        raise ValueError(f"Unknown tool_mode: {tool_mode}")
    return Agent(
        provider=provider,
        tools=registry(root_path),
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )


def run_task(
    prompt: str,
    *,
    provider: Provider,
    root: str | Path,
    system_prompt: str | None = None,
    max_iterations: int = 8,
    tool_mode: str = "coding",
    messages: list[Message] | None = None,
) -> AgentResult:
    agent = create_agent(
        provider=provider,
        root=root,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        tool_mode=tool_mode,
    )
    return agent.run(prompt, messages=messages)


def _build_messages(
    *,
    prompt: str | None,
    context: Context | None,
    messages: list[Message] | None,
    system_prompt: str | None,
) -> list[Message]:
    if context is not None:
        resolved = [message.model_copy(deep=True) for message in context.messages]
        if prompt is not None:
            resolved.append(Message.user(prompt))
        return resolved
    resolved = [message.model_copy(deep=True) for message in messages or []]
    if system_prompt and not any(message.role == "system" for message in resolved):
        resolved.insert(0, Message.system(system_prompt))
    if prompt is not None:
        resolved.append(Message.user(prompt))
    if not resolved:
        raise ValueError("Provide prompt, context, or messages.")
    return resolved
