from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from typing import Literal

from pi.agent.context import AgentContext, ContextManager
from pi.agent.models import Message, ToolCall
from pi.agent.providers.base import Provider
from pi.agent.tools import ToolRegistry


@dataclass(slots=True)
class AgentResult:
    output: str
    messages: list[Message]
    iterations: int


@dataclass(slots=True)
class BeforeToolCallContext:
    iteration: int
    tool_call: ToolCall
    arguments: dict[str, object]
    context: AgentContext


@dataclass(slots=True)
class BeforeToolCallResult:
    block: bool = False
    reason: str | None = None
    arguments: dict[str, object] | None = None


@dataclass(slots=True)
class AfterToolCallContext:
    iteration: int
    tool_call: ToolCall
    arguments: dict[str, object]
    result: dict[str, object]
    context: AgentContext


@dataclass(slots=True)
class AfterToolCallResult:
    result: dict[str, object] | None = None


@dataclass(slots=True)
class PreparedToolCall:
    tool_call: ToolCall
    arguments: dict[str, object]


class MaxIterationsExceededError(RuntimeError):
    pass


AgentEventHandler = Callable[[str, dict[str, object]], None]
BeforeToolCallHook = Callable[[BeforeToolCallContext], BeforeToolCallResult | None]
AfterToolCallHook = Callable[[AfterToolCallContext], AfterToolCallResult | None]
ToolExecutionMode = Literal["parallel", "sequential"]


class Agent:
    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry,
        system_prompt: str | None = None,
        max_iterations: int = 8,
        context_manager: ContextManager | None = None,
        tool_execution: ToolExecutionMode = "parallel",
        before_tool_call: BeforeToolCallHook | None = None,
        after_tool_call: AfterToolCallHook | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.context_manager = context_manager or ContextManager(system_prompt=system_prompt)
        self.tool_execution = tool_execution
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call

    def run(
        self,
        prompt: str,
        messages: list[Message] | None = None,
        *,
        on_event: AgentEventHandler | None = None,
        before_tool_call: BeforeToolCallHook | None = None,
        after_tool_call: AfterToolCallHook | None = None,
    ) -> AgentResult:
        context = self.context_manager.initialize(prompt, messages)
        active_before_hook = before_tool_call or self.before_tool_call
        active_after_hook = after_tool_call or self.after_tool_call

        for iteration in range(1, self.max_iterations + 1):
            provider_messages = self.context_manager.messages_for_provider(context)
            self._emit(on_event, "model_start", {"iteration": iteration, "message_count": len(provider_messages)})
            assistant_message = self.provider.complete(provider_messages, self.tools.definitions())
            self._emit(
                on_event,
                "model_end",
                {
                    "iteration": iteration,
                    "tool_calls": len(assistant_message.tool_calls),
                    "content": assistant_message.content or "",
                },
            )
            self.context_manager.append_message(context, assistant_message)

            if not assistant_message.tool_calls:
                self._emit(on_event, "iteration_end", {"iteration": iteration, "stop_reason": "assistant"})
                return AgentResult(
                    output=assistant_message.content or "",
                    messages=context.messages,
                    iterations=iteration,
                )

            prepared_calls = self._prepare_tool_calls(
                iteration=iteration,
                context=context,
                tool_calls=assistant_message.tool_calls,
                on_event=on_event,
                before_tool_call=active_before_hook,
            )
            tool_results = self._execute_tool_calls(
                iteration=iteration,
                prepared_calls=prepared_calls,
                context=context,
                on_event=on_event,
                after_tool_call=active_after_hook,
            )
            for tool_call, result in tool_results:
                self.context_manager.append_tool_result(context, tool_call_id=tool_call.id, result=result)
            self._emit(on_event, "iteration_end", {"iteration": iteration, "stop_reason": "tool_calls"})

        raise MaxIterationsExceededError(f"Agent exceeded max_iterations={self.max_iterations}")

    def _prepare_tool_calls(
        self,
        *,
        iteration: int,
        context: AgentContext,
        tool_calls: list[ToolCall],
        on_event: AgentEventHandler | None,
        before_tool_call: BeforeToolCallHook | None,
    ) -> list[PreparedToolCall | tuple[ToolCall, dict[str, object]]]:
        prepared: list[PreparedToolCall | tuple[ToolCall, dict[str, object]]] = []
        for tool_call in tool_calls:
            self._emit(
                on_event,
                "tool_execution_start",
                {
                    "iteration": iteration,
                    "tool_name": tool_call.function.name,
                    "tool_call_id": tool_call.id,
                    "tool_arguments": tool_call.function.arguments,
                },
            )
            try:
                parsed_arguments = self.tools.parse_arguments(tool_call)
                invocation = self.tools.prepare(tool_call.function.name, parsed_arguments)
                arguments = invocation.model_dump(by_alias=True, exclude_none=True)
                if before_tool_call is not None:
                    hook_result = before_tool_call(
                        BeforeToolCallContext(
                            iteration=iteration,
                            tool_call=tool_call,
                            arguments=dict(arguments),
                            context=context,
                        )
                    )
                    if hook_result is not None and hook_result.block:
                        prepared.append((tool_call, {"ok": False, "error": hook_result.reason or "Tool execution was blocked"}))
                        continue
                    if hook_result is not None and hook_result.arguments is not None:
                        invocation = self.tools.prepare(tool_call.function.name, hook_result.arguments)
                        arguments = invocation.model_dump(by_alias=True, exclude_none=True)
                prepared.append(PreparedToolCall(tool_call=tool_call, arguments=arguments))
            except Exception as exc:
                prepared.append((tool_call, {"ok": False, "error": str(exc)}))
        return prepared

    def _execute_tool_calls(
        self,
        *,
        iteration: int,
        prepared_calls: list[PreparedToolCall | tuple[ToolCall, dict[str, object]]],
        context: AgentContext,
        on_event: AgentEventHandler | None,
        after_tool_call: AfterToolCallHook | None,
    ) -> list[tuple[ToolCall, dict[str, object]]]:
        results: list[tuple[ToolCall, dict[str, object]]] = []
        runnable = [item for item in prepared_calls if isinstance(item, PreparedToolCall)]
        immediate = [item for item in prepared_calls if not isinstance(item, PreparedToolCall)]

        for tool_call, result in immediate:
            finalized = self._finalize_tool_result(
                iteration=iteration,
                tool_call=tool_call,
                arguments={},
                result=result,
                context=context,
                on_event=on_event,
                after_tool_call=after_tool_call,
            )
            results.append((tool_call, finalized))

        if self.tool_execution == "sequential" or len(runnable) < 2:
            for prepared in runnable:
                raw_result = self.tools.execute_name(prepared.tool_call.function.name, prepared.arguments)
                finalized = self._finalize_tool_result(
                    iteration=iteration,
                    tool_call=prepared.tool_call,
                    arguments=prepared.arguments,
                    result=raw_result,
                    context=context,
                    on_event=on_event,
                    after_tool_call=after_tool_call,
                )
                results.append((prepared.tool_call, finalized))
            return self._order_results(prepared_calls, results)

        with ThreadPoolExecutor(max_workers=len(runnable)) as executor:
            future_pairs = [
                (
                    prepared,
                    executor.submit(self.tools.execute_name, prepared.tool_call.function.name, prepared.arguments),
                )
                for prepared in runnable
            ]
            for prepared, future in future_pairs:
                raw_result = future.result()
                finalized = self._finalize_tool_result(
                    iteration=iteration,
                    tool_call=prepared.tool_call,
                    arguments=prepared.arguments,
                    result=raw_result,
                    context=context,
                    on_event=on_event,
                    after_tool_call=after_tool_call,
                )
                results.append((prepared.tool_call, finalized))
        return self._order_results(prepared_calls, results)

    def _order_results(
        self,
        prepared_calls: list[PreparedToolCall | tuple[ToolCall, dict[str, object]]],
        results: list[tuple[ToolCall, dict[str, object]]],
    ) -> list[tuple[ToolCall, dict[str, object]]]:
        ordered = {tool_call.id: (tool_call, result) for tool_call, result in results}
        return [
            ordered[item.tool_call.id if isinstance(item, PreparedToolCall) else item[0].id]
            for item in prepared_calls
            if (item.tool_call.id if isinstance(item, PreparedToolCall) else item[0].id) in ordered
        ]

    def _finalize_tool_result(
        self,
        *,
        iteration: int,
        tool_call: ToolCall,
        arguments: dict[str, object],
        result: dict[str, object],
        context: AgentContext,
        on_event: AgentEventHandler | None,
        after_tool_call: AfterToolCallHook | None,
    ) -> dict[str, object]:
        finalized = dict(result)
        if after_tool_call is not None:
            hook_result = after_tool_call(
                AfterToolCallContext(
                    iteration=iteration,
                    tool_call=tool_call,
                    arguments=arguments,
                    result=dict(finalized),
                    context=context,
                )
            )
            if hook_result is not None and hook_result.result is not None:
                finalized = hook_result.result
        self._emit(
            on_event,
            "tool_execution_end",
            {
                "iteration": iteration,
                "tool_name": tool_call.function.name,
                "tool_call_id": tool_call.id,
                "ok": finalized.get("ok", False),
                "result": finalized,
            },
        )
        return finalized

    def _emit(self, handler: AgentEventHandler | None, event: str, payload: dict[str, object]) -> None:
        if handler is not None:
            handler(event, payload)


def create_agent(
    *,
    provider: Provider,
    tools: ToolRegistry,
    system_prompt: str | None = None,
    max_iterations: int = 8,
    tool_execution: ToolExecutionMode = "parallel",
    before_tool_call: BeforeToolCallHook | None = None,
    after_tool_call: AfterToolCallHook | None = None,
) -> Agent:
    return Agent(
        provider=provider,
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        tool_execution=tool_execution,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
    )


def run_task(
    prompt: str,
    *,
    provider: Provider,
    tools: ToolRegistry,
    system_prompt: str | None = None,
    max_iterations: int = 8,
    messages: list[Message] | None = None,
    tool_execution: ToolExecutionMode = "parallel",
    before_tool_call: BeforeToolCallHook | None = None,
    after_tool_call: AfterToolCallHook | None = None,
    on_event: AgentEventHandler | None = None,
) -> AgentResult:
    agent = create_agent(
        provider=provider,
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        tool_execution=tool_execution,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
    )
    return agent.run(prompt, messages=messages, on_event=on_event)
