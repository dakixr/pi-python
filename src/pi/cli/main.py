from __future__ import annotations

from contextlib import nullcontext
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from queue import Empty, Queue
from threading import Thread
import time
from typing import Annotated, Protocol

import typer

from pi.agent.loop import Agent, AgentEventHandler, AgentResult, MaxIterationsExceededError
from pi.agent.models import Message
from pi.agent.providers.base import ProviderError
from pi.agent.providers.zai import ZAIConfig, ZAIProvider
from pi.agent.tools import ToolRegistry
from pi.cli.render import (
    InteractiveRenderer,
    OutputStream,
    StatusIndicator,
    build_console,
    print_agent_output,
    print_error,
    print_user_prompt,
)
from pi.cli.session import SessionStore

DEFAULT_SYSTEM_PROMPT = (
    "You are pi, a minimal coding agent. "
    "Use only the provided tools, do not invent unavailable capabilities, "
    "and finish with a direct assistant answer."
)
DEFAULT_MODEL = "glm-5.1"
DEFAULT_MAX_ITERATIONS = 20
RUN_ERRORS = (ProviderError, MaxIterationsExceededError)


class AgentRunner(Protocol):
    def run(
        self,
        prompt: str,
        messages: list[Message] | None = None,
        *,
        on_event: AgentEventHandler | None = None,
    ) -> AgentResult: ...


class InputFunc(Protocol):
    def __call__(self, prompt: object = "", /) -> str: ...


@dataclass(slots=True)
class TurnSuccess:
    result: AgentResult


@dataclass(slots=True)
class TurnFailure:
    error: Exception


@dataclass(slots=True)
class CLIArgs:
    prompt: str | None = None
    session: str | None = None
    api_key: str = ""
    model: str = DEFAULT_MODEL
    root: str = os.getcwd()
    max_iterations: int = DEFAULT_MAX_ITERATIONS


def build_agent_from_args(args: CLIArgs) -> Agent:
    if not args.api_key:
        raise ValueError("A ZAI API key is required. Pass --api-key or set ZAI_API_KEY.")
    return Agent(
        provider=ZAIProvider(
            config=ZAIConfig(
                api_key=args.api_key,
                model=args.model,
                debug_log_path=str(Path(args.root) / ".pi" / "logs" / "zai-debug.jsonl"),
            )
        ),
        tools=ToolRegistry(root=Path(args.root)),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_iterations=args.max_iterations,
    )


def execute_turn(
    agent: AgentRunner,
    prompt: str,
    messages: list[Message],
    *,
    stderr: OutputStream | None = None,
    indicator: StatusIndicator | None = None,
) -> AgentResult:
    active_indicator = indicator or StatusIndicator(stderr or sys.stderr)
    run_signature = inspect.signature(agent.run)
    with active_indicator:
        if "on_event" in run_signature.parameters:
            return agent.run(prompt, messages=messages, on_event=active_indicator.handle_event)
        return agent.run(prompt, messages=messages)


def run_interactive_cli(
    args: CLIArgs,
    agent: AgentRunner,
    session_messages: list[Message],
    *,
    session_store: SessionStore | None,
    input_func: InputFunc,
    stdout: OutputStream,
    stderr: OutputStream,
) -> int:
    messages = list(session_messages)
    stdout_console = build_console(stdout)
    renderer = InteractiveRenderer(stdout)
    input_queue: Queue[str | None] = Queue()
    result_queue: Queue[TurnSuccess | TurnFailure] = Queue()
    pending_prompts: list[str] = []
    worker: Thread | None = None
    shutdown_requested = False
    input_closed = False

    use_prompt_toolkit = input_func is input and stdout_console.is_terminal

    def read_input() -> None:
        prompt_session = None
        if use_prompt_toolkit:
            from prompt_toolkit import PromptSession

            prompt_session = PromptSession()
        while True:
            try:
                raw = prompt_session.prompt(">>> ") if prompt_session is not None else input_func(">>> ")
            except (EOFError, StopIteration):
                input_queue.put(None)
                return
            input_queue.put(raw.strip())

    def start_turn(prompt: str) -> Thread:
        history = list(messages)

        def run_turn() -> None:
            try:
                result = execute_turn(agent, prompt, history, stderr=stderr, indicator=renderer)
            except RUN_ERRORS as exc:
                result_queue.put(TurnFailure(error=exc))
                return
            result_queue.put(TurnSuccess(result=result))

        thread = Thread(target=run_turn, daemon=True)
        thread.start()
        return thread

    patch_context = nullcontext()
    if use_prompt_toolkit:
        from prompt_toolkit.patch_stdout import patch_stdout

        patch_context = patch_stdout(raw=True)

    with patch_context:
        Thread(target=read_input, daemon=True).start()
        if stdout_console.is_terminal:
            stdout_console.print("[bold cyan]pi[/bold cyan] interactive mode. Type `exit` or `quit` to leave.\n")
        while True:
            try:
                while True:
                    prompt = input_queue.get_nowait()
                    if prompt is None:
                        input_closed = True
                        shutdown_requested = True
                        continue
                    if not prompt:
                        continue
                    if prompt.lower() in {"exit", "quit"}:
                        shutdown_requested = True
                        continue
                    if worker is None and not pending_prompts:
                        print_user_prompt(stdout_console, prompt)
                        worker = start_turn(prompt)
                    else:
                        pending_prompts.append(prompt)
                        renderer.log_queued_message(prompt)
                        renderer.set_queue_count(len(pending_prompts))
            except Empty:
                pass

            try:
                outcome = result_queue.get_nowait()
            except Empty:
                outcome = None

            if outcome is not None:
                worker = None
                if isinstance(outcome, TurnFailure):
                    print_error(stdout_console, str(outcome.error))
                else:
                    messages = outcome.result.messages
                    if session_store and args.session:
                        session_store.save(args.session, messages)
                    print_agent_output(stdout_console, outcome.result.output)
                renderer.set_queue_count(len(pending_prompts))
                if pending_prompts:
                    next_prompt = pending_prompts.pop(0)
                    renderer.set_queue_count(len(pending_prompts))
                    print_user_prompt(stdout_console, next_prompt)
                    worker = start_turn(next_prompt)

            if shutdown_requested and worker is None and not pending_prompts:
                if input_closed:
                    stdout_console.print()
                return 0

            time.sleep(0.05)


def run_cli(
    args: CLIArgs,
    agent: AgentRunner | None = None,
    *,
    input_func: InputFunc = input,
    stdout: OutputStream | None = None,
    stderr: OutputStream | None = None,
) -> int:
    error_stream = stderr or sys.stderr
    output_stream = stdout or sys.stdout
    error_console = build_console(error_stream)
    output_console = build_console(output_stream)
    try:
        active_agent = agent or build_agent_from_args(args)
    except ValueError as exc:
        print_error(error_console, str(exc))
        return 1
    session_store = SessionStore(root=Path(args.root)) if args.session else None
    session_messages = session_store.load(args.session).messages if session_store and args.session else []
    if args.prompt:
        try:
            result = execute_turn(active_agent, args.prompt, session_messages, stderr=error_stream)
        except RUN_ERRORS as exc:
            print_error(error_console, str(exc))
            return 1
        if session_store and args.session:
            session_store.save(args.session, result.messages)
        print_agent_output(output_console, result.output)
        return 0
    return run_interactive_cli(
        args,
        active_agent,
        session_messages,
        session_store=session_store,
        input_func=input_func,
        stdout=output_stream,
        stderr=error_stream,
    )

app = typer.Typer(
    add_completion=False,
    help="Native Python coding agent CLI.",
    rich_markup_mode="rich",
    no_args_is_help=False,
)


@app.command()
def cli(
    prompt: Annotated[str | None, typer.Option("--prompt", help="Run one prompt and exit.")] = None,
    session: Annotated[
        str | None,
        typer.Option("--session", help="Persist conversation under [bold].pi/sessions/<name>.json[/bold]."),
    ] = None,
    api_key: Annotated[str, typer.Option("--api-key", envvar="ZAI_API_KEY", help="Z.AI API key.")] = "",
    model: Annotated[str, typer.Option("--model", help="Model name to send to the provider.")] = DEFAULT_MODEL,
    root: Annotated[
        Path,
        typer.Option("--root", help="Workspace root for tools.", file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path.cwd(),
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", min=1, help="Maximum number of agent iterations per prompt."),
    ] = DEFAULT_MAX_ITERATIONS,
) -> None:
    raise typer.Exit(
        run_cli(
            CLIArgs(
                prompt=prompt,
                session=session,
                api_key=api_key,
                model=model,
                root=str(root),
                max_iterations=max_iterations,
            )
        )
    )


def main(argv: list[str] | None = None) -> int:
    try:
        app(args=argv, prog_name="pi", standalone_mode=False)
    except typer.Exit as exc:
        return int(exc.exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
