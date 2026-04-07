from __future__ import annotations

from collections.abc import Callable
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from queue import Empty, Queue
from threading import Lock, Thread
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
    print_scrollback_agent,
    print_scrollback_error,
    print_scrollback_queue,
    print_scrollback_tool,
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


class EventRenderer(Protocol):
    def __enter__(self) -> "EventRenderer": ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...
    def handle_event(self, event: str, payload: dict[str, object]) -> None: ...


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


class PromptToolkitLiveRenderer:
    def __init__(
        self,
        *,
        emit_tool: Callable[[str, bool], None],
        emit_agent: Callable[[str], None],
        emit_error: Callable[[str], None],
        set_status: Callable[[str], None],
    ) -> None:
        self._emit_tool = emit_tool
        self._emit_agent = emit_agent
        self._emit_error = emit_error
        self._set_status = set_status

    def __enter__(self) -> "PromptToolkitLiveRenderer":
        self._set_status("Thinking")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._set_status("Ready")
        return None

    def handle_event(self, event: str, payload: dict[str, object]) -> None:
        from pi.cli.render import format_tool_preview, truncate_cli_text

        if event == "model_start":
            self._set_status("Thinking")
            return
        if event == "model_end":
            self._set_status("Planning next step")
            return
        if event == "tool_execution_start":
            tool_name = payload.get("tool_name", "tool")
            tool_arguments = payload.get("tool_arguments")
            self._emit_tool(format_tool_preview(str(tool_name), tool_arguments), False)
            self._set_status("Waiting on tool result")
            return
        if event == "tool_execution_end":
            ok = payload.get("ok", False)
            if not ok:
                result = payload.get("result")
                error_text = None
                if isinstance(result, dict):
                    raw_error = result.get("error")
                    if isinstance(raw_error, str) and raw_error.strip():
                        error_text = truncate_cli_text(raw_error, 120)
                self._emit_tool(error_text or "The tool returned an error.", True)
                self._set_status("Handling tool failure")
                return
            self._set_status("Thinking")

    def print_agent_output(self, text: str) -> None:
        self._emit_agent(text)

    def print_error(self, message: str) -> None:
        self._emit_error(message)


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
    indicator: EventRenderer | None = None,
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
    stdout_console = build_console(stdout)
    if input_func is input and stdout_console.is_terminal:
        return run_prompt_toolkit_cli(
            args,
            agent,
            session_messages,
            session_store=session_store,
        )

    return run_basic_interactive_cli(
        args,
        agent,
        session_messages,
        session_store=session_store,
        input_func=input_func,
        stdout=stdout,
        stderr=stderr,
    )


def run_basic_interactive_cli(
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

    def read_input() -> None:
        while True:
            try:
                raw = input_func(">>> ")
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

    Thread(target=read_input, daemon=True).start()
    if stdout_console.is_terminal:
        renderer.print_intro()
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


def run_prompt_toolkit_cli(
    args: CLIArgs,
    agent: AgentRunner,
    session_messages: list[Message],
    *,
    session_store: SessionStore | None,
) -> int:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application.run_in_terminal import run_in_terminal

    messages = list(session_messages)
    pending_prompts: list[str] = []
    worker: Thread | None = None
    shutdown_requested = False
    status_message = "Ready"
    spinner_index = 0
    spinner_frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    state_lock = Lock()
    prompt_session = PromptSession()
    scrollback_console = build_console(sys.stdout)

    def update_status(message: str) -> None:
        nonlocal status_message
        with state_lock:
            status_message = message
        invalidate_prompt()

    def invalidate_prompt() -> None:
        app = prompt_session.app
        loop = app.loop
        if loop is None:
            return
        loop.call_soon_threadsafe(app.invalidate)

    def run_in_scrollback(render: Callable[[], None]) -> None:
        app = prompt_session.app
        loop = app.loop
        if loop is None:
            render()
            return
        loop.call_soon_threadsafe(lambda: run_in_terminal(render))

    renderer = PromptToolkitLiveRenderer(
        emit_tool=lambda text, failed: run_in_scrollback(
            lambda: print_scrollback_tool(scrollback_console, text, failed=failed)
        ),
        emit_agent=lambda text: run_in_scrollback(lambda: print_scrollback_agent(scrollback_console, text)),
        emit_error=lambda text: run_in_scrollback(lambda: print_scrollback_error(scrollback_console, text)),
        set_status=update_status,
    )

    def start_turn(prompt: str) -> Thread:
        nonlocal worker
        with state_lock:
            history = list(messages)

        def run_turn() -> None:
            try:
                result = execute_turn(agent, prompt, history, indicator=renderer)
            except RUN_ERRORS as exc:
                handle_outcome(TurnFailure(error=exc))
                return
            handle_outcome(TurnSuccess(result=result))

        thread = Thread(target=run_turn, daemon=True)
        with state_lock:
            worker = thread
        thread.start()
        return thread

    def request_exit() -> None:
        nonlocal shutdown_requested
        shutdown_requested = True
        with state_lock:
            has_pending_work = worker is not None or bool(pending_prompts)
        if has_pending_work:
            update_status("Finishing queued work before exit")

    def handle_outcome(outcome: TurnSuccess | TurnFailure) -> None:
        nonlocal messages, worker, shutdown_requested
        if isinstance(outcome, TurnFailure):
            renderer.print_error(str(outcome.error))
        else:
            with state_lock:
                messages = outcome.result.messages
            if session_store and args.session:
                session_store.save(args.session, outcome.result.messages)
            renderer.print_agent_output(outcome.result.output)
        next_prompt: str | None = None
        should_finish = False
        with state_lock:
            worker = None
            if pending_prompts:
                next_prompt = pending_prompts.pop(0)
            else:
                should_finish = shutdown_requested
        if next_prompt is not None:
            start_turn(next_prompt)
            return
        if should_finish:
            invalidate_prompt()
            return
        update_status("Ready")

    def get_bottom_toolbar() -> list[tuple[str, str]]:
        nonlocal spinner_index
        with state_lock:
            current_worker = worker
            pending_count = len(pending_prompts)
            current_status = status_message
        if current_worker is not None:
            frame = spinner_frames[spinner_index % len(spinner_frames)]
            spinner_index += 1
            queued = f" | {pending_count} queued" if pending_count else ""
            return [("class:bottom-toolbar", f" {frame} {current_status}{queued} ")]
        if pending_count:
            return [("class:bottom-toolbar", f" Ready | {pending_count} queued ")]
        return [("class:bottom-toolbar", f" {current_status} ")]

    print("pi interactive mode. Type `exit` or `quit` to leave.\n")
    while True:
        with state_lock:
            if shutdown_requested and worker is None and not pending_prompts:
                return 0
        if shutdown_requested:
            time.sleep(0.05)
            continue
        try:
            prompt = prompt_session.prompt(
                ">>> ",
                bottom_toolbar=get_bottom_toolbar,
                refresh_interval=0.1,
            )
        except (EOFError, KeyboardInterrupt):
            request_exit()
            continue
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            request_exit()
            continue
        with state_lock:
            idle = worker is None and not pending_prompts
            if not idle:
                pending_prompts.append(prompt)
                queued_position = len(pending_prompts)
            else:
                queued_position = 0
        if idle:
            start_turn(prompt)
            continue
        run_in_scrollback(lambda: print_scrollback_queue(scrollback_console, queued_position, prompt))
        invalidate_prompt()


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
