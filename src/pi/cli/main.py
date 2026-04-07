from __future__ import annotations

from collections.abc import Callable
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from queue import Empty, Queue
from threading import Thread, get_ident
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
        emit_line: Callable[[str], None],
        set_status: Callable[[str], None],
    ) -> None:
        self._emit_line = emit_line
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
            self._emit_line(f"tool {format_tool_preview(str(tool_name), tool_arguments)}")
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
                self._emit_line(f"tool! {error_text or 'The tool returned an error.'}")
                self._set_status("Handling tool failure")
                return
            self._set_status("Thinking")

    def print_agent_output(self, text: str) -> None:
        lines = text.rstrip().splitlines() or ["(empty)"]
        self._emit_line(f"pi {lines[0]}")
        for line in lines[1:]:
            self._emit_line(f"   {line}")

    def print_error(self, message: str) -> None:
        self._emit_line(f"error {message}")


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
    from prompt_toolkit.application import Application
    from prompt_toolkit.document import Document
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import TextArea

    messages = list(session_messages)
    pending_prompts: list[str] = []
    worker: Thread | None = None
    shutdown_requested = False
    status_message = "Ready"
    spinner_index = 0
    spinner_frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    transcript_lines = ["pi interactive mode. Type `exit` or `quit` to leave.", ""]
    ui_thread_id = get_ident()
    app: Application[int]

    transcript = TextArea(
        text="",
        read_only=True,
        focusable=False,
        scrollbar=True,
        wrap_lines=True,
        style="class:transcript",
    )
    input_field = TextArea(
        height=1,
        prompt=">>> ",
        multiline=False,
        wrap_lines=False,
        style="class:input",
    )

    def sync_transcript() -> None:
        text = "\n".join(transcript_lines).rstrip() + "\n"
        transcript.buffer.set_document(Document(text, cursor_position=len(text)), bypass_readonly=True)

    def dispatch_ui(callback: Callable[[], None]) -> None:
        loop = app.loop
        if get_ident() == ui_thread_id or loop is None:
            callback()
            return
        loop.call_soon_threadsafe(callback)

    def append_line(text: str) -> None:
        def update() -> None:
            transcript_lines.append(text)
            sync_transcript()
            app.invalidate()

        dispatch_ui(update)

    def update_status(message: str) -> None:
        nonlocal status_message

        def update() -> None:
            nonlocal status_message
            status_message = message
            app.invalidate()

        dispatch_ui(update)

    renderer = PromptToolkitLiveRenderer(
        emit_line=append_line,
        set_status=update_status,
    )

    def start_turn(prompt: str) -> Thread:
        history = list(messages)

        def run_turn() -> None:
            try:
                result = execute_turn(agent, prompt, history, indicator=renderer)
            except RUN_ERRORS as exc:
                dispatch_ui(lambda: handle_outcome(TurnFailure(error=exc)))
                return
            dispatch_ui(lambda: handle_outcome(TurnSuccess(result=result)))

        thread = Thread(target=run_turn, daemon=True)
        thread.start()
        return thread

    def request_exit() -> None:
        nonlocal shutdown_requested
        shutdown_requested = True
        if worker is None and not pending_prompts:
            app.exit(result=0)
            return
        update_status("Finishing queued work before exit")

    def submit_prompt() -> None:
        nonlocal worker
        prompt = input_field.text.strip()
        input_field.buffer.set_document(Document("", cursor_position=0), bypass_readonly=True)
        if not prompt:
            return
        if prompt.lower() in {"exit", "quit"}:
            request_exit()
            return
        append_line(f">>> {prompt}")
        if worker is None and not pending_prompts:
            worker = start_turn(prompt)
            return
        pending_prompts.append(prompt)
        app.invalidate()

    def handle_outcome(outcome: TurnSuccess | TurnFailure) -> None:
        nonlocal messages, worker, shutdown_requested
        worker = None
        if isinstance(outcome, TurnFailure):
            renderer.print_error(str(outcome.error))
        else:
            messages = outcome.result.messages
            if session_store and args.session:
                session_store.save(args.session, messages)
            renderer.print_agent_output(outcome.result.output)
        if pending_prompts:
            next_prompt = pending_prompts.pop(0)
            worker = start_turn(next_prompt)
        elif shutdown_requested:
            app.exit(result=0)
            return
        else:
            update_status("Ready")
        app.invalidate()

    def get_status_fragments() -> list[tuple[str, str]]:
        nonlocal spinner_index
        if worker is not None:
            frame = spinner_frames[spinner_index % len(spinner_frames)]
            spinner_index += 1
            queued = f" | {len(pending_prompts)} queued" if pending_prompts else ""
            return [("class:status.busy", f" {frame} {status_message}{queued} ")]
        if pending_prompts:
            return [("class:status.queued", f" queued | {len(pending_prompts)} waiting ")]
        return [("class:status.idle", f" {status_message} ")]

    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event) -> None:
        submit_prompt()

    @kb.add("c-c")
    @kb.add("c-d")
    def _exit(event) -> None:
        request_exit()

    app = Application(
        layout=Layout(
            HSplit(
                [
                    transcript,
                    Window(height=1, content=FormattedTextControl(get_status_fragments)),
                    input_field,
                ]
            ),
            focused_element=input_field,
        ),
        key_bindings=kb,
        full_screen=False,
        mouse_support=False,
        refresh_interval=0.1,
        style=Style.from_dict(
            {
                "transcript": "fg:#d8dee9 bg:#2b303b",
                "input": "fg:#eceff4 bg:#2b303b",
                "status.busy": "fg:#eceff4 bg:#4c566a bold",
                "status.queued": "fg:#2e3440 bg:#ebcb8b bold",
                "status.idle": "fg:#2e3440 bg:#a3be8c bold",
            }
        ),
    )

    sync_transcript()
    return app.run()


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
