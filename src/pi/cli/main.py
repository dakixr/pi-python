from __future__ import annotations

from collections import deque
import inspect
import os
import select
import sys
import termios
import threading
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from pi.agent.loop import Agent, AgentResult, MaxIterationsExceededError
from pi.agent.models import Message
from pi.agent.providers.base import ProviderError
from pi.agent.providers.zai import ZAIConfig, ZAIProvider
from pi.agent.tools import ToolRegistry
from pi.cli.render import InteractiveRenderer, StatusIndicator, format_user_separator
from pi.cli.session import SessionStore

DEFAULT_SYSTEM_PROMPT = (
    "You are pi, a minimal coding agent. "
    "Use only the provided tools, do not invent unavailable capabilities, "
    "and finish with a direct assistant answer."
)

DEFAULT_MODEL = "glm-5.1"
DEFAULT_MAX_ITERATIONS = 20
RUN_ERRORS = (ProviderError, MaxIterationsExceededError)


@dataclass(slots=True)
class CLIArgs:
    prompt: str | None = None
    session: str | None = None
    api_key: str = ""
    model: str = DEFAULT_MODEL
    root: str = os.getcwd()
    max_iterations: int = DEFAULT_MAX_ITERATIONS


@dataclass(slots=True)
class QueuedPrompt:
    text: str


def build_agent_from_args(args: CLIArgs) -> Agent:
    if not args.api_key:
        raise ValueError("A ZAI API key is required. Pass --api-key or set ZAI_API_KEY.")

    provider = ZAIProvider(
        config=ZAIConfig(
            api_key=args.api_key,
            model=args.model,
            debug_log_path=str(Path(args.root) / ".pi" / "logs" / "zai-debug.jsonl"),
        )
    )
    return Agent(
        provider=provider,
        tools=ToolRegistry(root=Path(args.root)),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_iterations=args.max_iterations,
    )


def execute_turn(
    agent: Agent,
    prompt: str,
    messages: list[Message],
    *,
    stderr: object | None = None,
    indicator: StatusIndicator | None = None,
) -> AgentResult:
    if indicator is not None:
        return _run_agent_turn(agent, prompt, messages, indicator.handle_event)

    stream = stderr or sys.stderr
    with StatusIndicator(stream) as scoped_indicator:
        return _run_agent_turn(agent, prompt, messages, scoped_indicator.handle_event)


def _run_agent_turn(
    agent: Agent,
    prompt: str,
    messages: list[Message],
    on_event: object,
) -> AgentResult:
    run_signature = inspect.signature(agent.run)
    if callable(on_event):
        on_event("model_start", {"iteration": 1})
    if "on_event" in run_signature.parameters:
        return agent.run(prompt, messages=messages, on_event=on_event)
    return agent.run(prompt, messages=messages)


def run_interactive_cli(
    args: CLIArgs,
    agent: Agent,
    session_messages: list[Message],
    *,
    session_store: SessionStore | None,
    input_func: object,
    stderr: object,
) -> int:
    queue: deque[QueuedPrompt] = deque()
    queue_ready = threading.Event()
    queue_lock = threading.Lock()
    stop_accepting = threading.Event()
    worker_finished = threading.Event()
    messages_state = {"messages": session_messages}
    busy_state = {"active": False}

    with StatusIndicator(stderr, animate=False) as indicator:
        def worker() -> None:
            while True:
                queue_ready.wait()
                while True:
                    with queue_lock:
                        if not queue:
                            queue_ready.clear()
                            busy_state["active"] = False
                            indicator.set_queue_count(0)
                            if stop_accepting.is_set():
                                worker_finished.set()
                                return
                            break
                        queued_prompt = queue.popleft()
                        busy_state["active"] = True
                        indicator.set_queue_count(len(queue))

                    print(format_user_separator(queued_prompt.text))
                    try:
                        result = execute_turn(
                            agent,
                            queued_prompt.text,
                            messages_state["messages"],
                            stderr=stderr,
                            indicator=indicator,
                        )
                    except RUN_ERRORS as exc:
                        indicator.clear()
                        dropped_count = 0
                        if isinstance(exc, ProviderError):
                            with queue_lock:
                                dropped_count = len(queue)
                                queue.clear()
                                queue_ready.clear()
                                indicator.set_queue_count(0)
                        print(f"Error: {exc}", file=stderr)
                        if dropped_count:
                            suffix = "message" if dropped_count == 1 else "messages"
                            print(
                                f"Dropped {dropped_count} queued {suffix} after turn error.",
                                file=stderr,
                            )
                    else:
                        indicator.clear()
                        messages_state["messages"] = result.messages
                        if session_store and args.session:
                            session_store.save(args.session, messages_state["messages"])
                        print(result.output)

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        while True:
            try:
                prompt = input_func(">>> ").strip()
            except EOFError:
                print()
                stop_accepting.set()
                queue_ready.set()
                worker_finished.wait()
                return 0

            if not prompt:
                continue

            normalized = prompt.lower()
            if normalized in {"exit", "quit"}:
                stop_accepting.set()
                with queue_lock:
                    has_pending_work = busy_state["active"] or bool(queue)
                if has_pending_work:
                    queue_ready.set()
                    worker_finished.wait()
                return 0

            with queue_lock:
                queued_behind_active_turn = busy_state["active"] or bool(queue)
                queue.append(QueuedPrompt(text=prompt))
                pending_count = len(queue) if busy_state["active"] else max(len(queue) - 1, 0)

            if queued_behind_active_turn:
                indicator.log_queued_message(prompt)
            indicator.set_queue_count(pending_count)
            queue_ready.set()


def run_interactive_tty_cli(
    args: CLIArgs,
    agent: Agent,
    session_messages: list[Message],
    *,
    session_store: SessionStore | None,
) -> int:
    queue: deque[QueuedPrompt] = deque()
    queue_ready = threading.Event()
    queue_lock = threading.Lock()
    stop_accepting = threading.Event()
    worker_finished = threading.Event()
    messages_state = {"messages": session_messages}
    busy_state = {"active": False}

    stdin = sys.stdin
    stdout = sys.stdout
    fd = stdin.fileno()
    original_termios = termios.tcgetattr(fd)

    def pending_messages() -> list[str]:
        if busy_state["active"]:
            return [item.text for item in queue]
        return [item.text for item in list(queue)[1:]]

    try:
        tty.setcbreak(fd)
        with InteractiveRenderer(stdout) as renderer:
            def worker() -> None:
                while True:
                    queue_ready.wait()
                    while True:
                        with queue_lock:
                            if not queue:
                                queue_ready.clear()
                                busy_state["active"] = False
                                renderer.set_status(None)
                                renderer.set_activity(None)
                                renderer.set_queue_messages([])
                                if stop_accepting.is_set():
                                    worker_finished.set()
                                    return
                                break
                            queued_prompt = queue.popleft()
                            busy_state["active"] = True
                            renderer.set_queue_messages([item.text for item in queue])

                        renderer.print_message(format_user_separator(queued_prompt.text))
                        try:
                            result = execute_turn(
                                agent,
                                queued_prompt.text,
                                messages_state["messages"],
                                indicator=renderer,
                            )
                        except RUN_ERRORS as exc:
                            renderer.clear_turn_state()
                            dropped_count = 0
                            if isinstance(exc, ProviderError):
                                with queue_lock:
                                    dropped_count = len(queue)
                                    queue.clear()
                                    queue_ready.clear()
                                    renderer.set_queue_messages([])
                            renderer.print_message(f"Error: {exc}")
                            if dropped_count:
                                suffix = "message" if dropped_count == 1 else "messages"
                                renderer.print_message(
                                    f"Dropped {dropped_count} queued {suffix} after turn error."
                                )
                        else:
                            renderer.clear_turn_state()
                            messages_state["messages"] = result.messages
                            if session_store and args.session:
                                session_store.save(args.session, messages_state["messages"])
                            renderer.print_message(result.output)

            worker_thread = threading.Thread(target=worker, daemon=True)
            worker_thread.start()

            buffer = ""
            renderer.set_input_buffer(buffer)
            while True:
                ready, _, _ = select.select([stdin], [], [], 0.05)
                if not ready:
                    if stop_accepting.is_set() and not busy_state["active"] and not queue:
                        worker_finished.wait()
                        return 0
                    continue

                chunk = os.read(fd, 32).decode("utf-8", errors="ignore")
                if not chunk:
                    stop_accepting.set()
                    queue_ready.set()
                    worker_finished.wait()
                    return 0

                for char in chunk:
                    if char == "\x03":
                        stop_accepting.set()
                        queue_ready.set()
                        worker_finished.wait()
                        renderer.print_message("")
                        return 130
                    if char == "\x04":
                        if buffer:
                            continue
                        stop_accepting.set()
                        queue_ready.set()
                        worker_finished.wait()
                        renderer.print_message("")
                        return 0
                    if char in {"\r", "\n"}:
                        prompt = buffer.strip()
                        buffer = ""
                        renderer.set_input_buffer(buffer)
                        if not prompt:
                            continue
                        if prompt.lower() in {"exit", "quit"}:
                            stop_accepting.set()
                            with queue_lock:
                                has_pending_work = busy_state["active"] or bool(queue)
                            if has_pending_work:
                                queue_ready.set()
                                worker_finished.wait()
                            return 0
                        with queue_lock:
                            queue.append(QueuedPrompt(text=prompt))
                            renderer.set_queue_messages(pending_messages())
                        queue_ready.set()
                        continue
                    if char in {"\x7f", "\b"}:
                        buffer = buffer[:-1]
                        renderer.set_input_buffer(buffer)
                        continue
                    if char == "\x1b":
                        continue
                    if char.isprintable():
                        buffer += char
                        renderer.set_input_buffer(buffer)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_termios)


def run_cli(
    args: CLIArgs,
    agent: Agent | None = None,
    *,
    input_func: object = input,
    stderr: object | None = None,
) -> int:
    stream = stderr or sys.stderr
    try:
        active_agent = agent or build_agent_from_args(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=stream)
        return 1

    session_store = SessionStore(root=Path(args.root)) if args.session else None
    session_messages = (
        session_store.load(args.session).messages if session_store and args.session else []
    )

    if args.prompt:
        try:
            result = execute_turn(active_agent, args.prompt, session_messages, stderr=stream)
        except RUN_ERRORS as exc:
            print(f"Error: {exc}", file=stream)
            return 1

        if session_store and args.session:
            session_store.save(args.session, result.messages)
        print(result.output)
        return 0

    if (
        input_func is input
        and bool(getattr(sys.stdin, "isatty", lambda: False)())
        and bool(getattr(sys.stdout, "isatty", lambda: False)())
    ):
        return run_interactive_tty_cli(
            args,
            active_agent,
            session_messages,
            session_store=session_store,
        )

    return run_interactive_cli(
        args,
        active_agent,
        session_messages,
        session_store=session_store,
        input_func=input_func,
        stderr=stream,
    )


app = typer.Typer(add_completion=False, help="Minimal coding agent CLI.")


@app.command()
def cli(
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", help="Run one prompt and exit."),
    ] = None,
    session: Annotated[
        str | None,
        typer.Option(
            "--session",
            help="Load and persist conversation state under .pi/sessions/<name>.json.",
        ),
    ] = None,
    api_key: Annotated[
        str,
        typer.Option("--api-key", envvar="ZAI_API_KEY", help="Z.AI API key."),
    ] = "",
    model: Annotated[
        str,
        typer.Option("--model", help="Model name to send to the provider."),
    ] = DEFAULT_MODEL,
    root: Annotated[
        Path,
        typer.Option(
            "--root",
            help="Workspace root for tools.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    max_iterations: Annotated[
        int,
        typer.Option(
            "--max-iterations",
            min=1,
            help="Maximum number of agent iterations per prompt.",
        ),
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
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
