from __future__ import annotations

import inspect
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from pi.agent.loop import Agent, AgentResult, MaxIterationsExceededError
from pi.agent.models import Message
from pi.agent.providers.base import ProviderError
from pi.agent.providers.zai import ZAIConfig, ZAIProvider
from pi.agent.tools import ToolRegistry
from pi.cli.session import SessionStore

DEFAULT_SYSTEM_PROMPT = (
    "You are pi, a minimal coding agent. "
    "Use only the provided tools, do not invent unavailable capabilities, "
    "and finish with a direct assistant answer."
)

DEFAULT_MODEL = "glm-5.1"
DEFAULT_MAX_ITERATIONS = 20
SPINNER_FRAMES = ("|", "/", "-", "\\")
RUN_ERRORS = (ProviderError, MaxIterationsExceededError)


@dataclass(slots=True)
class CLIArgs:
    prompt: str | None = None
    session: str | None = None
    api_key: str = ""
    model: str = DEFAULT_MODEL
    root: str = os.getcwd()
    max_iterations: int = DEFAULT_MAX_ITERATIONS


class StatusIndicator:
    def __init__(self, stream: object) -> None:
        self._stream = stream
        self._enabled = bool(getattr(stream, "isatty", lambda: False)())
        self._label: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._width = 0

    def __enter__(self) -> "StatusIndicator":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.clear()

    def handle_event(self, event: str, payload: dict[str, object]) -> None:
        if event == "model_start":
            self.show("thinking")
            return
        if event == "tool_start":
            tool_name = str(payload.get("tool_name", "tool"))
            self.log_tool_call(tool_name, payload.get("tool_arguments"))
            self.show(f"tool {tool_name}")
            return
        if event == "tool_end" and payload.get("ok") is False:
            tool_name = str(payload.get("tool_name", "tool"))
            self.log_tool_error(tool_name, payload.get("result"))

    def show(self, label: str) -> None:
        if not self._enabled:
            return
        if self._label == label and self._thread is not None:
            return
        self._restart(label)

    def clear(self) -> None:
        if not self._enabled:
            return
        self._stop_spinner()
        if self._width:
            self._write("\r" + (" " * self._width) + "\r")

    def log_tool_call(self, tool_name: str, raw_arguments: object) -> None:
        if not self._enabled:
            return
        preview = format_tool_preview(tool_name, raw_arguments)
        self._print_line(preview)

    def log_tool_error(self, tool_name: str, result: object) -> None:
        if not self._enabled:
            return
        if not isinstance(result, dict):
            self._print_line(f"tool {tool_name} failed")
            return
        error = result.get("error")
        if isinstance(error, str) and error.strip():
            self._print_line(f"tool {tool_name} failed: {truncate_cli_text(error, 80)}")
        else:
            self._print_line(f"tool {tool_name} failed")

    def _restart(self, label: str) -> None:
        self._stop_spinner()
        self._label = label
        self._width = len(label) + 2
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, args=(label, self._stop), daemon=True)
        self._thread.start()

    def _stop_spinner(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _spin(self, label: str, stop: threading.Event) -> None:
        index = 0
        while not stop.is_set():
            frame = SPINNER_FRAMES[index % len(SPINNER_FRAMES)]
            self._write(f"\r{label} {frame}")
            index += 1
            stop.wait(0.1)

    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()

    def _print_line(self, text: str) -> None:
        self.clear()
        self._write(f"{text}\n")


def format_tool_preview(tool_name: str, raw_arguments: object) -> str:
    arguments = parse_tool_arguments(raw_arguments)
    preview = None
    if tool_name in {"read", "write", "edit", "ls", "find", "grep"}:
        preview = first_string(arguments, "path")
    elif tool_name == "bash":
        preview = first_string(arguments, "command")

    if preview:
        return f"tool {tool_name} {truncate_cli_text(preview, 72)}"
    return f"tool {tool_name}"


def parse_tool_arguments(raw_arguments: object) -> dict[str, object]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str):
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def first_string(arguments: dict[str, object], key: str) -> str | None:
    value = arguments.get(key)
    return value if isinstance(value, str) and value.strip() else None


def truncate_cli_text(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def build_agent_from_args(args: CLIArgs) -> Agent:
    if not args.api_key:
        raise ValueError("A ZAI API key is required. Pass --api-key or set ZAI_API_KEY.")

    provider = ZAIProvider(
        config=ZAIConfig(
            api_key=args.api_key,
            model=args.model,
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
) -> AgentResult:
    stream = stderr or sys.stderr
    with StatusIndicator(stream) as indicator:
        return _run_agent_turn(agent, prompt, messages, indicator.handle_event)


def _run_agent_turn(
    agent: Agent,
    prompt: str,
    messages: list[Message],
    on_event: object,
) -> AgentResult:
    run_signature = inspect.signature(agent.run)
    if "on_event" in run_signature.parameters:
        return agent.run(prompt, messages=messages, on_event=on_event)
    return agent.run(prompt, messages=messages)


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

    while True:
        try:
            prompt = input_func(">>> ").strip()
        except EOFError:
            print()
            return 0

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return 0

        try:
            result = execute_turn(active_agent, prompt, session_messages, stderr=stream)
        except RUN_ERRORS as exc:
            print(f"Error: {exc}", file=stream)
            continue

        session_messages = result.messages
        if session_store and args.session:
            session_store.save(args.session, session_messages)
        print(result.output)


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
