from __future__ import annotations

import json
from threading import Lock
from typing import Protocol, TextIO, cast

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.text import Text


def truncate_cli_text(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


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


def format_tool_preview(tool_name: str, raw_arguments: object) -> str:
    arguments = parse_tool_arguments(raw_arguments)
    preview = None
    if tool_name in {"read", "write", "edit", "ls", "find", "grep"}:
        preview = first_string(arguments, "path")
    elif tool_name == "bash":
        preview = first_string(arguments, "command")
    if preview:
        return f"{tool_name} {truncate_cli_text(preview, 72)}"
    return tool_name


def format_user_separator(prompt: str) -> str:
    return f"---\nuser:\n{prompt.rstrip()}\n---"


class OutputStream(Protocol):
    def write(self, text: str, /) -> object: ...
    def flush(self) -> object: ...
    def isatty(self) -> bool: ...


def build_console(stream: OutputStream) -> Console:
    is_tty = stream.isatty()
    return Console(
        file=cast(TextIO, stream),
        force_terminal=is_tty,
        color_system="auto" if is_tty else None,
        soft_wrap=True,
        highlight=False,
    )


def print_user_prompt(console: Console, prompt: str) -> None:
    if console.is_terminal:
        panel = Panel(
            Text(prompt.rstrip(), style="bold white"),
            title="[bold cyan]User[/bold cyan]",
            border_style="cyan",
            expand=True,
        )
        console.print(panel)
        return
    console.print(format_user_separator(prompt))


def print_agent_output(console: Console, output: str) -> None:
    if console.is_terminal:
        panel = Panel(
            Text(output.rstrip() or "(empty)", style="white"),
            title="[bold green]Pi[/bold green]",
            border_style="green",
            expand=True,
        )
        console.print(panel)
        return
    console.print(output)


def print_error(console: Console, message: str) -> None:
    if console.is_terminal:
        console.print(f"[bold red]Error:[/bold red] {message}")
        return
    console.print(f"Error: {message}")


class StatusIndicator:
    def __init__(self, stream: OutputStream, *, animate: bool = True) -> None:
        self._stream = stream
        self._animate = animate
        self._console = build_console(stream)
        self._status: Status | None = None
        self._last_message = "Thinking"
        self._lock = Lock()

    def __enter__(self) -> "StatusIndicator":
        with self._lock:
            if self._animate and self._console.is_terminal:
                self._status = self._console.status("[bold cyan]Thinking[/bold cyan]", spinner="dots")
                self._status.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.clear()
        return None

    def handle_event(self, event: str, payload: dict[str, object]) -> None:
        if event == "model_start":
            self._update("Thinking")
            return
        if event == "model_end":
            self._update("Planning next step")
            return
        if event == "tool_execution_start":
            tool_name = payload.get("tool_name", "tool")
            tool_arguments = payload.get("tool_arguments")
            self._log_event_line("Tool", format_tool_preview(str(tool_name), tool_arguments), style="cyan")
            self._update("Waiting on tool result")
            return
        if event == "tool_execution_end":
            ok = payload.get("ok", False)
            self._update("Thinking" if ok else "Handling tool failure")

    def clear(self) -> None:
        with self._lock:
            if self._status is not None:
                self._status.stop()
                self._status = None

    def set_queue_count(self, count: int) -> None:
        if count > 0:
            self._update(f"{self._last_message} [{count} queued]")
            return
        self._update(self._last_message.split(" [", 1)[0])

    def log_queued_message(self, text: str) -> None:
        self._log_event_line("Queued", truncate_cli_text(text, 72), style="magenta")

    def _update(self, message: str) -> None:
        with self._lock:
            self._last_message = message
            if self._status is not None:
                self._status.update(f"[bold cyan]{message}[/bold cyan]")

    def _log_event_line(self, label: str, text: str, *, style: str) -> None:
        with self._lock:
            if not self._console.is_terminal:
                return
            had_status = self._status is not None
            if had_status:
                assert self._status is not None
                self._status.stop()
            self._console.print(f"[bold {style}]{label}[/bold {style}] {text}")
            if had_status and self._animate:
                self._status = self._console.status(f"[bold cyan]{self._last_message}[/bold cyan]", spinner="dots")
                self._status.start()


class InteractiveRenderer(StatusIndicator):
    def print_message(self, text: str) -> None:
        print_agent_output(self._console, text)
