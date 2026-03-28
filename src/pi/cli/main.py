from __future__ import annotations

from collections import deque
import inspect
import json
import os
import shutil
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
ANSI_SAVE_CURSOR = "\x1b[s"
ANSI_RESTORE_CURSOR = "\x1b[u"
ANSI_CURSOR_HOME = "\x1b[H"
ANSI_CLEAR_LINE = "\x1b[2K"
ANSI_CURSOR_UP_FMT = "\x1b[{}A"
PANEL_MAX_LINES = 4
PANEL_MAX_QUEUE_LINES = 2


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
    show_separator: bool = False


class StatusIndicator:
    def __init__(
        self,
        stream: object,
        *,
        animate: bool = True,
        panel_mode: bool = False,
    ) -> None:
        self._stream = stream
        self._enabled = bool(getattr(stream, "isatty", lambda: False)())
        self._animate = animate
        self._panel_mode = panel_mode and self._enabled
        self._label: str | None = None
        self._base_label: str | None = None
        self._queue_count = 0
        self._queue_previews: list[str] = []
        self._activity_line: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._width = 0
        self._write_lock = threading.Lock()
        self._frame_index = 0

    def __enter__(self) -> "StatusIndicator":
        if self._panel_mode:
            self._reserve_panel_space()
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
        if event == "tool_end":
            tool_name = str(payload.get("tool_name", "tool"))
            result = payload.get("result")
            if payload.get("ok") is False:
                self.log_tool_error(tool_name, result)
                return
            self.log_tool_result(tool_name, result)

    def show(self, label: str) -> None:
        if not self._enabled:
            return
        self._base_label = label
        if self._panel_mode:
            self._ensure_spinner()
            self._render_panel()
            return
        if not self._animate:
            return
        display_label = self._compose_label()
        if self._label == display_label and self._thread is not None:
            return
        self._restart(display_label)

    def clear(self) -> None:
        if not self._enabled:
            return
        self._stop_spinner()
        if self._panel_mode:
            self._label = None
            self._base_label = None
            self._activity_line = None
            self._queue_count = 0
            self._queue_previews = []
            self._render_panel()
            return
        if self._width:
            self._write("\r" + (" " * self._width) + "\r")
        self._label = None
        self._base_label = None

    def set_queue_count(self, count: int) -> None:
        if not self._enabled:
            return
        normalized = max(count, 0)
        if self._queue_count == normalized:
            return
        self._queue_count = normalized
        if self._panel_mode:
            self._render_panel()
            return
        if self._animate and self._base_label is not None:
            self._restart(self._compose_label())

    def set_queue_messages(self, messages: list[str]) -> None:
        if not self._enabled:
            return
        self._queue_count = len(messages)
        self._queue_previews = [
            truncate_cli_text(message, 72) for message in messages[-PANEL_MAX_QUEUE_LINES:]
        ]
        if self._panel_mode:
            self._render_panel()

    def log_tool_call(self, tool_name: str, raw_arguments: object) -> None:
        preview = format_tool_preview(tool_name, raw_arguments)
        if self._panel_mode:
            self._activity_line = preview
            self._render_panel()
            return
        if not self._enabled:
            return
        self._print_line(preview)

    def log_queued_message(self, text: str) -> None:
        if self._panel_mode:
            return
        if not self._enabled:
            return
        self._print_line(f"queued {truncate_cli_text(text, 72)}")

    def log_tool_error(self, tool_name: str, result: object) -> None:
        if self._panel_mode:
            self._activity_line = None
            if self._base_label and self._base_label.startswith("tool "):
                self._base_label = "thinking"
            self._render_panel()
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

    def log_tool_result(self, tool_name: str, result: object) -> None:
        if self._panel_mode:
            self._activity_line = None
            if self._base_label and self._base_label.startswith("tool "):
                self._base_label = "thinking"
            self._render_panel()
            return
        if not self._enabled:
            return
        rendered = format_tool_result(tool_name, result)
        if rendered:
            self._print_line(rendered)

    def _restart(self, label: str) -> None:
        self._stop_spinner()
        self._label = label
        self._width = len(label) + 2
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, args=(label, self._stop), daemon=True)
        self._thread.start()

    def _ensure_spinner(self) -> None:
        if not self._animate or self._thread is not None:
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin_panel, args=(self._stop,), daemon=True)
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

    def _spin_panel(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self._frame_index = (self._frame_index + 1) % len(SPINNER_FRAMES)
            self._render_panel()
            stop.wait(0.1)

    def _write(self, text: str) -> None:
        with self._write_lock:
            self._stream.write(text)
            self._stream.flush()

    def _print_line(self, text: str) -> None:
        self._stop_spinner()
        if self._width:
            self._write("\r" + (" " * self._width) + "\r")
        self._write(f"{text}\n")
        if self._animate and self._base_label is not None:
            self._restart(self._compose_label())

    def _compose_label(self) -> str:
        if self._base_label is None:
            return ""
        if self._queue_count <= 0:
            return self._base_label
        suffix = "message" if self._queue_count == 1 else "messages"
        return f"{self._base_label} [{self._queue_count} queued {suffix}]"

    def _reserve_panel_space(self) -> None:
        self._write("\n" * PANEL_MAX_LINES)

    def _render_panel(self) -> None:
        if not self._panel_mode:
            return
        lines = self._compose_panel_lines()
        with self._write_lock:
            self._stream.write(ANSI_SAVE_CURSOR)
            self._stream.write(ANSI_CURSOR_HOME)
            for index in range(PANEL_MAX_LINES):
                if index:
                    self._stream.write("\n")
                self._stream.write(ANSI_CLEAR_LINE)
                if index < len(lines):
                    self._stream.write(lines[index])
            self._stream.write(ANSI_RESTORE_CURSOR)
            self._stream.flush()

    def _compose_panel_lines(self) -> list[str]:
        if self._base_label is None and self._activity_line is None and self._queue_count == 0:
            return []

        lines: list[str] = []
        if self._base_label is not None:
            status = self._base_label
            if self._animate:
                status = f"{status} {SPINNER_FRAMES[self._frame_index]}"
            if self._queue_count > 0:
                suffix = "message" if self._queue_count == 1 else "messages"
                status = f"{status} [{self._queue_count} queued {suffix}]"
            lines.append(status)

        if self._activity_line and len(lines) < PANEL_MAX_LINES:
            lines.append(self._activity_line)

        for preview in self._queue_previews:
            if len(lines) >= PANEL_MAX_LINES:
                break
            lines.append(f"queued {preview}")

        hidden_count = self._queue_count - len(self._queue_previews)
        if hidden_count > 0 and len(lines) < PANEL_MAX_LINES:
            suffix = "message" if hidden_count == 1 else "messages"
            lines.append(f"... {hidden_count} more queued {suffix}")

        return [
            truncate_cli_text(line, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
            for line in lines[:PANEL_MAX_LINES]
        ]


class InteractiveRenderer:
    def __init__(self, stream: object) -> None:
        self._stream = stream
        self._status: str | None = None
        self._activity_line: str | None = None
        self._last_tool_line: str | None = None
        self._queue_count = 0
        self._queue_previews: list[str] = []
        self._input_buffer = ""
        self._footer_height = 0
        self._last_rendered_lines: tuple[str, ...] = ()
        self._lock = threading.Lock()
        self._frame_index = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "InteractiveRenderer":
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        with self._lock:
            self._redraw_locked()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            self._clear_footer_locked()
            self._stream.flush()

    def handle_event(self, event: str, payload: dict[str, object]) -> None:
        if event == "model_start":
            self.set_status("thinking")
            return
        if event == "tool_start":
            tool_name = str(payload.get("tool_name", "tool"))
            preview = format_tool_preview(tool_name, payload.get("tool_arguments"))
            with self._lock:
                self._status = f"tool {tool_name}"
                self._activity_line = preview
                self._last_tool_line = preview
                self._print_message_locked(preview)
            return
        if event == "tool_end":
            tool_name = str(payload.get("tool_name", "tool"))
            if payload.get("ok") is False:
                result = payload.get("result")
                with self._lock:
                    self._status = "thinking"
                    self._activity_line = None
                    if isinstance(result, dict):
                        error = result.get("error")
                        if isinstance(error, str) and error.strip():
                            rendered_error = (
                                f"tool {tool_name} failed: {truncate_cli_text(error, 80)}"
                            )
                            self._last_tool_line = rendered_error
                            self._print_message_locked(rendered_error)
                            return
                    self._redraw_locked()
                return
            with self._lock:
                self._status = "thinking"
                self._activity_line = None
                self._redraw_locked()

    def set_status(self, label: str | None) -> None:
        with self._lock:
            self._status = label
            if label == "thinking":
                self._frame_index = 0
            self._redraw_locked()

    def set_activity(self, text: str | None) -> None:
        with self._lock:
            self._activity_line = text
            self._redraw_locked()

    def set_last_tool_line(self, text: str | None) -> None:
        with self._lock:
            self._last_tool_line = text
            self._redraw_locked()

    def set_queue_messages(self, messages: list[str]) -> None:
        with self._lock:
            self._queue_count = len(messages)
            self._queue_previews = [
                truncate_cli_text(message, 72) for message in messages[-PANEL_MAX_QUEUE_LINES:]
            ]
            self._redraw_locked()

    def set_input_buffer(self, text: str) -> None:
        with self._lock:
            self._input_buffer = text
            self._redraw_locked()

    def print_message(self, text: str) -> None:
        with self._lock:
            self._print_message_locked(text)

    def _print_message_locked(self, text: str) -> None:
        self._clear_footer_locked()
        self._stream.write(f"{text}\n")
        self._redraw_locked()

    def _spin(self) -> None:
        while not self._stop.wait(0.12):
            with self._lock:
                if self._status != "thinking":
                    continue
                self._frame_index = (self._frame_index + 1) % len(SPINNER_FRAMES)
                self._redraw_locked()

    def _compose_footer_lines(self) -> list[str]:
        width = max(shutil.get_terminal_size(fallback=(80, 24)).columns - 1, 20)
        lines: list[str] = []
        detail = self._compose_detail_line(width)
        status = self._compose_status_line(width)
        if detail is not None and detail.startswith("tool "):
            lines.append(detail)
            if status is not None:
                lines.append(status)
        else:
            if status is not None:
                lines.append(status)
            if detail is not None:
                lines.append(detail)
        lines.append(truncate_cli_text(f">>> {self._input_buffer}", width))
        return lines

    def _compose_status_line(self, width: int) -> str | None:
        if self._status is None:
            return None
        status = self._status
        if self._status == "thinking":
            status = f"{status} {SPINNER_FRAMES[self._frame_index]}"
        if self._queue_count > 0:
            suffix = "message" if self._queue_count == 1 else "messages"
            status = f"{status} [{self._queue_count} queued {suffix}]"
        return truncate_cli_text(status, width)

    def _compose_detail_line(self, width: int) -> str | None:
        if self._queue_count > 0:
            previews = self._queue_previews[-PANEL_MAX_QUEUE_LINES:]
            preview_text = " | ".join(previews)
            hidden_count = self._queue_count - len(previews)
            if hidden_count > 0:
                suffix = "message" if hidden_count == 1 else "messages"
                preview_text = f"{preview_text} | +{hidden_count} {suffix}"
            return truncate_cli_text(f"queued {preview_text}", width)

        if self._activity_line is not None:
            return truncate_cli_text(self._activity_line, width)

        if self._last_tool_line is not None:
            return truncate_cli_text(self._last_tool_line, width)

        return None

    def _redraw_locked(self) -> None:
        lines = self._compose_footer_lines()
        if tuple(lines) == self._last_rendered_lines:
            return
        self._clear_footer_locked()
        for index, line in enumerate(lines):
            self._stream.write(ANSI_CLEAR_LINE)
            self._stream.write(line)
            if index < len(lines) - 1:
                self._stream.write("\n")
        self._stream.flush()
        self._footer_height = len(lines)
        self._last_rendered_lines = tuple(lines)

    def _clear_footer_locked(self) -> None:
        if self._footer_height <= 0:
            self._last_rendered_lines = ()
            return
        self._stream.write("\r")
        if self._footer_height > 1:
            self._stream.write(ANSI_CURSOR_UP_FMT.format(self._footer_height - 1))
        for index in range(self._footer_height):
            self._stream.write(ANSI_CLEAR_LINE)
            if index < self._footer_height - 1:
                self._stream.write("\n")
        if self._footer_height > 1:
            self._stream.write(ANSI_CURSOR_UP_FMT.format(self._footer_height - 1))
        self._stream.write("\r")
        self._footer_height = 0
        self._last_rendered_lines = ()


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


def format_user_separator(prompt: str) -> str:
    return f"user:\n{truncate_cli_text(prompt, 72)}\n---"


def format_tool_result(tool_name: str, result: object) -> str | None:
    if not isinstance(result, dict):
        return None

    if tool_name not in {"edit", "write"}:
        return None

    diff = result.get("diff")
    if not isinstance(diff, str) or not diff.strip():
        return None

    lines = diff.splitlines()
    if result.get("diff_truncated") is True:
        lines.append("... diff truncated ...")
    return "\n".join(lines)


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

                    if queued_prompt.show_separator:
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
                queue.append(QueuedPrompt(text=prompt, show_separator=queued_behind_active_turn))
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
                            renderer.set_status(None)
                            renderer.set_activity(None)
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
                            renderer.set_status(None)
                            renderer.set_activity(None)
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
                            queued_behind_active_turn = busy_state["active"] or bool(queue)
                            queue.append(
                                QueuedPrompt(
                                    text=prompt,
                                    show_separator=queued_behind_active_turn,
                                )
                            )
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
