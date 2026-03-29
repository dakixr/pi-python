from __future__ import annotations

import json
import shutil
import threading

ANSI_CLEAR_LINE = "\x1b[2K"
ANSI_CURSOR_UP_FMT = "\x1b[{}A"
PANEL_MAX_QUEUE_LINES = 2
SPINNER_FRAMES = ("|", "/", "-", "\\")


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
        return f"tool {tool_name} {truncate_cli_text(preview, 72)}"
    return f"tool {tool_name}"


def format_user_separator(prompt: str) -> str:
    return f"---\nuser:\n{prompt.rstrip()}\n---"


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


class StatusIndicator:
    def __init__(
        self,
        stream: object,
        *,
        animate: bool = True,
    ) -> None:
        self._stream = stream
        self._enabled = bool(getattr(stream, "isatty", lambda: False)())
        self._animate = animate
        self._label: str | None = None
        self._base_label: str | None = None
        self._queue_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._width = 0
        self._write_lock = threading.Lock()

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
        if self._animate and self._base_label is not None:
            self._restart(self._compose_label())

    def log_tool_call(self, tool_name: str, raw_arguments: object) -> None:
        if not self._enabled:
            return
        self._print_line(format_tool_preview(tool_name, raw_arguments))

    def log_queued_message(self, text: str) -> None:
        if not self._enabled:
            return
        self._print_line(f"queued {truncate_cli_text(text, 72)}")

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

    def log_tool_result(self, tool_name: str, result: object) -> None:
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

    def clear_turn_state(self) -> None:
        with self._lock:
            self._status = None
            self._activity_line = None
            self._last_tool_line = None
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
