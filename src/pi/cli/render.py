from __future__ import annotations

import json
from typing import Protocol


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


class OutputStream(Protocol):
    def write(self, text: str, /) -> object: ...
    def flush(self) -> object: ...
    def isatty(self) -> bool: ...


class StatusIndicator:
    def __init__(self, stream: OutputStream, *, animate: bool = True) -> None:
        self._stream = stream
        self._animate = animate

    def __enter__(self) -> "StatusIndicator":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def handle_event(self, event: str, payload: dict[str, object]) -> None:
        return None

    def clear(self) -> None:
        return None

    def set_queue_count(self, count: int) -> None:
        return None

    def log_queued_message(self, text: str) -> None:
        if getattr(self._stream, "isatty", lambda: False)():
            self._stream.write(f"queued {truncate_cli_text(text, 72)}\n")
            self._stream.flush()


class InteractiveRenderer(StatusIndicator):
    def print_message(self, text: str) -> None:
        self._stream.write(f"{text}\n")
        self._stream.flush()
