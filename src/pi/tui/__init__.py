from __future__ import annotations

import re
import textwrap

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def visible_width(text: str) -> int:
    return len(strip_ansi(text))


def truncate_to_width(text: str, width: int, *, ellipsis: str = "...") -> str:
    if width <= 0:
        return ""
    if visible_width(text) <= width:
        return text
    if width <= len(ellipsis):
        return ellipsis[:width]
    stripped = strip_ansi(text)
    return stripped[: width - len(ellipsis)].rstrip() + ellipsis


def wrap_text_with_ansi(text: str, width: int) -> list[str]:
    if width <= 0:
        return [""]
    stripped = strip_ansi(text)
    return textwrap.wrap(stripped, width=width) or [""]


__all__ = [
    "strip_ansi",
    "truncate_to_width",
    "visible_width",
    "wrap_text_with_ansi",
]
