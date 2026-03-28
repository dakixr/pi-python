from __future__ import annotations

from typing import Protocol

from pi_python.agent.models import Message


class Provider(Protocol):
    def complete(self, messages: list[Message], tools: list[dict[str, object]]) -> Message:
        ...
