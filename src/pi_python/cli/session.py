from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from pi_python.agent.models import Message

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class SessionRecord(BaseModel):
    id: str
    messages: list[Message] = Field(default_factory=list)


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.directory = self.root / ".pi-python" / "sessions"

    def load(self, session_id: str) -> SessionRecord:
        path = self._path_for(session_id)
        if not path.exists():
            return SessionRecord(id=session_id)

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return SessionRecord.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Failed to load session {session_id!r}: {exc}") from exc

    def save(self, session_id: str, messages: list[Message]) -> Path:
        path = self._path_for(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = SessionRecord(id=session_id, messages=messages)
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return path

    def _path_for(self, session_id: str) -> Path:
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError(
                "Session id must start with an alphanumeric character and use only "
                "letters, numbers, dot, underscore, or dash."
            )
        return self.directory / f"{session_id}.json"
