from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from pi.agent.models import Message

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class SessionRecord(BaseModel):
    id: str
    messages: list[Message] = Field(default_factory=list)
    parent_session: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class SessionHeader(BaseModel):
    type: Literal["session"] = "session"
    id: str
    timestamp: str
    parent_session: str | None = None


class SessionMessageEntry(BaseModel):
    type: Literal["message"] = "message"
    id: str
    timestamp: str
    message: Message


@dataclass(slots=True, frozen=True)
class SessionPaths:
    snapshot_path: Path
    session_dir: Path
    events_path: Path


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.directory = self.root / ".pi" / "sessions"

    def load(self, session_id: str) -> SessionRecord:
        paths = self._paths_for(session_id)
        if paths.snapshot_path.exists():
            try:
                return SessionRecord.model_validate(json.loads(paths.snapshot_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"Failed to load session {session_id!r}: {exc}") from exc
        if not paths.events_path.exists():
            return SessionRecord(id=session_id)
        try:
            record = self._rebuild_from_events(paths.events_path)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Failed to rebuild session {session_id!r}: {exc}") from exc
        self._write_snapshot(paths.snapshot_path, record)
        return record

    def save(self, session_id: str, messages: list[Message], *, parent_session: str | None = None) -> Path:
        paths = self._paths_for(session_id)
        existing = self.load(session_id) if paths.snapshot_path.exists() or paths.events_path.exists() else SessionRecord(id=session_id)
        now = _timestamp()
        created_at = existing.created_at or now
        parent = existing.parent_session or parent_session

        paths.session_dir.mkdir(parents=True, exist_ok=True)
        if not paths.events_path.exists():
            self._write_events_header(paths.events_path, session_id=session_id, created_at=created_at, parent_session=parent)

        previous_count = len(existing.messages)
        if previous_count > len(messages):
            paths.events_path.unlink(missing_ok=True)
            self._write_events_header(paths.events_path, session_id=session_id, created_at=created_at, parent_session=parent)
            previous_count = 0

        with paths.events_path.open("a", encoding="utf-8") as handle:
            for index, message in enumerate(messages[previous_count:], start=previous_count + 1):
                entry = SessionMessageEntry(id=f"msg-{index}", timestamp=now, message=message)
                handle.write(entry.model_dump_json() + "\n")

        record = SessionRecord(
            id=session_id,
            messages=list(messages),
            parent_session=parent,
            created_at=created_at,
            updated_at=now,
        )
        self._write_snapshot(paths.snapshot_path, record)
        return paths.snapshot_path

    def fork(self, source_session_id: str, target_session_id: str) -> Path:
        source = self.load(source_session_id)
        return self.save(target_session_id, source.messages, parent_session=source.id)

    def events_path(self, session_id: str) -> Path:
        return self._paths_for(session_id).events_path

    def _paths_for(self, session_id: str) -> SessionPaths:
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError(
                "Session id must start with an alphanumeric character and use only letters, numbers, dot, underscore, or dash."
            )
        session_dir = self.directory / session_id
        return SessionPaths(
            snapshot_path=self.directory / f"{session_id}.json",
            session_dir=session_dir,
            events_path=session_dir / "events.jsonl",
        )

    def _write_snapshot(self, path: Path, record: SessionRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def _write_events_header(self, path: Path, *, session_id: str, created_at: str, parent_session: str | None) -> None:
        header = SessionHeader(id=session_id, timestamp=created_at, parent_session=parent_session)
        path.write_text(header.model_dump_json() + "\n", encoding="utf-8")

    def _rebuild_from_events(self, path: Path) -> SessionRecord:
        header: SessionHeader | None = None
        messages: list[Message] = []
        updated_at: str | None = None
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("type") == "session":
                    header = SessionHeader.model_validate(payload)
                    updated_at = header.timestamp
                    continue
                entry = SessionMessageEntry.model_validate(payload)
                messages.append(entry.message)
                updated_at = entry.timestamp
        if header is None:
            raise ValueError(f"Session log {path} is missing a header")
        assert header is not None
        return SessionRecord(
            id=header.id,
            messages=messages,
            parent_session=header.parent_session,
            created_at=header.timestamp,
            updated_at=updated_at,
        )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
