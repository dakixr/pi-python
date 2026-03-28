from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant", "tool"]


class ToolFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolFunction


class Message(BaseModel):
    role: Role
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> "Message":
        return cls(role="tool", tool_call_id=tool_call_id, content=content)

    def to_api_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"role": self.role}
        if self.content is not None or self.role == "assistant":
            payload["content"] = self.content
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [tool_call.model_dump() for tool_call in self.tool_calls]
        return payload
