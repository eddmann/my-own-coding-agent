"""Message types for the conversation history."""

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, Field


class Role(StrEnum):
    """Message roles following OpenAI convention."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool call from the assistant."""

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Self:
        """Parse from OpenAI API format."""
        args = data.get("function", {}).get("arguments", "{}")
        if isinstance(args, str):
            args = json.loads(args) if args else {}
        return cls(
            id=data.get("id", uuid4().hex[:12]),
            name=data.get("function", {}).get("name", ""),
            arguments=args,
        )

    def to_api_dict(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


class ToolResult(BaseModel):
    """Result from executing a tool."""

    tool_call_id: str
    name: str
    result: str


class ToolCallStart(BaseModel):
    """Notification that a tool call is starting to stream.

    Emitted when toolcall_start event is received, before arguments
    are fully parsed. Allows UI to show tool name immediately.
    """

    id: str
    name: str


class ThinkingContent(BaseModel):
    """Thinking/reasoning block from LLM.

    Represents model reasoning that may be shown to users but is not
    part of the main response. Used by models like Claude (extended thinking)
    and O1/O3 (reasoning effort).
    """

    text: str = ""
    signature: str | None = None  # For replay (Anthropic redacted thinking)


class Message(BaseModel):
    """A message in the conversation."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    role: Role
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool results
    thinking: ThinkingContent | None = None  # For reasoning/thinking content
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    parent_id: str | None = None  # For branching

    def to_api_dict(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}

        if self.tool_calls:
            d["tool_calls"] = [tc.to_api_dict() for tc in self.tool_calls]

        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id

        return d

    @classmethod
    def user(cls, content: str) -> Self:
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: str,
        tool_calls: list[ToolCall] | None = None,
        thinking: ThinkingContent | None = None,
    ) -> Self:
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls, thinking=thinking)

    @classmethod
    def system(cls, content: str) -> Self:
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> Self:
        """Create a tool result message."""
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)
