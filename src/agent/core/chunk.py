"""Agent output chunk types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agent.core.message import Message, ThinkingContent, ToolCall, ToolCallStart, ToolResult

AgentChunkType = Literal[
    "text_delta",
    "thinking_delta",
    "tool_call_start",
    "tool_call",
    "tool_result",
    "message",
]


@dataclass(slots=True)
class TextDeltaChunk:
    """A streamed text token from the assistant."""

    payload: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True)
class ThinkingDeltaChunk:
    """A streamed thinking token from the assistant."""

    payload: ThinkingContent
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(slots=True)
class ToolCallStartChunk:
    """Notification that a tool call has started streaming."""

    payload: ToolCallStart
    type: Literal["tool_call_start"] = "tool_call_start"


@dataclass(slots=True)
class ToolCallChunk:
    """Completed tool call emitted by the model."""

    payload: ToolCall
    type: Literal["tool_call"] = "tool_call"


@dataclass(slots=True)
class ToolResultChunk:
    """Result from executing a tool call."""

    payload: ToolResult
    type: Literal["tool_result"] = "tool_result"


@dataclass(slots=True)
class MessageChunk:
    """A full message chunk, currently used for system messages."""

    payload: Message
    type: Literal["message"] = "message"


AgentChunk = (
    TextDeltaChunk
    | ThinkingDeltaChunk
    | ToolCallStartChunk
    | ToolCallChunk
    | ToolResultChunk
    | MessageChunk
)
