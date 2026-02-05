"""Streaming event types used by the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import asyncio

StopReason = Literal["stop", "length", "tool_use", "error", "aborted"]


@dataclass(slots=True)
class Cost:
    """Cost breakdown for token usage."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass(slots=True)
class Usage:
    """Token usage and cost tracking."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: Cost = field(default_factory=Cost)

    def update_total(self) -> None:
        """Update total_tokens from components."""
        self.total_tokens = self.input + self.output


# Tool choice type: "auto", "none", "required"/"any", or {"name": "tool_name"}
ToolChoice = str | dict[str, str] | None


@dataclass(slots=True)
class StreamOptions:
    """Options for streaming completion."""

    temperature: float | None = None
    max_tokens: int | None = None
    thinking_level: str | None = None  # "off", "minimal", "low", "medium", "high"
    api_key: str | None = None
    tool_choice: ToolChoice = None  # "auto", "none", "required"/"any", {"name": "tool"}
    cancel_event: asyncio.Event | None = None  # Set to abort the request


@dataclass(slots=True)
class PartialMessage:
    """In-progress assistant message built during streaming."""

    content: list[Any] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    error_message: str | None = None


# Content blocks
@dataclass(slots=True)
class TextBlock:
    """Text content block."""

    type: Literal["text"] = "text"
    text: str = ""


@dataclass(slots=True)
class ThinkingBlock:
    """Thinking/reasoning content block."""

    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    signature: str | None = None


@dataclass(slots=True)
class ToolCallBlock:
    """Tool call content block."""

    type: Literal["tool_call"] = "tool_call"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    _partial_json: str = ""


ContentBlock = TextBlock | ThinkingBlock | ToolCallBlock


# Events
@dataclass(slots=True)
class StartEvent:
    """Stream started."""

    type: Literal["start"] = "start"
    partial: PartialMessage = field(default_factory=PartialMessage)


@dataclass(slots=True)
class TextStartEvent:
    """Text block started."""

    type: Literal["text_start"] = "text_start"
    content_index: int = 0
    partial: PartialMessage = field(default_factory=PartialMessage)


@dataclass(slots=True)
class TextDeltaEvent:
    """Text delta received."""

    type: Literal["text_delta"] = "text_delta"
    content_index: int = 0
    delta: str = ""
    partial: PartialMessage = field(default_factory=PartialMessage)


@dataclass(slots=True)
class TextEndEvent:
    """Text block ended."""

    type: Literal["text_end"] = "text_end"
    content_index: int = 0
    text: str = ""
    partial: PartialMessage = field(default_factory=PartialMessage)


@dataclass(slots=True)
class ThinkingStartEvent:
    """Thinking block started."""

    type: Literal["thinking_start"] = "thinking_start"
    content_index: int = 0
    partial: PartialMessage = field(default_factory=PartialMessage)


@dataclass(slots=True)
class ThinkingDeltaEvent:
    """Thinking delta received."""

    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int = 0
    delta: str = ""
    partial: PartialMessage = field(default_factory=PartialMessage)


@dataclass(slots=True)
class ThinkingEndEvent:
    """Thinking block ended."""

    type: Literal["thinking_end"] = "thinking_end"
    content_index: int = 0
    thinking: str = ""
    signature: str | None = None
    partial: PartialMessage = field(default_factory=PartialMessage)


@dataclass(slots=True)
class ToolCallStartEvent:
    """Tool call started."""

    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int = 0
    tool_id: str = ""
    tool_name: str = ""
    partial: PartialMessage = field(default_factory=PartialMessage)


@dataclass(slots=True)
class ToolCallDeltaEvent:
    """Tool call arguments delta."""

    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int = 0
    delta: str = ""
    partial: PartialMessage = field(default_factory=PartialMessage)


@dataclass(slots=True)
class ToolCallEndEvent:
    """Tool call completed."""

    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int = 0
    tool_call: ToolCallBlock = field(default_factory=ToolCallBlock)
    partial: PartialMessage = field(default_factory=PartialMessage)


@dataclass(slots=True)
class AssistantMetadataEvent:
    """Provider-specific metadata for the assistant message."""

    type: Literal["assistant_metadata"] = "assistant_metadata"
    metadata: dict[str, Any] = field(default_factory=dict)
    partial: PartialMessage = field(default_factory=PartialMessage)


@dataclass(slots=True)
class DoneEvent:
    """Stream completed successfully."""

    type: Literal["done"] = "done"
    stop_reason: StopReason = "stop"
    message: PartialMessage = field(default_factory=PartialMessage)


@dataclass(slots=True)
class ErrorEvent:
    """Stream ended with error."""

    type: Literal["error"] = "error"
    stop_reason: StopReason = "error"
    message: PartialMessage = field(default_factory=PartialMessage)


StreamEvent = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | AssistantMetadataEvent
    | DoneEvent
    | ErrorEvent
)
