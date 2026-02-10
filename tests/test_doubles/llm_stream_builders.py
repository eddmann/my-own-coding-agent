"""Test double helpers for building LLM stream event scripts."""

from __future__ import annotations

from typing import Any

from agent.llm.events import (
    DoneEvent,
    ErrorEvent,
    PartialMessage,
    StreamEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
    ToolCallEndEvent,
    ToolCallStartEvent,
)


def make_text_events(text: str) -> list[StreamEvent]:
    """Build a simple text stream."""
    events: list[StreamEvent] = [TextStartEvent(content_index=0)]
    if text:
        events.append(TextDeltaEvent(content_index=0, delta=text))
    events.append(TextEndEvent(content_index=0, text=text))
    events.append(DoneEvent(message=PartialMessage()))
    return events


def make_tool_call_events(tool_id: str, name: str, arguments: dict[str, Any]) -> list[StreamEvent]:
    """Build a tool call stream."""
    block = ToolCallBlock(id=tool_id, name=name, arguments=arguments)
    return [
        ToolCallStartEvent(content_index=0, tool_id=tool_id, tool_name=name),
        ToolCallEndEvent(content_index=0, tool_call=block),
        DoneEvent(message=PartialMessage()),
    ]


def make_error_events(message: str) -> list[StreamEvent]:
    """Build an error stream."""
    partial = PartialMessage(error_message=message, stop_reason="error")
    return [ErrorEvent(stop_reason="error", message=partial)]
