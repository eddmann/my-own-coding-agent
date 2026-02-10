"""Test fakes for external dependencies."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agent.llm.events import (
    DoneEvent,
    ErrorEvent,
    PartialMessage,
    StreamEvent,
    StreamOptions,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from agent.llm.provider import LLMProvider
from agent.llm.stream import AssistantMessageEventStream

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agent.core.message import Message


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


class FakeLLMProvider(LLMProvider):
    """Fake LLM provider that streams pre-scripted events."""

    def __init__(self, scripts: list[list[StreamEvent]] | None = None) -> None:
        self._scripts = list(scripts or [])
        self.stream_calls: list[dict[str, Any]] = []
        self._closed = False
        self.model = "fake-model"

    def set_model(self, model: str) -> None:
        if model:
            self.model = model

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        self.stream_calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "options": options,
            }
        )

        stream = AssistantMessageEventStream()
        events = self._scripts.pop(0) if self._scripts else make_text_events("")

        async def _run(events_to_send: Iterable[StreamEvent]) -> None:
            for event in events_to_send:
                stream.push(event)
                await asyncio.sleep(0)
            if not any(e.type in ("done", "error") for e in events_to_send):
                stream.push(DoneEvent(message=PartialMessage()))
            stream.end()

        stream._task = asyncio.create_task(_run(events))
        return stream

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_messages_tokens(self, messages: list[Message]) -> int:
        return sum(self.count_tokens(m.content) + 4 for m in messages)

    def supports_thinking(self) -> bool:
        return True

    async def list_models(self) -> list[str]:
        return ["fake-model"]

    async def close(self) -> None:
        self._closed = True
