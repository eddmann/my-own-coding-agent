"""Test double: fake LLM provider with in-memory scripted streams."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agent.llm.events import (
    DoneEvent,
    PartialMessage,
    StreamEvent,
    StreamOptions,
)
from agent.llm.provider import LLMProvider
from agent.llm.stream import AssistantMessageEventStream
from tests.test_doubles.llm_stream_builders import make_text_events

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agent.core.message import Message


class LLMProviderFake(LLMProvider):
    """Fake provider that behaves like a real provider without network calls."""

    def __init__(
        self,
        scripts: list[list[StreamEvent]] | None = None,
        *,
        name: str = "fake",
        model: str = "fake-model",
        available_models: list[str] | None = None,
        thinking_supported: bool = True,
    ) -> None:
        self._scripts = list(scripts or [])
        self.stream_calls: list[dict[str, Any]] = []
        self._closed = False
        self.name = name
        self.model = model
        self._available_models = list(available_models or [model])
        self._thinking_supported = thinking_supported

    def set_model(self, model: str) -> None:
        from agent.llm.models import is_model_valid_for_provider

        if model:
            if not is_model_valid_for_provider(model, self.name):
                raise ValueError(f"Model '{model}' is not valid for provider '{self.name}'")
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

        stream.attach_task(asyncio.create_task(_run(events)))
        return stream

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_messages_tokens(self, messages: list[Message]) -> int:
        return sum(self.count_tokens(m.content) + 4 for m in messages)

    def supports_thinking(self) -> bool:
        return self._thinking_supported

    async def list_models(self) -> list[str]:
        return list(self._available_models)

    async def close(self) -> None:
        self._closed = True
