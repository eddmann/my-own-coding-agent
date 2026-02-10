"""Behavior tests for event stream state transitions."""

from __future__ import annotations

import pytest

from agent.llm.events import DoneEvent, ErrorEvent, PartialMessage
from agent.llm.stream import AssistantMessageEventStream


@pytest.mark.asyncio
async def test_event_stream_delivers_in_order():
    stream = AssistantMessageEventStream()
    message = PartialMessage()

    stream.push(DoneEvent(message=message))
    result = await stream.result()

    assert result is message


@pytest.mark.asyncio
async def test_event_stream_abort_marks_aborted():
    stream = AssistantMessageEventStream()

    stream.abort("cancelled")

    assert stream.is_aborted
    result = await stream.result()
    assert result.stop_reason == "aborted"


@pytest.mark.asyncio
async def test_event_stream_error_sets_result():
    stream = AssistantMessageEventStream()
    message = PartialMessage(error_message="boom")

    stream.push(ErrorEvent(stop_reason="error", message=message))
    result = await stream.result()

    assert result.error_message == "boom"
