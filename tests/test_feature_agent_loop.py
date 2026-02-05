"""Behavior tests for the agent loop."""

from __future__ import annotations

import asyncio

import pytest

from agent.core.agent import Agent
from agent.core.config import Config
from agent.core.message import ToolResult
from agent.llm.events import (
    AssistantMetadataEvent,
    DoneEvent,
    PartialMessage,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
)
from tests.fakes import FakeLLMProvider, make_text_events, make_tool_call_events


def build_agent(temp_dir, scripts):
    """Create an agent wired to a fake LLM provider."""
    config = Config(
        provider="openai",
        model="fake-model",
        api_key="test",
        session_dir=temp_dir,
        context_max_tokens=2048,
        max_output_tokens=2048,
    )
    agent = Agent(config)
    fake = FakeLLMProvider(scripts)
    agent.provider = fake
    agent.context.provider = fake
    return agent, fake


@pytest.mark.asyncio
async def test_agent_streams_text_and_persists_message(temp_dir):
    agent, fake = build_agent(temp_dir, [make_text_events("Hello world")])

    chunks = []
    async for chunk in agent.run("Hi"):
        chunks.append(chunk)

    assert "Hello world" in "".join(c for c in chunks if isinstance(c, str))
    assert len(fake.stream_calls) == 1
    assert agent.session.messages[0].role.value == "system"
    assert agent.session.messages[1].content == "Hi"
    assert agent.session.messages[2].content == "Hello world"


@pytest.mark.asyncio
async def test_agent_runs_tool_call_then_continues(temp_dir):
    test_file = temp_dir / "note.txt"
    test_file.write_text("alpha")

    scripts = [
        make_tool_call_events("call_1", "read", {"path": str(test_file)}),
        make_text_events("Done."),
    ]
    agent, _ = build_agent(temp_dir, scripts)

    chunks = []
    async for chunk in agent.run("Read the file"):
        chunks.append(chunk)

    tool_results = [c for c in chunks if isinstance(c, ToolResult)]
    assert tool_results
    assert "alpha" in tool_results[0].result

    assistant_messages = [m for m in agent.session.messages if m.role.value == "assistant"]
    assert assistant_messages
    assert assistant_messages[-1].content == "Done."


@pytest.mark.asyncio
async def test_agent_respects_cancel_before_stream(temp_dir):
    agent, fake = build_agent(temp_dir, [make_text_events("Should not run")])

    cancel_event = asyncio.Event()
    cancel_event.set()

    chunks = []
    async for chunk in agent.run("Hi", cancel_event=cancel_event):
        chunks.append(chunk)

    assert chunks == []
    assert len(fake.stream_calls) == 0
    assert agent.session.messages[1].content == "Hi"


@pytest.mark.asyncio
async def test_agent_records_thinking_content(temp_dir):
    events = [
        ThinkingStartEvent(content_index=0),
        ThinkingDeltaEvent(content_index=0, delta="step1"),
        ThinkingEndEvent(content_index=0, thinking="step1"),
        DoneEvent(message=PartialMessage()),
    ]
    agent, _ = build_agent(temp_dir, [events])

    chunks = []
    async for chunk in agent.run("Think"):
        chunks.append(chunk)

    assistant_messages = [m for m in agent.session.messages if m.role.value == "assistant"]
    assert assistant_messages
    assert assistant_messages[-1].thinking is not None
    assert "step1" in assistant_messages[-1].thinking.text


@pytest.mark.asyncio
async def test_agent_merges_provider_metadata(temp_dir):
    events = [
        TextStartEvent(content_index=0),
        TextDeltaEvent(content_index=0, delta="Hello"),
        TextEndEvent(content_index=0, text="Hello"),
        AssistantMetadataEvent(metadata={"openai_responses": {"output_item_id": "msg_1"}}),
        AssistantMetadataEvent(
            metadata={"openai_responses": {"reasoning_item": '{"type":"reasoning"}'}}
        ),
        DoneEvent(message=PartialMessage()),
    ]
    agent, _ = build_agent(temp_dir, [events])

    chunks = []
    async for chunk in agent.run("Hi"):
        chunks.append(chunk)

    assistant_messages = [m for m in agent.session.messages if m.role.value == "assistant"]
    assert assistant_messages
    provider_metadata = assistant_messages[-1].provider_metadata
    assert provider_metadata
    assert provider_metadata["openai_responses"]["output_item_id"] == "msg_1"
    assert provider_metadata["openai_responses"]["reasoning_item"] == '{"type":"reasoning"}'
