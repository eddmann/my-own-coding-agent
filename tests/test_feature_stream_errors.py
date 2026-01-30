"""Behavior tests for stream error handling."""

from __future__ import annotations

import pytest

from agent.core.agent import Agent
from agent.core.config import Config
from agent.core.message import Message
from tests.fakes import FakeLLMProvider, make_error_events


def build_agent(temp_dir, scripts):
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
    return agent


@pytest.mark.asyncio
async def test_stream_error_surfaces_as_system_message(temp_dir):
    agent = build_agent(temp_dir, [make_error_events("boom")])

    chunks = []
    async for chunk in agent.run("Hello"):
        chunks.append(chunk)

    system_messages = [c for c in chunks if isinstance(c, Message) and c.role.value == "system"]
    assert system_messages
    assert "LLM stream error" in system_messages[0].content

    persisted = [m for m in agent.session.messages if m.role.value == "system"]
    assert any("LLM stream error" in m.content for m in persisted)
