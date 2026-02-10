"""Behavior tests for model selection in the agent."""

from __future__ import annotations

import asyncio

import pytest

from agent.config import Config
from agent.core.agent import Agent
from agent.core.events import ModelSelectEvent
from agent.core.session import Session
from agent.core.settings import ThinkingLevel
from tests.test_doubles.llm_provider_fake import LLMProviderFake


def build_agent(
    temp_dir,
    *,
    provider_name: str = "fake",
    provider_model: str = "fake-model",
    thinking_level: ThinkingLevel = ThinkingLevel.OFF,
    session: Session | None = None,
) -> tuple[Agent, LLMProviderFake]:
    config = Config(
        provider="openai",
        model=provider_model,
        api_key="test",
        session_dir=temp_dir,
        context_max_tokens=2048,
        max_output_tokens=2048,
        thinking_level=thinking_level,
    )
    provider = LLMProviderFake([], name=provider_name, model=provider_model)
    return Agent(config.to_agent_settings(), provider, session=session), provider


def test_agent_set_model_updates_provider_and_persists_selection(temp_dir):
    agent, provider = build_agent(temp_dir, provider_name="openai", provider_model="gpt-4o")

    agent.set_model("gpt-5")

    assert provider.model == "gpt-5"
    assert agent.session.get_model_selection() == ("openai", "gpt-5")
    assert any(entry.type == "model_change" for entry in agent.session.entries)


def test_agent_set_model_noop_for_same_model_does_not_append_entry(temp_dir):
    agent, _ = build_agent(temp_dir, provider_name="openai", provider_model="gpt-4o")
    entry_count = len(agent.session.entries)

    agent.set_model("gpt-4o")

    assert len(agent.session.entries) == entry_count


def test_agent_set_model_clamps_thinking_level_for_lower_capability_model(temp_dir):
    agent, provider = build_agent(
        temp_dir,
        provider_name="openai",
        provider_model="gpt-5",
        thinking_level=ThinkingLevel.HIGH,
    )

    agent.set_model("gpt-4o")

    assert provider.model == "gpt-4o"
    assert agent.config.thinking_level == ThinkingLevel.OFF


def test_agent_set_model_rejects_invalid_provider_model_pair(temp_dir):
    agent, provider = build_agent(temp_dir, provider_name="openai", provider_model="gpt-4o")

    with pytest.raises(ValueError):
        agent.set_model("claude-sonnet-4-5")

    assert provider.model == "gpt-4o"


def test_agent_restores_model_selection_from_session_entries(temp_dir):
    session = Session.new(temp_dir, provider="openai", model="gpt-4o")
    session.append_model_change("openai", "gpt-5")

    agent, provider = build_agent(
        temp_dir,
        provider_name="openai",
        provider_model="gpt-4o",
        session=session,
    )

    assert provider.model == "gpt-5"
    assert agent.session.get_model_selection() == ("openai", "gpt-5")


def test_agent_restore_ignores_selection_for_different_provider(temp_dir):
    session = Session.new(temp_dir, provider="openai", model="gpt-4o")
    session.append_model_change("anthropic", "claude-sonnet-4-5")

    agent, provider = build_agent(
        temp_dir,
        provider_name="openai",
        provider_model="gpt-4o",
        session=session,
    )

    assert provider.model == "gpt-4o"
    assert agent.session.get_model_selection() == ("anthropic", "claude-sonnet-4-5")


def test_agent_restore_ignores_invalid_model_for_provider(temp_dir):
    session = Session.new(temp_dir, provider="openai", model="gpt-4o")
    session.append_model_change("openai", "claude-sonnet-4-5")

    agent, provider = build_agent(
        temp_dir,
        provider_name="openai",
        provider_model="gpt-4o",
        session=session,
    )

    assert provider.model == "gpt-4o"


@pytest.mark.asyncio
async def test_agent_set_model_emits_model_select_event_to_subscribers(temp_dir):
    agent, _ = build_agent(temp_dir, provider_name="openai", provider_model="gpt-4o")
    events: list[ModelSelectEvent] = []

    def listener(event):
        if isinstance(event, ModelSelectEvent):
            events.append(event)

    agent.subscribe(listener)
    agent.set_model("gpt-5")

    for _ in range(3):
        await asyncio.sleep(0)

    assert len(events) == 1
    assert events[0].provider == "openai"
    assert events[0].previous_model == "gpt-4o"
    assert events[0].model == "gpt-5"
    assert events[0].source == "set"


@pytest.mark.asyncio
async def test_agent_load_session_emits_restore_model_select_event(temp_dir):
    saved = Session.new(temp_dir, provider="openai", model="gpt-4o")
    saved.append_model_change("openai", "gpt-5")

    agent, _ = build_agent(temp_dir, provider_name="openai", provider_model="gpt-4o")
    events: list[ModelSelectEvent] = []

    def listener(event):
        if isinstance(event, ModelSelectEvent):
            events.append(event)

    agent.subscribe(listener)
    agent.load_session(saved)

    for _ in range(3):
        await asyncio.sleep(0)

    assert events
    restore_event = events[-1]
    assert restore_event.provider == "openai"
    assert restore_event.model == "gpt-5"
    assert restore_event.source == "restore"
