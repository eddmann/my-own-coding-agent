"""Behavior tests for extension runner semantics."""

from __future__ import annotations

import asyncio

import pytest

from agent.config import Config
from agent.core.agent import Agent
from agent.core.events import AgentStartEvent
from agent.core.message import Message
from agent.extensions import ExtensionAPI, ExtensionRunner
from agent.extensions.types import (
    ContextModification,
    InputResult,
    ToolCallEvent,
    ToolCallResult,
    ToolResultEvent,
    ToolResultModification,
)
from tests.test_doubles.llm_provider_fake import LLMProviderFake


def build_runner(temp_dir) -> ExtensionRunner:
    config = Config(
        provider="openai",
        model="gpt-4o",
        api_key="sk-test",
        session_dir=temp_dir,
    )
    agent = Agent(
        config.to_agent_settings(),
        LLMProviderFake([], name="openai", model="gpt-4o"),
        cwd=temp_dir,
    )
    return ExtensionRunner(ExtensionAPI(), agent)


@pytest.mark.asyncio
async def test_emit_tool_call_stops_on_first_blocking_result_and_ignores_handler_errors(temp_dir):
    runner = build_runner(temp_dir)
    order: list[str] = []

    def broken_handler(_event, _ctx):
        order.append("broken")
        raise RuntimeError("boom")

    def blocking_handler(_event, _ctx):
        order.append("blocking")
        return ToolCallResult(block=True, reason="blocked by policy")

    def late_handler(_event, _ctx):
        order.append("late")
        return None

    runner.api.on("tool_call", broken_handler)
    runner.api.on("tool_call", blocking_handler)
    runner.api.on("tool_call", late_handler)

    result = await runner.emit_tool_call(ToolCallEvent(tool_name="bash", tool_call_id="c1"))

    assert result is not None
    assert result.block is True
    assert result.reason == "blocked by policy"
    assert order == ["broken", "blocking"]


@pytest.mark.asyncio
async def test_emit_tool_result_merges_modifications_last_value_wins(temp_dir):
    runner = build_runner(temp_dir)

    runner.api.on("tool_result", lambda _event, _ctx: ToolResultModification(content="first"))
    runner.api.on("tool_result", lambda _event, _ctx: ToolResultModification(is_error=True))
    runner.api.on("tool_result", lambda _event, _ctx: ToolResultModification(content="second"))

    modification = await runner.emit_tool_result(
        ToolResultEvent(tool_name="read", tool_call_id="c2", content="orig", is_error=False)
    )

    assert modification is not None
    assert modification.content == "second"
    assert modification.is_error is True


@pytest.mark.asyncio
async def test_emit_context_applies_modifications_in_handler_order(temp_dir):
    runner = build_runner(temp_dir)

    def add_assistant(event, _ctx):
        return ContextModification(messages=[*event.messages, Message.assistant("a1")])

    def replace_with_system(event, _ctx):
        assert [m.content for m in event.messages] == ["u1", "a1"]
        return ContextModification(messages=[Message.system("s1")])

    runner.api.on("context", add_assistant)
    runner.api.on("context", replace_with_system)

    result = await runner.emit_context([Message.user("u1")])

    assert len(result) == 1
    assert result[0].role.value == "system"
    assert result[0].content == "s1"


@pytest.mark.asyncio
async def test_emit_input_returns_block_result_and_stops_later_handlers(temp_dir):
    runner = build_runner(temp_dir)
    order: list[str] = []

    def transform_handler(event, _ctx):
        order.append("transform")
        return InputResult(text=f"{event.text}!")

    def block_handler(_event, _ctx):
        order.append("block")
        return InputResult(block=True, reason="blocked")

    def late_handler(_event, _ctx):
        order.append("late")
        return None

    runner.api.on("input", transform_handler)
    runner.api.on("input", block_handler)
    runner.api.on("input", late_handler)

    result = await runner.emit_input("hello")

    assert result is not None
    assert result.block is True
    assert result.reason == "blocked"
    assert order == ["transform", "block"]


@pytest.mark.asyncio
async def test_emit_input_returns_transformed_text_when_not_blocked(temp_dir):
    runner = build_runner(temp_dir)
    runner.api.on("input", lambda event, _ctx: InputResult(text=event.text.upper()))

    result = await runner.emit_input("hello")

    assert result is not None
    assert result.text == "HELLO"
    assert result.block is False


@pytest.mark.asyncio
async def test_emit_agent_event_runs_handlers_and_swallows_exceptions(temp_dir):
    runner = build_runner(temp_dir)
    called: list[str] = []

    async def slow_handler(_event, _ctx):
        await asyncio.sleep(0)
        called.append("slow")

    async def failing_handler(_event, _ctx):
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    runner.api.on("agent_start", slow_handler)
    runner.api.on("agent_start", failing_handler)

    await runner.emit_agent_event(AgentStartEvent())

    assert called == ["slow"]


@pytest.mark.asyncio
async def test_execute_command_supports_sync_async_and_unknown(temp_dir):
    runner = build_runner(temp_dir)
    runner.api.register_command("sync", lambda args, _ctx: f"sync:{args}")

    async def async_command(args, _ctx):
        await asyncio.sleep(0)
        return f"async:{args}"

    runner.api.register_command("async", async_command)

    unknown = await runner.execute_command("missing", "x")
    sync_result = await runner.execute_command("sync", "a b")
    async_result = await runner.execute_command("async", "c d")

    assert unknown is None
    assert sync_result == "sync:a b"
    assert async_result == "async:c d"
