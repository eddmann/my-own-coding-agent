"""Behavior tests for agent and extension runtime interop."""

from __future__ import annotations

import textwrap

import pytest

from agent.config import Config
from agent.extensions.host import ExtensionHost
from agent.runtime.agent import Agent
from tests.test_doubles.llm_provider_fake import LLMProviderFake
from tests.test_doubles.llm_stream_builders import make_text_events


def build_agent(temp_dir, scripts):
    config = Config(
        provider="openai",
        model="fake-model",
        api_key="test",
        session_dir=temp_dir,
        context_max_tokens=2048,
        max_output_tokens=2048,
    )
    provider = LLMProviderFake(scripts)
    agent = Agent(config.to_agent_settings(), provider)
    host = ExtensionHost(agent)
    agent.set_hooks(host)
    return agent, provider, host


def write_extension(path, content: str) -> None:
    path.write_text(textwrap.dedent(content))


@pytest.mark.asyncio
async def test_agent_runs_extension_command_without_calling_llm(temp_dir):
    ext_path = temp_dir / "ext_cmd.py"
    write_extension(
        ext_path,
        """
        from agent.extensions.api import ExtensionAPI

        def setup(api: ExtensionAPI):
            def ping(args, ctx):
                return "pong"
            api.register_command("ping", ping)
        """,
    )

    agent, provider, host = build_agent(temp_dir, [make_text_events("should not run")])
    await host.load_extensions([ext_path])

    chunks = []
    async for chunk in agent.run("/ping hello"):
        chunks.append(chunk)

    system_messages = [
        c.payload for c in chunks if c.type == "message" and c.payload.role.value == "system"
    ]
    assert system_messages
    assert "pong" in system_messages[0].content
    assert len(provider.stream_calls) == 0


@pytest.mark.asyncio
async def test_agent_applies_extension_input_transformation_before_prompting_llm(temp_dir):
    ext_path = temp_dir / "ext_input.py"
    write_extension(
        ext_path,
        """
        from agent.extensions.api import ExtensionAPI
        from agent.extensions.types import InputResult

        def setup(api: ExtensionAPI):
            def rewrite(event, ctx):
                return InputResult(text="rewritten")
            api.on("input", rewrite)
        """,
    )

    agent, _, host = build_agent(temp_dir, [make_text_events("ok")])
    await host.load_extensions([ext_path])

    async for _ in agent.run("original"):
        pass

    assert agent.session.messages[1].content == "rewritten"


@pytest.mark.asyncio
async def test_agent_can_handle_input_entirely_from_input_hook_without_llm(temp_dir):
    ext_path = temp_dir / "ext_input_handle.py"
    write_extension(
        ext_path,
        """
        from agent.extensions.api import ExtensionAPI
        from agent.extensions.types import InputResult

        def setup(api: ExtensionAPI):
            def handle(event, ctx):
                if event.text == "special":
                    return InputResult(handled=True, handled_output="handled")
                return None
            api.on("input", handle)
        """,
    )

    agent, provider, host = build_agent(temp_dir, [make_text_events("should not run")])
    await host.load_extensions([ext_path])

    chunks = []
    async for chunk in agent.run("special"):
        chunks.append(chunk)

    system_messages = [
        c.payload for c in chunks if c.type == "message" and c.payload.role.value == "system"
    ]
    assert len(provider.stream_calls) == 0
    assert system_messages
    assert system_messages[0].content == "handled"
