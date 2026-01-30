"""Behavior tests for extensions."""

from __future__ import annotations

import textwrap

import pytest

from agent.core.agent import Agent
from agent.core.config import Config
from agent.core.message import Message
from tests.fakes import FakeLLMProvider, make_text_events, make_tool_call_events


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
    return agent, fake


def write_extension(path, content: str) -> None:
    path.write_text(textwrap.dedent(content))


@pytest.mark.asyncio
async def test_extension_command_executes_without_llm(temp_dir):
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

    agent, fake = build_agent(temp_dir, [make_text_events("should not run")])
    await agent.load_extensions([ext_path])

    chunks = []
    async for chunk in agent.run("/ping hello"):
        chunks.append(chunk)

    system_msgs = [c for c in chunks if isinstance(c, Message) and c.role.value == "system"]
    assert system_msgs
    assert "pong" in system_msgs[0].content
    assert len(fake.stream_calls) == 0


@pytest.mark.asyncio
async def test_extension_blocks_tool_execution(temp_dir):
    ext_path = temp_dir / "ext_block.py"
    write_extension(
        ext_path,
        """
        from agent.extensions.api import ExtensionAPI
        from agent.extensions.types import ToolCallResult

        def setup(api: ExtensionAPI):
            def block_bash(event, ctx):
                if event.tool_name == "bash":
                    return ToolCallResult(block=True, reason="no bash")
            api.on("tool_call", block_bash)
        """,
    )

    touch_file = temp_dir / "blocked.txt"
    scripts = [make_tool_call_events("call_1", "bash", {"command": f"touch {touch_file}"})]
    agent, _ = build_agent(temp_dir, scripts)
    await agent.load_extensions([ext_path])

    async for _ in agent.run("run bash"):
        pass

    assert not touch_file.exists()
    tool_msgs = [m for m in agent.session.messages if m.role.value == "tool"]
    assert tool_msgs
    assert "Tool blocked" in tool_msgs[-1].content


@pytest.mark.asyncio
async def test_extension_modifies_tool_result(temp_dir):
    ext_path = temp_dir / "ext_modify.py"
    write_extension(
        ext_path,
        """
        from agent.extensions.api import ExtensionAPI
        from agent.extensions.types import ToolResultModification

        def setup(api: ExtensionAPI):
            def redact(event, ctx):
                if event.tool_name == "read":
                    return ToolResultModification(content="[redacted]")
            api.on("tool_result", redact)
        """,
    )

    test_file = temp_dir / "note.txt"
    test_file.write_text("secret")

    scripts = [make_tool_call_events("call_1", "read", {"path": str(test_file)})]
    agent, _ = build_agent(temp_dir, scripts)
    await agent.load_extensions([ext_path])

    async for _ in agent.run("read file"):
        pass

    tool_msgs = [m for m in agent.session.messages if m.role.value == "tool"]
    assert tool_msgs
    assert tool_msgs[-1].content == "[redacted]"


@pytest.mark.asyncio
async def test_extension_transforms_input(temp_dir):
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

    agent, _ = build_agent(temp_dir, [make_text_events("ok")])
    await agent.load_extensions([ext_path])

    async for _ in agent.run("original"):
        pass

    assert agent.session.messages[1].content == "rewritten"


@pytest.mark.asyncio
async def test_extension_registers_custom_tool(temp_dir):
    ext_path = temp_dir / "ext_tool.py"
    write_extension(
        ext_path,
        """
        from pydantic import BaseModel, Field
        from agent.extensions.api import ExtensionAPI
        from agent.tools.base import BaseTool

        class EchoParams(BaseModel):
            text: str = Field(description="text to echo")

        class EchoTool(BaseTool[EchoParams]):
            name = "echo"
            description = "Echo text"
            parameters = EchoParams

            async def execute(self, params: EchoParams) -> str:
                return f"echo: {params.text}"

        def setup(api: ExtensionAPI):
            api.register_tool(EchoTool())
        """,
    )

    agent, _ = build_agent(temp_dir, [make_text_events("ok")])
    await agent.load_extensions([ext_path])

    assert "echo" in agent.tools
    result = await agent.tools.execute("echo", {"text": "hi"})
    assert result.is_error is False
    assert result.content == "echo: hi"
