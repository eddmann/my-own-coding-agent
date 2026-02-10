"""Behavior tests for extension loading and runtime integration."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from agent.config import Config
from agent.core.agent import Agent
from agent.extensions.api import ExtensionAPI
from agent.extensions.loader import ExtensionLoader
from tests.test_doubles.llm_provider_fake import LLMProviderFake
from tests.test_doubles.llm_stream_builders import make_text_events, make_tool_call_events

if TYPE_CHECKING:
    from pathlib import Path


def build_agent(temp_dir, scripts):
    config = Config(
        provider="openai",
        model="fake-model",
        api_key="test",
        session_dir=temp_dir,
        context_max_tokens=2048,
        max_output_tokens=2048,
    )
    fake = LLMProviderFake(scripts)
    agent = Agent(config.to_agent_settings(), fake)
    return agent, fake


def write_extension(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content))


def write_ext(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content))


@pytest.mark.asyncio
async def test_extension_loader_errors_for_missing_file(temp_dir):
    api = ExtensionAPI()

    error = await ExtensionLoader.load(temp_dir / "missing.py", api)

    assert error is not None
    assert "not found" in error


@pytest.mark.asyncio
async def test_extension_loader_rejects_non_py(temp_dir):
    api = ExtensionAPI()
    path = temp_dir / "ext.txt"
    path.write_text("x")

    error = await ExtensionLoader.load(path, api)

    assert error is not None
    assert "Python file" in error


@pytest.mark.asyncio
async def test_extension_loader_requires_setup(temp_dir):
    api = ExtensionAPI()
    path = temp_dir / "ext.py"
    write_ext(path, "x = 1")

    error = await ExtensionLoader.load(path, api)

    assert error is not None
    assert "setup" in error


@pytest.mark.asyncio
async def test_extension_loader_directory_skips_private_files(temp_dir):
    api = ExtensionAPI()
    directory = temp_dir / "exts"
    directory.mkdir()

    write_ext(directory / "_skip.py", "raise RuntimeError('skip')")
    write_ext(
        directory / "ok.py",
        """
        def setup(api):
            pass
        """,
    )

    errors = await ExtensionLoader.load_directory(directory, api)

    assert errors == []


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
