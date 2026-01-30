"""Behavior tests for the bash tool."""

from __future__ import annotations

import pytest

from agent.tools import BashTool, ToolError


@pytest.mark.asyncio
async def test_bash_echo():
    tool = BashTool()

    result = await tool.execute(tool.parameters(command="echo 'hello'"))

    assert "hello" in result


@pytest.mark.asyncio
async def test_bash_with_cwd(temp_dir):
    tool = BashTool()

    result = await tool.execute(tool.parameters(command="pwd", cwd=str(temp_dir)))

    assert str(temp_dir) in result


@pytest.mark.asyncio
async def test_bash_timeout():
    tool = BashTool()

    with pytest.raises(ToolError, match="timed out"):
        await tool.execute(tool.parameters(command="sleep 2", timeout=1))
