"""Behavior tests for the grep tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent.tools import GrepTool, ToolError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_grep_pattern(temp_dir: Path):
    test_file = temp_dir / "test.txt"
    test_file.write_text("hello world\nfoo bar\nhello again\n")

    tool = GrepTool()

    result = await tool.execute(tool.parameters(pattern="hello", path=str(test_file)))

    assert "hello world" in result
    assert "hello again" in result
    assert "foo bar" not in result


@pytest.mark.asyncio
async def test_grep_invalid_regex(temp_dir: Path):
    test_file = temp_dir / "test.txt"
    test_file.write_text("hello")

    tool = GrepTool()

    with pytest.raises(ToolError, match="Invalid regex"):
        await tool.execute(tool.parameters(pattern="[", path=str(test_file)))
