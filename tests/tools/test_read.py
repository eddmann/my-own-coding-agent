"""Behavior tests for the read tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent.tools import ReadTool, ToolError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_read_file(temp_dir: Path):
    test_file = temp_dir / "test.txt"
    test_file.write_text("line 1\nline 2\nline 3\n")

    tool = ReadTool()

    result = await tool.execute(tool.parameters(path=str(test_file)))

    assert "line 1" in result
    assert "line 2" in result
    assert "line 3" in result


@pytest.mark.asyncio
async def test_read_nonexistent_file(temp_dir: Path):
    tool = ReadTool()

    with pytest.raises(ToolError, match="not found"):
        await tool.execute(tool.parameters(path=str(temp_dir / "missing.txt")))


@pytest.mark.asyncio
async def test_read_with_offset(temp_dir: Path):
    test_file = temp_dir / "test.txt"
    test_file.write_text("\n".join(f"line {i}" for i in range(1, 11)))

    tool = ReadTool()

    result = await tool.execute(tool.parameters(path=str(test_file), offset=5, limit=3))

    assert "line 6" in result
    assert "line 8" in result
    assert "line 1" not in result
