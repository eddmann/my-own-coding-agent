"""Behavior tests for the edit tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent.tools import EditTool, ToolError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_edit_replace(temp_dir: Path):
    test_file = temp_dir / "test.txt"
    test_file.write_text("hello world")

    tool = EditTool()

    result = await tool.execute(
        tool.parameters(path=str(test_file), old_string="world", new_string="there")
    )

    assert "replaced" in result.lower()
    assert test_file.read_text() == "hello there"


@pytest.mark.asyncio
async def test_edit_string_not_found(temp_dir: Path):
    test_file = temp_dir / "test.txt"
    test_file.write_text("hello world")

    tool = EditTool()

    with pytest.raises(ToolError, match="not found"):
        await tool.execute(tool.parameters(path=str(test_file), old_string="xyz", new_string="abc"))


@pytest.mark.asyncio
async def test_edit_replace_all(temp_dir: Path):
    test_file = temp_dir / "test.txt"
    test_file.write_text("a a a")

    tool = EditTool()

    result = await tool.execute(
        tool.parameters(
            path=str(test_file),
            old_string="a",
            new_string="b",
            replace_all=True,
        )
    )

    assert "replaced" in result
    assert test_file.read_text() == "b b b"


@pytest.mark.asyncio
async def test_edit_requires_unique_match(temp_dir: Path):
    test_file = temp_dir / "test.txt"
    test_file.write_text("x x")

    tool = EditTool()

    with pytest.raises(ToolError, match="occurrences"):
        await tool.execute(tool.parameters(path=str(test_file), old_string="x", new_string="y"))
