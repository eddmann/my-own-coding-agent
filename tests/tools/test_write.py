"""Behavior tests for the write tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent.tools import WriteTool

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_write_new_file(temp_dir: Path):
    test_file = temp_dir / "new.txt"

    tool = WriteTool()

    result = await tool.execute(tool.parameters(path=str(test_file), content="hello world"))

    assert "Created" in result
    assert test_file.exists()
    assert test_file.read_text() == "hello world"


@pytest.mark.asyncio
async def test_write_creates_directories(temp_dir: Path):
    test_file = temp_dir / "nested" / "dir" / "file.txt"

    tool = WriteTool()

    await tool.execute(tool.parameters(path=str(test_file), content="nested content"))

    assert test_file.exists()
    assert test_file.read_text() == "nested content"
