"""Behavior tests for the find tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent.tools import FindTool

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_find_files(temp_dir: Path):
    (temp_dir / "file1.py").write_text("")
    (temp_dir / "file2.py").write_text("")
    (temp_dir / "file3.txt").write_text("")

    tool = FindTool()

    result = await tool.execute(tool.parameters(pattern="*.py", path=str(temp_dir)))

    assert "file1.py" in result
    assert "file2.py" in result
    assert "file3.txt" not in result


@pytest.mark.asyncio
async def test_find_ignores_node_modules(temp_dir: Path):
    node_modules = temp_dir / "node_modules"
    node_modules.mkdir()
    (node_modules / "ignore.js").write_text("x")
    (temp_dir / "keep.py").write_text("x")

    tool = FindTool()

    result = await tool.execute(tool.parameters(pattern="**/*", path=str(temp_dir)))

    assert "keep.py" in result
    assert "node_modules" not in result
