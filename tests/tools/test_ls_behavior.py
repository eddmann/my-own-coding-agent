"""Behavior tests for the ls tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent.tools import LsTool

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_ls_directory(temp_dir: Path):
    (temp_dir / "file.txt").write_text("")
    (temp_dir / "subdir").mkdir()

    tool = LsTool()

    result = await tool.execute(tool.parameters(path=str(temp_dir)))

    assert "file.txt" in result
    assert "subdir/" in result


@pytest.mark.asyncio
async def test_ls_hidden_files(temp_dir: Path):
    (temp_dir / ".hidden").write_text("")
    (temp_dir / "visible").write_text("")

    tool = LsTool()

    result = await tool.execute(tool.parameters(path=str(temp_dir), all=False))
    assert ".hidden" not in result
    assert "visible" in result

    result = await tool.execute(tool.parameters(path=str(temp_dir), all=True))
    assert ".hidden" in result


@pytest.mark.asyncio
async def test_ls_long_format_includes_filename(temp_dir: Path):
    test_file = temp_dir / "file.txt"
    test_file.write_text("hi")

    tool = LsTool()

    result = await tool.execute(tool.parameters(path=str(temp_dir), long=True))

    assert "file.txt" in result
