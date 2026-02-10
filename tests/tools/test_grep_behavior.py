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


@pytest.mark.asyncio
async def test_grep_errors_when_path_missing(temp_dir: Path):
    tool = GrepTool()

    with pytest.raises(ToolError, match="Path not found"):
        await tool.execute(tool.parameters(pattern="hello", path=str(temp_dir / "missing.txt")))


@pytest.mark.asyncio
async def test_grep_returns_no_match_message(temp_dir: Path):
    test_file = temp_dir / "test.txt"
    test_file.write_text("alpha\nbeta\n")
    tool = GrepTool()

    result = await tool.execute(tool.parameters(pattern="gamma", path=str(test_file)))

    assert result == "No matches found for pattern: gamma"


@pytest.mark.asyncio
async def test_grep_include_glob_filters_files_in_python_fallback(temp_dir: Path, monkeypatch):
    src_dir = temp_dir / "src"
    src_dir.mkdir()
    py_file = src_dir / "main.py"
    txt_file = src_dir / "notes.txt"
    py_file.write_text("needle = True\n")
    txt_file.write_text("needle in text\n")

    monkeypatch.setenv("PATH", "")

    tool = GrepTool()
    result = await tool.execute(
        tool.parameters(pattern="needle", path=str(src_dir), include="*.py"),
    )

    assert "main.py" in result
    assert "notes.txt" not in result


@pytest.mark.asyncio
async def test_grep_excludes_common_ignored_directories_in_python_fallback(
    temp_dir: Path, monkeypatch
):
    visible = temp_dir / "visible.txt"
    visible.write_text("needle\n")

    ignored_dir = temp_dir / "node_modules"
    ignored_dir.mkdir()
    ignored_file = ignored_dir / "hidden.txt"
    ignored_file.write_text("needle\n")

    monkeypatch.setenv("PATH", "")

    tool = GrepTool()
    result = await tool.execute(tool.parameters(pattern="needle", path=str(temp_dir)))

    assert "visible.txt" in result
    assert str(ignored_file) not in result


@pytest.mark.asyncio
async def test_grep_limits_results_and_appends_notice(temp_dir: Path, monkeypatch):
    test_file = temp_dir / "many.txt"
    test_file.write_text("match-1\nmatch-2\nmatch-3\n")

    monkeypatch.setenv("PATH", "")

    tool = GrepTool()
    result = await tool.execute(
        tool.parameters(pattern="match-", path=str(test_file), max_results=2),
    )

    assert "match-1" in result
    assert "match-2" in result
    assert "match-3" not in result
    assert "[Results limited to 2 matches]" in result
