"""Behavior tests for the tool registry."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent.tools import BaseTool, ReadTool, ToolRegistry, WriteTool


def test_register_and_get():
    registry = ToolRegistry()
    read_tool = ReadTool()

    registry.register(read_tool)

    assert "read" in registry
    assert registry.get("read") is read_tool


def test_get_schemas():
    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())

    schemas = registry.get_schemas()

    assert len(schemas) == 2
    assert all(schema["type"] == "function" for schema in schemas)


@pytest.mark.asyncio
async def test_execute(temp_dir):
    test_file = temp_dir / "test.txt"
    test_file.write_text("content")

    registry = ToolRegistry()
    registry.register(ReadTool())

    result = await registry.execute("read", {"path": str(test_file)})

    assert result.is_error is False
    assert "content" in result.content


@pytest.mark.asyncio
async def test_execute_unknown_tool():
    registry = ToolRegistry()

    result = await registry.execute("unknown", {})

    assert result.is_error is True
    assert "Error" in result.content
    assert "Unknown tool" in result.content
    assert result.error is not None
    assert result.error.kind == "unknown_tool"
    assert result.error.retryable is False


class EmptyParams(BaseModel):
    pass


class TimeoutTool(BaseTool[EmptyParams]):
    name = "timeout_tool"
    description = "Raises TimeoutError to simulate retryable failures."
    parameters = EmptyParams

    async def execute(self, params: EmptyParams) -> str:
        raise TimeoutError("timed out")


@pytest.mark.asyncio
async def test_execute_marks_timeout_errors_retryable():
    registry = ToolRegistry()
    registry.register(TimeoutTool())

    result = await registry.execute("timeout_tool", {})

    assert result.is_error is True
    assert result.error is not None
    assert result.error.kind == "unexpected"
    assert result.error.retryable is True
