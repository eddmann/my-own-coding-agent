"""Tool registry for lookup and execution."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from agent.tools.base import BaseTool, ToolError


@dataclass(slots=True)
class ToolExecutionResult:
    """Result of executing a tool with an error flag."""

    content: str
    is_error: bool = False


class ToolRegistry:
    """Registry for tool lookup and execution."""

    __slots__ = ("_tools",)

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool[Any]] = {}

    def register[P: BaseModel](self, tool: BaseTool[P]) -> None:
        """Register a tool by name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool[Any] | None:
        """Get a tool by name."""
        return self._tools.get(name)

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        """Execute a tool with JSON arguments, returning content + error flag.

        Args:
            name: The tool name
            arguments: The tool arguments as a dictionary

        Returns:
            ToolExecutionResult with content and is_error flag
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolExecutionResult(
                content=f"Error: Unknown tool: {name}",
                is_error=True,
            )

        try:
            params = tool.parameters.model_validate(arguments)
            content = await tool.execute(params)
            return ToolExecutionResult(
                content=content,
                is_error=False,
            )
        except ToolError as e:
            return ToolExecutionResult(content=str(e), is_error=True)
        except ValidationError as e:
            return ToolExecutionResult(
                content=f"Error: Invalid parameters: {e}",
                is_error=True,
            )
        except Exception as e:
            return ToolExecutionResult(
                content=f"Error: {type(e).__name__}: {e}",
                is_error=True,
            )

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI function schemas for all tools."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
