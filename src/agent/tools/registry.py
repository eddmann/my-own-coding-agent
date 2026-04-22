"""Tool registry for lookup and execution."""

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from agent.tools.base import BaseTool, ToolError


@dataclass(slots=True)
class ToolExecutionError:
    """Structured tool failure details."""

    kind: Literal["unknown_tool", "validation", "tool_error", "unexpected"]
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionResult:
    """Result of executing a tool with an error flag."""

    content: str
    is_error: bool = False
    error: ToolExecutionError | None = None


class ToolRegistry:
    """Registry for tool lookup and execution."""

    __slots__ = ("_active_tools", "_tools")

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool[Any]] = {}
        self._active_tools: set[str] | None = None

    def register[P: BaseModel](self, tool: BaseTool[P]) -> None:
        """Register a tool by name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool[Any] | None:
        """Get a tool by name."""
        return self._tools.get(name)

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        """Execute a tool with JSON arguments, returning content + structured error info.

        Args:
            name: The tool name
            arguments: The tool arguments as a dictionary

        Returns:
            ToolExecutionResult with content and is_error flag
        """
        if self._active_tools is not None and name not in self._active_tools:
            err = ToolExecutionError(
                kind="unknown_tool",
                message=f"Inactive tool: {name}",
                retryable=False,
            )
            return ToolExecutionResult(
                content=f"Error: {err.message}",
                is_error=True,
                error=err,
            )

        tool = self._tools.get(name)
        if tool is None:
            err = ToolExecutionError(
                kind="unknown_tool",
                message=f"Unknown tool: {name}",
                retryable=False,
            )
            return ToolExecutionResult(
                content=f"Error: {err.message}",
                is_error=True,
                error=err,
            )

        try:
            params = tool.parameters.model_validate(arguments)
            content = await tool.execute(params)
            return ToolExecutionResult(
                content=content,
                is_error=False,
            )
        except ToolError as e:
            err = ToolExecutionError(
                kind="tool_error",
                message=str(e),
                retryable=bool(getattr(e, "retryable", False)),
            )
            return ToolExecutionResult(content=str(e), is_error=True, error=err)
        except ValidationError as e:
            err = ToolExecutionError(
                kind="validation",
                message="Invalid parameters",
                retryable=False,
                details={"errors": e.errors()},
            )
            return ToolExecutionResult(
                content=f"Error: Invalid parameters: {e}",
                is_error=True,
                error=err,
            )
        except Exception as e:
            retryable = isinstance(e, (TimeoutError, ConnectionError))
            err = ToolExecutionError(
                kind="unexpected",
                message=f"{type(e).__name__}: {e}",
                retryable=retryable,
            )
            return ToolExecutionResult(
                content=f"Error: {type(e).__name__}: {e}",
                is_error=True,
                error=err,
            )

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI function schemas for all tools."""
        return [self._tools[name].to_openai_schema() for name in self.list_active_tools()]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_active_tools(self) -> list[str]:
        """List the currently active tool names."""
        if self._active_tools is None:
            return self.list_tools()
        return [name for name in self._tools if name in self._active_tools]

    def set_active_tools(self, names: list[str]) -> None:
        """Set the active tools by name.

        Raises:
            ValueError: If any requested tool does not exist
        """
        unknown = [name for name in names if name not in self._tools]
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown tools: {joined}")
        self._active_tools = set(names)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
