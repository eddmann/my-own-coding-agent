"""Base tool protocol and implementation."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolError(Exception):
    """Raised by tools to signal an execution error."""


class BaseTool[P: BaseModel](ABC):
    """Base implementation for tools with typed parameters.

    Type parameter P is the Pydantic model for tool parameters.
    """

    name: str
    description: str
    parameters: type[P]

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.model_json_schema(),
            },
        }

    @abstractmethod
    async def execute(self, params: P) -> str:
        """Execute the tool with validated parameters.

        Args:
            params: The validated parameters

        Returns:
            The tool output as a string
        """
        ...
