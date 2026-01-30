"""Read file contents tool."""

from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool, ToolError


class ReadParams(BaseModel):
    """Parameters for read tool."""

    path: str = Field(description="The file path to read")
    offset: int = Field(default=0, description="Line number to start reading from (0-indexed)")
    limit: int = Field(default=2000, description="Maximum number of lines to read")


class ReadTool(BaseTool[ReadParams]):
    """Read contents of a file."""

    name = "read"
    description = "Read the contents of a file. Returns lines with line numbers."
    parameters = ReadParams

    async def execute(self, params: ReadParams) -> str:
        """Execute the read tool."""
        path = Path(params.path).expanduser()

        if not path.exists():
            raise ToolError(f"Error: File not found: {params.path}")

        if not path.is_file():
            raise ToolError(f"Error: Not a file: {params.path}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding="latin-1")
            except Exception as e:
                raise ToolError(f"Error reading file: {e}") from e
        except Exception as e:
            raise ToolError(f"Error reading file: {e}") from e

        lines = content.splitlines()
        total_lines = len(lines)

        # Apply offset and limit
        start = max(0, params.offset)
        end = min(start + params.limit, total_lines)
        selected_lines = lines[start:end]

        # Format with line numbers (1-indexed for display)
        formatted_lines = [
            f"{i + start + 1:6}\t{line[:2000]}"  # Truncate very long lines
            for i, line in enumerate(selected_lines)
        ]

        result = "\n".join(formatted_lines)

        # Add info about truncation if applicable
        if end < total_lines or start > 0:
            result += f"\n\n[Showing lines {start + 1}-{end} of {total_lines}]"

        return result
