"""Write file contents tool."""

from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool, ToolError


class WriteParams(BaseModel):
    """Parameters for write tool."""

    path: str = Field(description="The file path to write to")
    content: str = Field(description="The content to write to the file")


class WriteTool(BaseTool[WriteParams]):
    """Create or overwrite a file with content."""

    name = "write"
    description = (
        "Write content to a file. Creates parent directories if needed. Overwrites existing files."
    )
    parameters = WriteParams

    async def execute(self, params: WriteParams) -> str:
        """Execute the write tool."""
        path = Path(params.path).expanduser()

        try:
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists for reporting
            existed = path.exists()

            # Write the content
            path.write_text(params.content, encoding="utf-8")

            # Count lines for feedback
            line_count = len(params.content.splitlines())

            if existed:
                return f"Overwrote {params.path} ({line_count} lines)"
            else:
                return f"Created {params.path} ({line_count} lines)"

        except PermissionError as err:
            raise ToolError(f"Error: Permission denied: {params.path}") from err
        except Exception as e:
            raise ToolError(f"Error writing file: {e}") from e
