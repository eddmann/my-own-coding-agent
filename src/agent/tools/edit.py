"""Edit file contents tool using find and replace."""

from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool, ToolError


class EditParams(BaseModel):
    """Parameters for edit tool."""

    path: str = Field(description="The file path to edit")
    old_string: str = Field(description="The exact string to find and replace")
    new_string: str = Field(description="The string to replace with")
    replace_all: bool = Field(
        default=False, description="Replace all occurrences (default: first only)"
    )


class EditTool(BaseTool[EditParams]):
    """Edit a file by finding and replacing text."""

    name = "edit"
    description = (
        "Edit a file by finding and replacing text. "
        "The old_string must match exactly (including whitespace and indentation). "
        "Use replace_all=true to replace all occurrences."
    )
    parameters = EditParams

    async def execute(self, params: EditParams) -> str:
        """Execute the edit tool."""
        path = Path(params.path).expanduser()

        if not path.exists():
            raise ToolError(f"Error: File not found: {params.path}")

        if not path.is_file():
            raise ToolError(f"Error: Not a file: {params.path}")

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            raise ToolError(f"Error reading file: {e}") from e

        # Check if old_string exists
        if params.old_string not in content:
            # Try to provide helpful feedback
            lines = content.splitlines()
            preview = "\n".join(lines[:20])
            raise ToolError(
                f"Error: String not found in {params.path}\n\n"
                f"Searched for:\n```\n{params.old_string}\n```\n\n"
                f"File preview (first 20 lines):\n```\n{preview}\n```"
            )

        # Count occurrences
        count = content.count(params.old_string)

        # Perform replacement
        if params.replace_all:
            new_content = content.replace(params.old_string, params.new_string)
            replaced_count = count
        else:
            if count > 1:
                raise ToolError(
                    f"Error: Found {count} occurrences of the string in {params.path}. "
                    f"Please provide more context to make the match unique, "
                    f"or use replace_all=true to replace all occurrences."
                )
            new_content = content.replace(params.old_string, params.new_string, 1)
            replaced_count = 1

        try:
            path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            raise ToolError(f"Error writing file: {e}") from e

        if replaced_count == 1:
            return f"Edited {params.path}: replaced 1 occurrence"
        else:
            return f"Edited {params.path}: replaced {replaced_count} occurrences"
