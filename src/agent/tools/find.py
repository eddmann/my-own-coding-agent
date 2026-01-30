"""Find files by pattern tool."""

from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool, ToolError


class FindParams(BaseModel):
    """Parameters for find tool."""

    pattern: str = Field(description="Glob pattern to match files (e.g., '*.py', '**/*.ts')")
    path: str = Field(default=".", description="Directory to search in")
    max_results: int = Field(default=100, description="Maximum number of results to return")


class FindTool(BaseTool[FindParams]):
    """Find files by glob pattern."""

    name = "find"
    description = (
        "Find files matching a glob pattern. "
        "Use '**' for recursive matching (e.g., '**/*.py'). "
        "Returns file paths relative to the search directory."
    )
    parameters = FindParams

    async def execute(self, params: FindParams) -> str:
        """Execute the find tool."""
        path = Path(params.path).expanduser()

        if not path.exists():
            raise ToolError(f"Error: Directory not found: {params.path}")

        if not path.is_dir():
            raise ToolError(f"Error: Not a directory: {params.path}")

        # Collect matching files
        matches: list[str] = []
        ignore_dirs = {
            ".git",
            ".hg",
            ".svn",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".tox",
            ".mypy_cache",
            ".pytest_cache",
            "dist",
            "build",
            ".eggs",
        }

        try:
            for match in path.glob(params.pattern):
                # Skip ignored directories
                if any(part in ignore_dirs for part in match.parts):
                    continue

                if match.is_file():
                    # Make path relative to search directory
                    try:
                        rel_path = match.relative_to(path)
                        matches.append(str(rel_path))
                    except ValueError:
                        matches.append(str(match))

                if len(matches) >= params.max_results:
                    break

        except Exception as e:
            raise ToolError(f"Error searching: {e}") from e

        if not matches:
            return f"No files found matching: {params.pattern}"

        # Sort results
        matches.sort()

        result = "\n".join(matches)
        if len(matches) >= params.max_results:
            result += f"\n\n[Results limited to {params.max_results} files]"

        return result
