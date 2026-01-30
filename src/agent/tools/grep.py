"""Search file contents tool."""

import asyncio
import re
from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool, ToolError


class GrepParams(BaseModel):
    """Parameters for grep tool."""

    pattern: str = Field(description="Regex pattern to search for")
    path: str = Field(default=".", description="File or directory to search in")
    include: str | None = Field(default=None, description="Glob pattern for files to include")
    ignore_case: bool = Field(default=False, description="Case insensitive search")
    max_results: int = Field(default=100, description="Maximum number of matches to return")


class GrepTool(BaseTool[GrepParams]):
    """Search file contents using regex patterns."""

    name = "grep"
    description = (
        "Search for a regex pattern in files. "
        "Returns matching lines with file paths and line numbers. "
        "Respects .gitignore by default."
    )
    parameters = GrepParams

    async def execute(self, params: GrepParams) -> str:
        """Execute the grep tool."""
        path = Path(params.path).expanduser()

        if not path.exists():
            raise ToolError(f"Error: Path not found: {params.path}")

        # Compile regex
        flags = re.IGNORECASE if params.ignore_case else 0
        try:
            regex = re.compile(params.pattern, flags)
        except re.error as e:
            raise ToolError(f"Error: Invalid regex pattern: {e}") from e

        # Collect files to search
        files_to_search: list[Path] = []
        if path.is_file():
            files_to_search = [path]
        else:
            # Use rg if available for speed and .gitignore support
            rg_available = await self._check_rg_available()
            if rg_available:
                return await self._grep_with_rg(params)

            # Fall back to Python implementation
            files_to_search = self._collect_files(path, params.include)

        # Search files
        matches: list[str] = []
        for file_path in files_to_search:
            if len(matches) >= params.max_results:
                break

            remaining = params.max_results - len(matches)
            file_matches = await self._search_file(file_path, regex, remaining)
            matches.extend(file_matches)

        if not matches:
            return f"No matches found for pattern: {params.pattern}"

        result = "\n".join(matches)
        if len(matches) >= params.max_results:
            result += f"\n\n[Results limited to {params.max_results} matches]"

        return result

    async def _check_rg_available(self) -> bool:
        """Check if ripgrep is available."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "rg",
                "--version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except Exception:
            return False

    async def _grep_with_rg(self, params: GrepParams) -> str:
        """Use ripgrep for faster searching."""
        cmd = ["rg", "--line-number", "--no-heading", "--color=never"]

        if params.ignore_case:
            cmd.append("-i")

        if params.include:
            cmd.extend(["--glob", params.include])

        cmd.extend(["-m", str(params.max_results)])
        cmd.append(params.pattern)
        cmd.append(params.path)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            output = stdout.decode("utf-8", errors="replace")
            if not output.strip():
                return f"No matches found for pattern: {params.pattern}"

            # Truncate if needed
            if len(output) > 50000:
                output = output[:50000] + "\n\n[Output truncated]"

            return output

        except Exception as e:
            raise ToolError(f"Error running ripgrep: {e}") from e

    def _collect_files(self, directory: Path, include: str | None) -> list[Path]:
        """Collect files to search, respecting patterns."""
        files: list[Path] = []

        if include:
            for file_path in directory.rglob(include):
                if file_path.is_file() and not self._should_ignore(file_path):
                    files.append(file_path)
        else:
            for file_path in directory.rglob("*"):
                if file_path.is_file() and not self._should_ignore(file_path):
                    files.append(file_path)

        return files[:1000]  # Limit total files

    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        ignore_patterns = {
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
            "*.egg-info",
        }

        for part in path.parts:
            if part in ignore_patterns:
                return True
            for pattern in ignore_patterns:
                if "*" in pattern and path.match(pattern):
                    return True

        return False

    async def _search_file(self, path: Path, regex: re.Pattern[str], limit: int) -> list[str]:
        """Search a single file for matches."""
        matches: list[str] = []

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        for i, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                matches.append(f"{path}:{i}: {line[:500]}")
                if len(matches) >= limit:
                    break

        return matches
