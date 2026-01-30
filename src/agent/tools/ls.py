"""List directory contents tool."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool, ToolError


class LsParams(BaseModel):
    """Parameters for ls tool."""

    path: str = Field(default=".", description="Directory to list")
    all: bool = Field(default=False, description="Include hidden files (starting with .)")
    long: bool = Field(default=False, description="Use long listing format with details")


class LsTool(BaseTool[LsParams]):
    """List directory contents."""

    name = "ls"
    description = (
        "List directory contents. "
        "Use all=true to include hidden files. "
        "Use long=true for detailed listing with sizes and dates."
    )
    parameters = LsParams

    async def execute(self, params: LsParams) -> str:
        """Execute the ls tool."""
        path = Path(params.path).expanduser()

        if not path.exists():
            raise ToolError(f"Error: Path not found: {params.path}")

        if not path.is_dir():
            raise ToolError(f"Error: Not a directory: {params.path}")

        try:
            entries = list(path.iterdir())
        except PermissionError as err:
            raise ToolError(f"Error: Permission denied: {params.path}") from err

        # Filter hidden files unless --all
        if not params.all:
            entries = [e for e in entries if not e.name.startswith(".")]

        # Sort: directories first, then alphabetically
        entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))

        if not entries:
            return f"Directory is empty: {params.path}"

        if params.long:
            return self._format_long(entries)
        else:
            return self._format_short(entries)

    def _format_short(self, entries: list[Path]) -> str:
        """Format as simple list."""
        lines = []
        for entry in entries:
            name = entry.name
            if entry.is_dir():
                name += "/"
            elif entry.is_symlink():
                name += "@"
            lines.append(name)
        return "\n".join(lines)

    def _format_long(self, entries: list[Path]) -> str:
        """Format as detailed listing."""
        lines = []

        for entry in entries:
            try:
                stat = entry.stat()
                size = self._format_size(stat.st_size)
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

                # Type indicator
                if entry.is_dir():
                    type_char = "d"
                    name = entry.name + "/"
                elif entry.is_symlink():
                    type_char = "l"
                    name = f"{entry.name} -> {entry.resolve()}"
                else:
                    type_char = "-"
                    name = entry.name

                lines.append(f"{type_char} {size:>8} {mtime} {name}")

            except (OSError, ValueError):
                lines.append(f"? {'?':>8} {'?':>16} {entry.name}")

        return "\n".join(lines)

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable form."""
        size_f = float(size)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size_f < 1024:
                if unit == "B":
                    return f"{int(size_f)}{unit}"
                return f"{size_f:.1f}{unit}"
            size_f /= 1024
        return f"{size_f:.1f}PB"
