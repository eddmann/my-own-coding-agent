"""Execute shell commands tool."""

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool, ToolError


class BashParams(BaseModel):
    """Parameters for bash tool."""

    command: str = Field(description="The shell command to execute")
    cwd: str | None = Field(default=None, description="Working directory for the command")
    timeout: int = Field(default=120, description="Timeout in seconds (max 600)")


class BashTool(BaseTool[BashParams]):
    """Execute shell commands."""

    name = "bash"
    description = (
        "Execute a shell command. "
        "Returns stdout and stderr. "
        "Use cwd to set working directory. "
        "Commands timeout after 120 seconds by default (max 600)."
    )
    parameters = BashParams

    async def execute(self, params: BashParams) -> str:
        """Execute the bash command."""
        # Validate timeout
        timeout = min(max(params.timeout, 1), 600)

        # Determine working directory
        cwd = Path(params.cwd).expanduser() if params.cwd else Path.cwd()
        if not cwd.exists():
            raise ToolError(f"Error: Working directory not found: {cwd}")

        try:
            process = await asyncio.create_subprocess_shell(
                params.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except TimeoutError as err:
                process.kill()
                await process.wait()
                raise ToolError(f"Error: Command timed out after {timeout} seconds") from err

            # Decode output
            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")

            # Build result
            result_parts = []

            if stdout_text:
                # Truncate very long output
                if len(stdout_text) > 30000:
                    stdout_text = stdout_text[:30000] + "\n\n[Output truncated at 30000 chars]"
                result_parts.append(stdout_text)

            if stderr_text:
                if len(stderr_text) > 10000:
                    stderr_text = stderr_text[:10000] + "\n\n[Stderr truncated at 10000 chars]"
                result_parts.append(f"[stderr]\n{stderr_text}")

            if process.returncode != 0:
                result_parts.append(f"\n[Exit code: {process.returncode}]")

            if not result_parts:
                return f"Command completed with exit code {process.returncode}"

            return "\n".join(result_parts)

        except Exception as e:
            raise ToolError(f"Error executing command: {e}") from e
