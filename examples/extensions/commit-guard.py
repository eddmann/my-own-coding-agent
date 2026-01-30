from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from agent.extensions.types import ToolCallEvent, ToolCallResult

if TYPE_CHECKING:
    from agent.extensions.api import ExtensionAPI


def _git_status() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout.strip()
        if not output:
            return True, "working tree clean"
        return False, "dirty working tree:\n" + output
    except Exception as exc:
        return False, f"git status failed: {exc}"


def setup(api: ExtensionAPI):
    # Slash command for quick visibility
    api.register_command("dirty", lambda args, ctx: _git_status()[1])

    # Guard commits when repo is dirty
    async def block_dirty_commit(event: ToolCallEvent, ctx):
        if event.tool_name != "bash":
            return None

        cmd = str(event.input.get("command", "")).strip()
        if not cmd:
            return None

        if "git commit" in cmd or "git push" in cmd:
            clean, status = _git_status()
            if not clean:
                return ToolCallResult(block=True, reason=f"dirty working tree:\n{status}")

        return None

    api.on("tool_call", block_dirty_commit)
