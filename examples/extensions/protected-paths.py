from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agent.extensions.types import ToolCallEvent, ToolCallResult

if TYPE_CHECKING:
    from agent.extensions.api import ExtensionAPI

PROTECTED_PARTS = {".git", ".agent", ".env", "secrets"}


def _is_protected_path(path_str: str) -> bool:
    if not path_str:
        return False
    try:
        parts = Path(path_str).expanduser().parts
    except Exception:
        parts = Path(path_str).parts
    return any(part in PROTECTED_PARTS for part in parts)


def setup(api: ExtensionAPI):
    async def block_protected(event: ToolCallEvent, ctx):
        if event.tool_name in {"write", "edit"}:
            path = str(event.input.get("path", ""))
            if _is_protected_path(path):
                return ToolCallResult(block=True, reason=f"protected path: {path}")

        if event.tool_name == "bash":
            cmd = str(event.input.get("command", ""))
            if any(part in cmd for part in PROTECTED_PARTS):
                return ToolCallResult(block=True, reason="protected path in command")

    api.on("tool_call", block_protected)
