from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.extensions.api import ExtensionAPI


def setup(api: ExtensionAPI):
    def add_todo(args: str, ctx) -> str:
        text = args.strip()
        if not text:
            return "Usage: /todo <item>"

        path = Path("TODO.md")
        line = f"- [ ] {text}\n"
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            path.write_text(existing + line, encoding="utf-8")
        else:
            path.write_text("# TODO\n\n" + line, encoding="utf-8")
        return f"Added to TODO.md: {text}"

    api.register_command("todo", add_todo)
