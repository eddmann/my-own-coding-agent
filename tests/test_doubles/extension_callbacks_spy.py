"""Test double: spy callback helpers for extension events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExtensionCallbacksSpy:
    """Records extension callback invocations for assertions."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, event: Any, ctx: Any) -> None:
        self.calls.append({"event": event, "context": ctx})
