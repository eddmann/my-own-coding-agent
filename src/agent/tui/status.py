"""Status bar widget showing model and working directory."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from textual.containers import Horizontal
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from agent.core.config import ThinkingLevel


class StatusBar(Horizontal):
    """Status bar with left (model + tokens) and right (cwd) sections."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model = "unknown"
        self._thinking = "off"
        self._tokens = 0
        self._max_tokens = 0

    def compose(self) -> ComposeResult:
        yield Static(id="status-left")
        yield Static(id="status-right")

    def on_mount(self) -> None:
        self._update_display()

    def _update_display(self) -> None:
        left = self._model

        # Show thinking level if not off
        if self._thinking != "off":
            left += f" [thinking:{self._thinking}]"

        # Add token usage next to model
        if self._max_tokens > 0:
            pct = (self._tokens / self._max_tokens) * 100
            left += f"  {self._tokens:,}/{self._max_tokens:,} ({pct:.0f}%)"
        elif self._tokens > 0:
            left += f"  {self._tokens:,} tokens"

        self.query_one("#status-left", Static).update(left)
        self.query_one("#status-right", Static).update(os.getcwd())

    def set_model(self, model: str) -> None:
        """Set the model name."""
        self._model = model
        self._update_display()

    def set_thinking(self, level: ThinkingLevel) -> None:
        """Set the thinking level display."""
        self._thinking = level.value
        self._update_display()

    def set_tokens(self, tokens: int, max_tokens: int = 0) -> None:
        """Set the token count."""
        self._tokens = tokens
        self._max_tokens = max_tokens
        self._update_display()
