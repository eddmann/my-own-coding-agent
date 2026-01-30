"""Behavior tests for the status bar."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from agent.core.config import ThinkingLevel
from agent.tui.status import StatusBar


class StatusApp(App[None]):
    def compose(self) -> ComposeResult:
        yield StatusBar(id="status")


@pytest.mark.asyncio
async def test_status_bar_displays_model_thinking_and_tokens():
    app = StatusApp()

    async with app.run_test() as pilot:
        await pilot.pause()

        status = app.query_one("#status", StatusBar)

        status.set_model("gpt-4o")
        status.set_thinking(ThinkingLevel.LOW)
        status.set_tokens(50, 100)

        left = status.query_one("#status-left")
        # Note: render() access is a UI-layer hook; no public text getter exists.
        renderable = left.render()
        text = getattr(renderable, "plain", str(renderable))

        assert "gpt-4o" in text
        assert "thinking:low" in text
        assert "50/100" in text
