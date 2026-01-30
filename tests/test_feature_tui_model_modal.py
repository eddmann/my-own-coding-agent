"""Behavior tests for the model modal."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, RadioButton, RadioSet, Select, Static

from agent.core.config import Config, ThinkingLevel
from agent.tui.model_modal import ModelModal
from tests.fakes import FakeLLMProvider


class ModelApp(App[None]):
    def __init__(self, modal: ModelModal) -> None:
        super().__init__()
        self._modal = modal

    def on_mount(self) -> None:
        self.push_screen(self._modal)


def get_text(static: Static) -> str:
    # Note: render() access is a UI-layer hook; no public text getter exists.
    renderable = static.render()
    return getattr(renderable, "plain", str(renderable))


@pytest.mark.asyncio
async def test_model_modal_updates_summary_and_calls_callback(temp_dir):
    config = Config(
        provider="openai",
        model="gpt-4o",
        api_key="test",
        session_dir=temp_dir,
        thinking_level=ThinkingLevel.OFF,
    )

    provider = FakeLLMProvider([])
    provider.model = "gpt-4o"

    async def _list_models():
        return ["gpt-4o", "gpt-5"]

    provider.list_models = _list_models

    callback_args = []

    def on_change(model, thinking):
        callback_args.append((model, thinking))

    modal = ModelModal(config, provider, on_change)
    app = ModelApp(modal)

    async with app.run_test() as pilot:
        await pilot.pause()

        select = modal.query_one("#model-select", Select)
        select.value = "gpt-5"
        await pilot.pause()

        summary = modal.query_one("#summary-info", Static)
        summary_text = get_text(summary)
        assert "Model: gpt-5" in summary_text

        radio = modal.query_one("#thinking-radio", RadioSet)
        target = None
        for button in radio.query(RadioButton):
            if button.name == "low" and not button.disabled:
                target = button
                break

        assert target is not None
        target.toggle()
        await pilot.pause()

        summary_text = get_text(summary)
        assert "Thinking: low" in summary_text

        save = modal.query_one("#save-btn", Button)
        await pilot.click(save)

        assert callback_args
        assert callback_args[-1][0] == "gpt-5"
        assert callback_args[-1][1] == ThinkingLevel.LOW
