"""Model modal - configure model and thinking settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, RadioButton, RadioSet, Select, Static

from agent.core.settings import ThinkingLevel, get_available_thinking_levels
from agent.llm.models import get_model_info, resolve_capability_provider, supports_reasoning

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.app import ComposeResult

    from agent.core.settings import AgentSettings
    from agent.llm.provider import LLMProvider


class ModelModal(ModalScreen[None]):
    """Modal for configuring model and thinking settings."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
    ]

    def __init__(
        self,
        config: AgentSettings,
        provider: LLMProvider,
        on_change: Callable[[str | None, ThinkingLevel | None], None],
    ) -> None:
        super().__init__()
        self._config = config
        self._provider = provider
        self._on_change = on_change
        self._available_models: list[str] = []
        self._pending_model: str | None = None
        self._pending_thinking: ThinkingLevel | None = None

    def compose(self) -> ComposeResult:
        with Container(id="model-modal"):
            yield Static("Model Settings", id="model-title")
            with VerticalScroll(id="model-content"):
                # Model selection section
                yield Static("[bold cyan]SELECT MODEL[/]", classes="section-header")
                yield Select(
                    [(self._provider.model, self._provider.model)],
                    value=self._provider.model,
                    id="model-select",
                    allow_blank=False,
                )

                # Thinking level section
                yield Static("[bold cyan]THINKING LEVEL[/]", classes="section-header")
                yield self._build_thinking_radio()

                # Summary section
                yield Static("[bold cyan]SUMMARY[/]", classes="section-header")
                yield Static(self._build_summary(), id="summary-info")

                # Save button
                yield Button("Save", id="save-btn", variant="primary")

            yield Static("Press [bold]ESC[/] to close", id="model-hint")

    async def on_mount(self) -> None:
        """Fetch models on mount."""
        await self._refresh_models()

    def _build_thinking_radio(self) -> RadioSet:
        """Build thinking level radio set."""
        provider_hint = resolve_capability_provider(self._provider.name)
        available = get_available_thinking_levels(self._provider.model, provider=provider_hint)

        buttons = []
        for level in ThinkingLevel:
            is_available = level in available
            label = level.value
            if not is_available:
                label += " (unavailable)"
            buttons.append(
                RadioButton(
                    label,
                    disabled=not is_available,
                    name=level.value,
                )
            )
        return RadioSet(*buttons, id="thinking-radio")

    def _build_summary(self) -> str:
        """Build summary of current and pending settings."""
        lines = []

        current_model = self._provider.model
        current_thinking = self._config.thinking_level

        new_model = self._pending_model or current_model
        new_thinking = self._pending_thinking or current_thinking

        model_info = get_model_info(new_model)
        if model_info:
            reasoning_support = "Yes" if model_info.reasoning else "No"
            max_tokens = f"{model_info.max_output_tokens:,}"
        else:
            provider_hint = resolve_capability_provider(self._provider.name)
            reasoning = supports_reasoning(new_model, provider_hint)
            reasoning_support = "Yes" if reasoning else "No"
            max_tokens = "(unknown)"

        lines.append(f"  Provider: {self._provider.name}")
        lines.append(f"  Model: {new_model}")
        lines.append(f"  Thinking: {new_thinking.value}")
        lines.append(f"  Supports Reasoning: {reasoning_support}")
        lines.append(f"  Max Output Tokens: {max_tokens}")

        return "\n".join(lines)

    async def _refresh_models(self) -> None:
        """Fetch available models from provider and update dropdown."""
        select = self.query_one("#model-select", Select)

        self._available_models = await self._provider.list_models()

        current = self._provider.model
        options: list[tuple[str, str]] = []

        if self._available_models:
            for model in sorted(self._available_models):
                options.append((model, model))
            if current not in self._available_models:
                options.insert(0, (f"{current} (current)", current))
        else:
            options.append((current, current))

        select.set_options(options)
        select.value = current

    def _update_summary(self) -> None:
        """Update the summary display."""
        self.query_one("#summary-info", Static).update(self._build_summary())

    async def _rebuild_thinking_radio(self, model: str) -> None:
        """Rebuild thinking radio for a new model."""
        provider_hint = resolve_capability_provider(self._provider.name)
        available = get_available_thinking_levels(model, provider=provider_hint)

        buttons = []
        for level in ThinkingLevel:
            is_available = level in available
            label = level.value
            if not is_available:
                label += " (unavailable)"
            buttons.append(
                RadioButton(
                    label,
                    disabled=not is_available,
                    name=level.value,
                )
            )

        old_radio = self.query_one("#thinking-radio", RadioSet)
        new_radio = RadioSet(*buttons, id="thinking-radio")

        await old_radio.remove()
        headers = list(self.query(".section-header"))
        if len(headers) >= 2:
            await headers[1].mount(new_radio, after=headers[1])

    def action_close(self) -> None:
        """Close the modal."""
        self.app.pop_screen()

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle model selection change."""
        if event.select.id == "model-select" and event.value:
            new_model = str(event.value)
            if new_model != self._provider.model:
                self._pending_model = new_model
            else:
                self._pending_model = None
            await self._rebuild_thinking_radio(new_model)
            self._update_summary()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle thinking level change."""
        if event.radio_set.id == "thinking-radio" and event.pressed:
            level_name = event.pressed.name
            if level_name:
                new_level = ThinkingLevel(level_name)
                self._pending_thinking = new_level
                self._update_summary()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save-btn":
            if self._pending_model or self._pending_thinking:
                self._on_change(self._pending_model, self._pending_thinking)
            self.app.pop_screen()
