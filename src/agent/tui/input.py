"""User input widget with dropdown autocomplete."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Key

    from agent.prompts.loader import PromptTemplateLoader
    from agent.skills.loader import SkillLoader


# Built-in commands available for autocomplete
BUILTIN_COMMANDS = [
    "/clear",
    "/new",
    "/load",
    "/resume",
    "/fork",
    "/tree",
    "/context",
    "/help",
    "/model",
    "/quit",
]


class PromptInput(Widget, can_focus=False):
    """Input with dropdown command autocomplete."""

    class Submitted(Message):
        """Posted when user submits input."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

        @property
        def control(self) -> PromptInput:
            return self._sender  # type: ignore

    def __init__(
        self,
        skill_loader: SkillLoader | None = None,
        template_loader: PromptTemplateLoader | None = None,
        extension_commands: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._skill_loader = skill_loader
        self._template_loader = template_loader
        self._extension_commands = extension_commands or []
        self._history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self._suppress_history_reset = False

    def _set_input_value(self, value: str, *, reset_history: bool) -> None:
        input_widget = self.query_one("#prompt-inner", Input)
        self._suppress_history_reset = True
        input_widget.value = value
        with contextlib.suppress(Exception):
            input_widget.cursor_position = len(value)
        if reset_history:
            self._history_index = None
            self._history_draft = value

    def _get_slash_commands(self) -> list[str]:
        """Build list of slash commands (templates, built-ins, extensions)."""
        commands = list(BUILTIN_COMMANDS)

        # Add template names as commands
        if self._template_loader:
            for name in self._template_loader.list_templates():
                commands.append(f"/{name}")

        # Add extension commands
        for name in self._extension_commands:
            commands.append(f"/{name}")

        return sorted(set(commands))

    def _get_skill_commands(self) -> list[str]:
        """Build list of skill commands ($skill-name)."""
        commands: list[str] = []
        if self._skill_loader:
            for skill in self._skill_loader.skills.values():
                commands.append(f"${skill.name}")
        return sorted(set(commands))

    def set_extension_commands(self, commands: list[str]) -> None:
        """Update available extension commands."""
        self._extension_commands = commands

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type here...", id="prompt-inner")
        yield OptionList(id="suggestions")

    def on_mount(self) -> None:
        """Hide suggestions on mount."""
        self.query_one("#suggestions", OptionList).display = False

    def on_input_changed(self, event: Input.Changed) -> None:
        """Show/hide suggestions based on input."""
        option_list = self.query_one("#suggestions", OptionList)
        value = event.value
        if self._suppress_history_reset:
            self._suppress_history_reset = False
        else:
            if self._history_index is not None:
                self._history_index = None
            self._history_draft = value

        if value.startswith("/"):
            # Filter commands matching input
            query = value.lower()
            all_commands = self._get_slash_commands()
            filtered = [cmd for cmd in all_commands if cmd.lower().startswith(query)]
        elif value.startswith("$"):
            query = value.lower()
            all_commands = self._get_skill_commands()
            filtered = [cmd for cmd in all_commands if cmd.lower().startswith(query)]
        else:
            filtered = []

        if filtered:
            option_list.clear_options()
            option_list.add_options([Option(cmd) for cmd in filtered])
            option_list.highlighted = 0
            # Position dropdown above input - adjust offset based on item count
            # Each item is 1 row, plus 2 for border, plus gap to reach -13 at max
            item_count = min(len(filtered), 10)  # max-height is 10
            offset_y = -(item_count + 7)
            option_list.styles.offset = (0, offset_y)
            option_list.display = True
        else:
            option_list.display = False

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key - select suggestion or submit."""
        event.stop()
        option_list = self.query_one("#suggestions", OptionList)

        if option_list.display and option_list.highlighted is not None:
            # Select highlighted option
            option = option_list.get_option_at_index(option_list.highlighted)
            if option:
                value = str(option.prompt)
                if not value.endswith(" "):
                    value += " "
                event.input.value = value
                with contextlib.suppress(Exception):
                    event.input.cursor_position = len(value)
                option_list.display = False
                return

        # Submit the input
        value = event.value.strip()
        if value:
            lower = value.lower()
            excluded = {
                "/clear",
                "/new",
                "/load",
                "/resume",
                "/fork",
                "/tree",
                "/quit",
                "/help",
                "/context",
                "/model",
            }
            if lower.startswith("/model "):
                excluded.add(lower.split(" ", 1)[0])
            if lower not in excluded and (not self._history or value != self._history[-1]):
                self._history.append(value)
            self._history_index = None
            self._history_draft = ""
            self.post_message(self.Submitted(value))

    def on_key(self, event: Key) -> None:
        """Handle arrow keys for suggestion navigation."""
        option_list = self.query_one("#suggestions", OptionList)

        def apply_suggestion(trailing_space: bool) -> None:
            if option_list.highlighted is None:
                return
            option = option_list.get_option_at_index(option_list.highlighted)
            if not option:
                return
            value = str(option.prompt)
            if trailing_space and not value.endswith(" "):
                value += " "
            self._set_input_value(value, reset_history=True)
            option_list.display = False

        if not option_list.display:
            if event.key in {"up", "down"}:
                if not self._history:
                    return
                event.stop()
                input_widget = self.query_one("#prompt-inner", Input)
                if event.key == "up":
                    if self._history_index is None:
                        self._history_draft = input_widget.value
                        self._history_index = len(self._history) - 1
                    elif self._history_index > 0:
                        self._history_index -= 1
                    value = self._history[self._history_index]
                    self._set_input_value(value, reset_history=False)
                else:
                    if self._history_index is None:
                        return
                    if self._history_index < len(self._history) - 1:
                        self._history_index += 1
                        value = self._history[self._history_index]
                        self._set_input_value(value, reset_history=False)
                    else:
                        self._history_index = None
                        self._set_input_value(self._history_draft, reset_history=False)
            return

        if event.key == "up":
            event.stop()
            if option_list.highlighted is not None and option_list.highlighted > 0:
                option_list.highlighted -= 1
            elif option_list.option_count > 0:
                option_list.highlighted = option_list.option_count - 1

        elif event.key == "down":
            event.stop()
            if option_list.highlighted is not None:
                if option_list.highlighted < option_list.option_count - 1:
                    option_list.highlighted += 1
                else:
                    option_list.highlighted = 0
            elif option_list.option_count > 0:
                option_list.highlighted = 0

        elif event.key == "tab":
            event.stop()
            # Tab selects the current suggestion
            apply_suggestion(trailing_space=False)

        elif event.key == "escape":
            event.stop()
            option_list.display = False

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle mouse click on suggestion."""
        event.stop()
        self._set_input_value(str(event.option.prompt), reset_history=True)
        self.query_one("#suggestions", OptionList).display = False
        self.query_one("#prompt-inner", Input).focus()

    def focus(self, scroll_visible: bool = True) -> PromptInput:
        """Focus the inner input."""
        self.query_one("#prompt-inner", Input).focus(scroll_visible)
        return self

    def clear(self) -> None:
        """Clear the input."""
        self.query_one("#prompt-inner", Input).value = ""
