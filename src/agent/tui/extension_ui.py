"""TUI screens for extension-owned UI interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from textual import on
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, ListItem, ListView, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.timer import Timer

    from agent.extensions.api import PresentedView, ViewControl

T = TypeVar("T")


class PromptModal(ModalScreen[str | None]):
    """Simple prompt modal for freeform text input."""

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, prompt: str, default: str | None = None) -> None:
        super().__init__()
        self._prompt = prompt
        self._default = default or ""

    def compose(self) -> ComposeResult:
        with Container(id="extension-modal", classes="extension-prompt-modal"):
            yield Static("Extension Input", classes="extension-modal-kicker")
            yield Static(self._prompt, classes="extension-modal-title")
            yield Static(
                "Enter to submit, or escape to cancel.",
                classes="extension-modal-subtitle",
            )
            with Container(classes="extension-input-shell"):
                yield Input(
                    value=self._default,
                    placeholder="Type here and press Enter",
                    id="extension-prompt-input",
                )
            with Horizontal(classes="extension-modal-actions"):
                yield Button("Submit", id="extension-submit-btn", variant="primary")
                yield Button("Cancel", id="extension-cancel-btn")

    async def on_mount(self) -> None:
        self.query_one("#extension-prompt-input", Input).focus()

    def action_close(self) -> None:
        self.dismiss(None)

    @on(Input.Submitted, "#extension-prompt-input")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    @on(Button.Pressed, "#extension-submit-btn")
    def on_submit_pressed(self) -> None:
        value = self.query_one("#extension-prompt-input", Input).value
        self.dismiss(value)

    @on(Button.Pressed, "#extension-cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.dismiss(None)


class ConfirmModal(ModalScreen[bool]):
    """Simple yes/no confirm modal."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("y", "yes", "Yes"),
        Binding("n", "cancel", "No"),
    ]

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        with Container(id="extension-modal"):
            yield Static(self._prompt, classes="extension-modal-title")
            yield Button("Yes", id="extension-yes-btn", variant="primary")
            yield Button("No", id="extension-no-btn")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#extension-yes-btn")
    def on_yes_pressed(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#extension-no-btn")
    def on_no_pressed(self) -> None:
        self.dismiss(False)


class SelectModal(ModalScreen[str | None]):
    """Simple single-select modal."""

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, prompt: str, options: list[str]) -> None:
        super().__init__()
        self._prompt = prompt
        self._options = options

    def compose(self) -> ComposeResult:
        with Container(id="extension-modal"):
            yield Static(self._prompt, classes="extension-modal-title")
            with VerticalScroll(id="extension-select-content"):
                items = [ListItem(Static(option), name=option) for option in self._options]
                yield ListView(*items, id="extension-select-list", initial_index=0)
            yield Button("Cancel", id="extension-cancel-btn")

    async def on_mount(self) -> None:
        self.query_one("#extension-select-list", ListView).focus()

    def action_close(self) -> None:
        self.dismiss(None)

    @on(ListView.Selected, "#extension-select-list")
    def on_selected(self, event: ListView.Selected) -> None:
        if event.item.name:
            self.dismiss(event.item.name)

    @on(Button.Pressed, "#extension-cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.dismiss(None)


class PresentedViewModal[T](ModalScreen[T | None]):
    """Host a small custom extension-provided view."""

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, view: PresentedView[T]) -> None:
        super().__init__()
        self._view = view
        self._controls = view.controls()
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Container(id="extension-modal"):
            with VerticalScroll(id="extension-presented-scroll"):
                yield Static(self._view.render(), id="extension-presented-content")
            for control in self._non_button_controls():
                if control.kind == "input":
                    if control.label:
                        yield Static(control.label, classes="extension-presented-control-label")
                    with Container(classes="extension-input-shell"):
                        yield Input(
                            placeholder=control.placeholder or "Type input and press Enter",
                            id=self._control_id(control),
                        )
                elif control.kind == "select":
                    if control.label:
                        yield Static(control.label, classes="extension-presented-control-label")
                    with VerticalScroll(classes="extension-presented-select-content"):
                        items = [
                            ListItem(Static(option), name=option) for option in control.options
                        ]
                        yield ListView(*items, id=self._control_id(control), initial_index=0)
            buttons = self._button_controls()
            if buttons:
                with Horizontal(classes="extension-modal-actions"):
                    for control in buttons:
                        yield Button(
                            control.label or control.name.title(),
                            id=self._control_id(control),
                            variant="primary" if control.primary else "default",
                        )
            yield Static("Press ESC to close", classes="extension-modal-hint")

    async def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._refresh)
        first = self._first_focusable_control()
        if first is None:
            return
        if first.kind == "input":
            self.query_one(f"#{self._control_id(first)}", Input).focus()
        elif first.kind == "select":
            self.query_one(f"#{self._control_id(first)}", ListView).focus()
        elif first.kind == "button":
            self.query_one(f"#{self._control_id(first)}", Button).focus()

    def action_close(self) -> None:
        self.dismiss(None)

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        control = self._control_by_id(event.input.id)
        if control is None:
            return
        self._dispatch(control, event.value)
        event.input.value = ""

    @on(ListView.Selected)
    def on_list_selected(self, event: ListView.Selected) -> None:
        control = self._control_by_id(event.list_view.id)
        if control is None:
            return
        value = event.item.name if isinstance(event.item.name, str) else None
        self._dispatch(control, value)

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        control = self._control_by_id(event.button.id)
        if control is None:
            return
        self._dispatch(control, None)

    def _dispatch(self, control: ViewControl, value: str | None) -> None:
        self._view.handle_action(control.name, value)
        self._finish_if_done()
        self._refresh()

    def _control_id(self, control: ViewControl) -> str:
        return f"extension-presented-{control.kind}-{control.name}"

    def _control_by_id(self, control_id: str | None) -> ViewControl | None:
        if control_id is None:
            return None
        for control in self._controls:
            if self._control_id(control) == control_id:
                return control
        return None

    def _non_button_controls(self) -> list[ViewControl]:
        return [control for control in self._controls if control.kind != "button"]

    def _button_controls(self) -> list[ViewControl]:
        return [control for control in self._controls if control.kind == "button"]

    def _first_focusable_control(self) -> ViewControl | None:
        for kind in ("input", "select", "button"):
            for control in self._controls:
                if control.kind == kind:
                    return control
        return None

    def _refresh(self) -> None:
        self.query_one("#extension-presented-content", Static).update(self._view.render())
        self._finish_if_done()

    def _finish_if_done(self) -> None:
        if self._view.is_done():
            if self._timer is not None:
                self._timer.stop()
                self._timer = None
            self.dismiss(self._view.result())
