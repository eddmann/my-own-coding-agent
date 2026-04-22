"""Bridge extension UI callbacks into Textual UI interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.widgets import Static

from agent.extensions.api import ExtensionUIBindings
from agent.tui.chat import ChatView
from agent.tui.extension_ui import ConfirmModal, PresentedViewModal, PromptModal, SelectModal
from agent.tui.status import StatusBar

if TYPE_CHECKING:
    from agent.extensions import PresentedView, WidgetView
    from agent.tui.app import AgentApp


class TUIExtensionBridge:
    """Bind extension-owned UI surfaces onto the Textual app."""

    __slots__ = ("_app", "_widgets")

    def __init__(self, app: AgentApp) -> None:
        self._app = app
        self._widgets: dict[str, WidgetView | None] = {
            "footer": None,
            "right_panel": None,
        }

    def bindings(self) -> ExtensionUIBindings:
        """Return host UI bindings for the extension host."""
        return ExtensionUIBindings(
            notify=self.notify,
            set_status=self.set_status,
            input=self.input,
            confirm=self.confirm,
            select=self.select,
            present=self.present,
            set_widget=self.set_widget,
        )

    def notify(self, message: str, level: str) -> None:
        """Render extension notifications in the chat view."""
        try:
            chat = self._app.query_one("#chat-view", ChatView)
        except Exception:
            return
        prefix = f"[{level}] " if level != "info" else ""
        chat.add_system_message(f"{prefix}{message}")

    def set_status(self, text: str | None) -> None:
        """Update extension status text in the status bar."""
        try:
            status = self._app.query_one("#status-line", StatusBar)
        except Exception:
            return
        status.set_extension_status(text)

    async def input(self, prompt: str, default: str | None = None) -> str | None:
        """Prompt for freeform text from the TUI."""
        return cast(
            "str | None",
            await self._app.push_extension_screen(PromptModal(prompt, default)),
        )

    async def confirm(self, prompt: str) -> bool:
        """Prompt for confirmation in the TUI."""
        return bool(await self._app.push_extension_screen(ConfirmModal(prompt)))

    async def select(self, prompt: str, options: list[str]) -> str | None:
        """Prompt to select from a list in the TUI."""
        return cast(
            "str | None",
            await self._app.push_extension_screen(SelectModal(prompt, options)),
        )

    async def present(self, view: PresentedView[object]) -> object | None:
        """Present a temporary custom extension view."""
        return await self._app.push_extension_screen(PresentedViewModal(view))

    def set_widget(self, slot: str, view: WidgetView | None) -> None:
        """Set or clear a persistent extension widget."""
        if slot not in self._widgets:
            raise ValueError(f"Unknown widget slot: {slot}")
        self._widgets[slot] = view
        self.render_widgets()

    def render_widgets(self) -> None:
        """Render extension widgets into their persistent slots."""
        try:
            footer = self._app.query_one("#extension-footer", Static)
            right_panel = self._app.query_one("#extension-right-panel", Static)
        except Exception:
            return

        self._render_slot(footer, self._widgets.get("footer"))
        self._render_slot(right_panel, self._widgets.get("right_panel"))

    def _render_slot(self, widget: Static, view: WidgetView | None) -> None:
        """Render a single extension widget slot."""
        if view is None:
            widget.update("")
            widget.add_class("hidden")
            return

        try:
            content = view.render()
        except Exception as exc:
            content = f"extension widget error: {exc}"

        widget.update(content)
        widget.remove_class("hidden")
