"""Bridge extension UI callbacks into WebSocket events."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from agent.extensions.api import ExtensionUIBindings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agent.extensions import PresentedView, ViewControl, WidgetView


@dataclass(slots=True)
class _PresentedViewState:
    """Active presented view tracked by the bridge."""

    view: PresentedView[object]
    future: asyncio.Future[object | None]


class WebExtensionBridge:
    """Bind extension-owned UI surfaces onto a WebSocket client."""

    __slots__ = ("_presented_views", "_sender", "_ui_requests")

    def __init__(self, sender: Callable[[dict[str, object]], Awaitable[None]]) -> None:
        self._sender = sender
        self._ui_requests: dict[str, asyncio.Future[str | bool | None]] = {}
        self._presented_views: dict[str, _PresentedViewState] = {}

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

    async def notify(self, message: str, level: str) -> None:
        """Send an extension notification to the browser."""
        await self._sender(
            {
                "type": "ui.notify",
                "level": level,
                "message": message,
            }
        )

    async def set_status(self, text: str | None) -> None:
        """Update extension status text."""
        await self._sender({"type": "ui.status", "text": text})

    async def input(self, prompt: str, default: str | None = None) -> str | None:
        """Prompt for freeform text."""
        value = await self._request("input", prompt, default=default)
        return value if isinstance(value, str) or value is None else str(value)

    async def confirm(self, prompt: str) -> bool:
        """Prompt for confirmation."""
        value = await self._request("confirm", prompt)
        return bool(value)

    async def select(self, prompt: str, options: list[str]) -> str | None:
        """Prompt for a single selection."""
        value = await self._request("select", prompt, options=options)
        return value if isinstance(value, str) or value is None else str(value)

    async def present(self, view: PresentedView[object]) -> object | None:
        """Present a temporary custom extension view."""
        view_id = uuid4().hex
        future: asyncio.Future[object | None] = asyncio.get_running_loop().create_future()
        self._presented_views[view_id] = _PresentedViewState(view=view, future=future)
        await self._send_presented_view(view_id, view)

        if view.is_done():
            result = view.result()
            future.set_result(result)
            del self._presented_views[view_id]
            await self._sender({"type": "ui.view_close", "view_id": view_id})
            return result

        return await future

    async def set_widget(self, slot: str, view: WidgetView | None) -> None:
        """Set or clear a persistent extension widget."""
        if view is None:
            await self._sender({"type": "ui.widget", "slot": slot, "content": None})
            return

        try:
            content = view.render()
        except Exception as exc:
            content = f"extension widget error: {exc}"

        await self._sender(
            {
                "type": "ui.widget",
                "slot": slot,
                "content": content,
            }
        )

    def resolve_request(self, request_id: str, value: str | bool | None) -> None:
        """Resolve one pending interactive UI request."""
        future = self._ui_requests.pop(request_id, None)
        if future is None or future.done():
            return
        future.set_result(value)

    async def handle_view_action(self, view_id: str, action: str, value: str | None = None) -> None:
        """Apply an action to one active presented view."""
        state = self._presented_views.get(view_id)
        if state is None:
            return

        state.view.handle_action(action, value)
        if state.view.is_done():
            result = state.view.result()
            if not state.future.done():
                state.future.set_result(result)
            del self._presented_views[view_id]
            await self._sender({"type": "ui.view_close", "view_id": view_id})
            return

        await self._send_presented_view(view_id, state.view)

    async def close(self) -> None:
        """Cancel any pending interactive requests when the client disconnects."""
        for future in self._ui_requests.values():
            if not future.done():
                future.cancel()
        self._ui_requests.clear()

        for state in self._presented_views.values():
            if not state.future.done():
                state.future.cancel()
        self._presented_views.clear()

    async def _request(
        self,
        kind: str,
        prompt: str,
        *,
        default: str | None = None,
        options: list[str] | None = None,
    ) -> str | bool | None:
        request_id = uuid4().hex
        future: asyncio.Future[str | bool | None] = asyncio.get_running_loop().create_future()
        self._ui_requests[request_id] = future

        payload: dict[str, object] = {
            "type": "ui.request",
            "request_id": request_id,
            "kind": kind,
            "prompt": prompt,
        }
        if default is not None:
            payload["default"] = default
        if options is not None:
            payload["options"] = [{"label": option, "value": option} for option in options]

        await self._sender(payload)
        return await future

    async def _send_presented_view(self, view_id: str, view: PresentedView[object]) -> None:
        await self._sender(
            {
                "type": "ui.view_present",
                "view_id": view_id,
                "title": self._view_title(view),
                "body": view.render(),
                "controls": [self._serialize_control(control) for control in view.controls()],
            }
        )

    def _serialize_control(self, control: ViewControl) -> dict[str, object]:
        return {
            "kind": control.kind,
            "name": control.name,
            "label": control.label,
            "placeholder": control.placeholder,
            "options": list(control.options),
            "primary": control.primary,
        }

    def _view_title(self, view: PresentedView[object]) -> str:
        name = type(view).__name__
        if name.endswith("View") and len(name) > 4:
            name = name[:-4]
        return re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip() or "Extension View"
