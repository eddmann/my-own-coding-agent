"""TUI delivery controller for commands and runtime actions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from agent.runtime.message import Role
from agent.runtime.session import MessageEntry, Session
from agent.runtime.settings import ThinkingLevel
from agent.tui.chat import ChatView
from agent.tui.context_modal import ContextModal
from agent.tui.model_modal import ModelModal
from agent.tui.session_modal import SessionForkModal, SessionLoadModal, SessionTreeModal
from agent.tui.status import StatusBar

if TYPE_CHECKING:
    from agent.tui.app import AgentApp


class TUIController:
    """Own delivery commands and runtime control actions for the TUI."""

    __slots__ = ("_app",)

    def __init__(self, app: AgentApp) -> None:
        self._app = app

    async def load_session(self, session: Session, note: str | None = None) -> None:
        """Load a session into the agent and re-render UI."""
        await self._app.agent.load_session(session)
        self._app._session = session
        self._app._renderer.render_loaded_session(session, note=note)

    async def load_session_path(self, path: Path) -> None:
        """Load a session from a specific file path."""
        chat = self._app.query_one("#chat-view", ChatView)
        if not path.exists():
            chat.add_system_message(f"session file not found: {path}")
            return
        try:
            session = Session.load(path)
        except Exception as exc:
            chat.add_system_message(f"failed to load session: {exc}")
            return
        await self.load_session(session, note=f"loaded session {session.metadata.id}")

    def queue_load_session_path(self, path: Path) -> None:
        """Queue a session-load task for modal callbacks."""
        asyncio.create_task(self.load_session_path(path))

    def resolve_message_id(self, session: Session, spec: str) -> str | None:
        """Resolve a message id from a spec (id, prefix, index, last)."""
        messages = session.messages
        if not messages:
            return None
        spec = spec.strip()
        if not spec or spec.lower() in {"last", "latest"}:
            return messages[-1].id
        if spec.lower() in {"assistant", "last-assistant"}:
            for msg in reversed(messages):
                if msg.role == Role.ASSISTANT:
                    return msg.id
            return messages[-1].id
        if spec.isdigit():
            idx = int(spec)
            if 0 <= idx < len(messages):
                return messages[idx].id
        for msg in messages:
            if msg.id == spec:
                return msg.id
        matches = [msg for msg in messages if msg.id.startswith(spec)]
        if len(matches) == 1:
            return matches[0].id
        return None

    async def fork_from_message(self, message_id: str) -> None:
        """Fork the current session from a given message id."""
        chat = self._app.query_one("#chat-view", ChatView)
        parent_id = self._app.agent.session_id
        try:
            new_session = self._app.agent.fork_session(message_id)
        except Exception as exc:
            chat.add_system_message(f"failed to fork session: {exc}")
            return
        note = f"forked from {parent_id} at {message_id}"
        await self.load_session(new_session, note=note)

    def queue_fork_from_message(self, message_id: str) -> None:
        """Queue a fork task for modal callbacks."""
        asyncio.create_task(self.fork_from_message(message_id))

    async def set_leaf(self, message_id: str) -> None:
        """Move the active leaf to a given message id."""
        chat = self._app.query_one("#chat-view", ChatView)
        try:
            self._app.agent.set_leaf(message_id)
        except Exception as exc:
            chat.add_system_message(f"failed to set leaf: {exc}")
            return
        self._app._session = self._app.agent.session
        self._app._renderer.render_loaded_session(
            self._app.agent.session,
            note=f"branched to {message_id}",
        )

    def queue_set_leaf(self, message_id: str) -> None:
        """Queue a leaf-switch task for modal callbacks."""
        asyncio.create_task(self.set_leaf(message_id))

    def resolve_entry_id(self, session: Session, spec: str) -> str | None:
        """Resolve an entry id across all message entries in the session."""
        entries = list(session.entries)
        if not entries:
            return None
        spec = spec.strip()
        if not spec or spec.lower() in {"last", "latest"}:
            return entries[-1].id
        if spec.lower() in {"assistant", "last-assistant"}:
            for entry in reversed(entries):
                if isinstance(entry, MessageEntry) and entry.message.role == Role.ASSISTANT:
                    return entry.id
            return entries[-1].id
        for entry in entries:
            if entry.id == spec:
                return entry.id
        matches = [entry for entry in entries if entry.id.startswith(spec)]
        if len(matches) == 1:
            return matches[0].id
        if spec.isdigit():
            msg_entries = [entry for entry in entries if isinstance(entry, MessageEntry)]
            idx = int(spec)
            if 0 <= idx < len(msg_entries):
                return msg_entries[idx].id
        return None

    async def handle_prompt_command(self, prompt: str) -> bool:
        """Handle TUI-native prompt commands."""
        chat = self._app.query_one("#chat-view", ChatView)

        if prompt.lower() == "/clear":
            self.action_clear()
            return True
        if prompt.lower() == "/new":
            await self.action_new()
            return True
        if prompt.lower() == "/load":
            self._app.push_screen(
                SessionLoadModal(
                    self._app.agent.session_dir,
                    on_load=self.queue_load_session_path,
                )
            )
            return True
        if prompt.lower().startswith("/load "):
            path_text = prompt[6:].strip()
            if path_text:
                await self.load_session_path(Path(path_text).expanduser())
            return True
        if prompt.lower() == "/resume":
            self._app.push_screen(
                SessionLoadModal(
                    self._app.agent.session_dir,
                    on_load=self.queue_load_session_path,
                )
            )
            return True
        if prompt.lower().startswith("/resume "):
            path_text = prompt[8:].strip()
            if path_text:
                await self.load_session_path(Path(path_text).expanduser())
            return True
        if prompt.lower() == "/fork":
            self._app.push_screen(
                SessionForkModal(
                    self._app.agent.session,
                    on_fork=self.queue_fork_from_message,
                )
            )
            return True
        if prompt.lower().startswith("/fork "):
            spec = prompt[6:].strip()
            message_id = self.resolve_message_id(self._app.agent.session, spec)
            if not message_id:
                chat.add_system_message(f"could not resolve message: {spec}")
                return True
            await self.fork_from_message(message_id)
            return True
        if prompt.lower() == "/tree":
            self._app.push_screen(
                SessionTreeModal(
                    self._app.agent.session,
                    on_select=self.queue_set_leaf,
                )
            )
            return True
        if prompt.lower().startswith("/tree "):
            spec = prompt[6:].strip()
            message_id = self.resolve_entry_id(self._app.agent.session, spec)
            if not message_id:
                chat.add_system_message(f"could not resolve entry: {spec}")
                return True
            await self.set_leaf(message_id)
            return True
        if prompt.lower() == "/quit":
            self._app.exit()
            return True
        if prompt.lower() == "/help":
            chat.add_system_message(
                "ctrl+c quit | ctrl+l clear | /clear | /new | /load | /resume | /fork "
                "| /tree | /context | /help | /model | /quit"
            )
            return True
        if prompt.lower() == "/context":
            self._app.push_screen(ContextModal(self._app.agent))
            return True
        if prompt.lower() == "/model":
            self._app.push_screen(
                ModelModal(
                    self._app.agent,
                    self.on_model_modal_change,
                )
            )
            return True
        if prompt.lower().startswith("/model "):
            model_name = prompt[7:].strip()
            if model_name:
                self.switch_model(model_name)
            return True
        return False

    def action_clear(self) -> None:
        """Clear chat history."""
        self._app._renderer.clear_chat()

    async def action_new(self) -> None:
        """Start a new session."""
        await self._app.agent.new_session()
        self._app._session = self._app.agent.session
        self._app._renderer.render_new_session()

    def switch_model(self, model_name: str) -> None:
        """Switch to a different model."""
        chat = self._app.query_one("#chat-view", ChatView)
        status = self._app.query_one("#status-line", StatusBar)

        try:
            self._app.agent.set_model(model_name)
        except ValueError as err:
            chat.add_system_message(str(err))
            return

        status.set_model(model_name)
        status.set_thinking(self._app.agent.thinking_level)
        chat.add_system_message(f"switched to {model_name}")

    def switch_thinking(self, level_name: str) -> None:
        """Switch thinking level."""
        chat = self._app.query_one("#chat-view", ChatView)
        status = self._app.query_one("#status-line", StatusBar)

        if not self._app.agent.supports_thinking():
            chat.add_system_message(f"model {self._app.agent.model_name} does not support thinking")
            chat.add_system_message("supported: claude-*, o1-*, o3-*, gpt-5-*")
            return

        try:
            level = ThinkingLevel(level_name)
        except ValueError:
            valid = ", ".join(t.value for t in ThinkingLevel)
            chat.add_system_message(f"invalid thinking level: {level_name}")
            chat.add_system_message(f"valid levels: {valid}")
            return

        self._app.agent.set_thinking_level(level)
        status.set_thinking(level)
        chat.add_system_message(f"thinking level: {level.value}")

    def on_model_modal_change(
        self,
        model: str | None,
        thinking: ThinkingLevel | None,
    ) -> None:
        """Callback when model modal changes settings."""
        if model:
            self.switch_model(model)

        if thinking:
            self.switch_thinking(thinking.value)
