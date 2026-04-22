"""Render agent runtime state into the Textual chat UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Static

from agent.runtime.chunk import (
    MessageChunk,
    TextDeltaChunk,
    ThinkingDeltaChunk,
    ToolCallChunk,
    ToolCallStartChunk,
    ToolResultChunk,
)
from agent.tui.chat import ChatView, MessageWidget, ThinkingWidget
from agent.tui.status import StatusBar

if TYPE_CHECKING:
    from asyncio import Event

    from agent.runtime.session import Session
    from agent.tui.app import AgentApp
    from agent.tui.compose import TUILoaders

# Startup banner
BANNER = """
█▀▄▀█ █▄█   █▀█ █ █ █ █▄ █   ▄▀█ █▀▀ █▀▀ █▄ █ ▀█▀
█ ▀ █  █    █▄█ ▀▄▀▄▀ █ ▀█   █▀█ █▄█ ██▄ █ ▀█  █
""".strip()


class TUIRenderer:
    """Render runtime output and session state into Textual widgets."""

    __slots__ = ("_app", "_loaders")

    def __init__(self, app: AgentApp, *, loaders: TUILoaders) -> None:
        self._app = app
        self._loaders = loaders

    def render_banner(self) -> None:
        """Render the startup banner in chat."""
        chat = self._app.query_one("#chat-view", ChatView)
        chat.mount(
            Static(
                f"{BANNER}\n\nmy-own-coding-agent | {self._app.agent.model_name}",
                classes="message-system",
            )
        )

    def render_session_messages(self, session: Session) -> None:
        """Render existing session messages into the chat view."""
        chat = self._app.query_one("#chat-view", ChatView)
        for msg in session.messages:
            if msg.role.value == "system":
                continue
            if msg.role.value == "user":
                chat.add_user_message(msg.content)
                continue
            if msg.role.value == "assistant":
                if msg.thinking and msg.thinking.text:
                    thinking = ThinkingWidget()
                    chat.mount(thinking)
                    thinking.append_text(msg.thinking.text)
                chat.mount(MessageWidget("assistant", msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        chat.complete_tool_call(tc)
                continue
            if msg.role.value == "tool" and msg.tool_call_id:
                chat.set_tool_result(
                    msg.tool_call_id,
                    msg.content,
                    show_waiting=False,
                    create_if_missing=True,
                )

    def render_loaded_session(self, session: Session, note: str | None = None) -> None:
        """Render the currently active session into the UI."""
        chat = self._app.query_one("#chat-view", ChatView)
        chat.clear_chat()
        self.render_banner()
        if note:
            chat.add_system_message(note)
        self.render_session_messages(session)

        status = self._app.query_one("#status-line", StatusBar)
        status.set_model(self._app.agent.model_name)
        status.set_thinking(self._app.agent.thinking_level)
        status.set_session(session.metadata.id, session.metadata.parent_session_id)
        status.set_tokens(self._app.agent.total_tokens, self._app.agent.context_max_tokens)
        status.set_extension_status(None)

    def render_skill_invocation(self, prompt: str) -> bool:
        """Render a skill invocation block if prompt is $skill-name."""
        if not prompt.startswith("$"):
            return False

        space_index = prompt.find(" ")
        skill_name = prompt[1:] if space_index == -1 else prompt[1:space_index]
        args = "" if space_index == -1 else prompt[space_index + 1 :].strip()

        if not skill_name:
            return False

        skill = self._loaders.skill_loader.get(skill_name)
        if not skill:
            return False

        try:
            body = skill.read_body()
        except Exception:
            return False

        content = f"References are relative to {skill.base_dir}.\n\n{body}"
        chat = self._app.query_one("#chat-view", ChatView)
        chat.add_skill_invocation(skill.name, content, str(skill.readme_path), args or None)
        return True

    async def render_agent_run(self, prompt: str, cancel_event: Event | None) -> None:
        """Stream agent output into chat widgets."""
        chat = self._app.query_one("#chat-view", ChatView)
        in_thinking = False

        try:
            async for chunk in self._app.agent.run(prompt, cancel_event=cancel_event):
                if cancel_event and cancel_event.is_set():
                    break

                match chunk:
                    case ThinkingDeltaChunk(payload=thinking):
                        if not in_thinking:
                            await chat.start_thinking()
                            in_thinking = True
                        chat.append_to_thinking(thinking.text)
                    case TextDeltaChunk(payload=text):
                        if in_thinking:
                            chat.end_thinking()
                            in_thinking = False
                        await chat.append_to_assistant(text)
                    case ToolCallStartChunk(payload=tool_call_start):
                        if in_thinking:
                            chat.end_thinking()
                            in_thinking = False
                        chat.end_assistant_message()
                        await chat.start_tool_call(tool_call_start.id, tool_call_start.name)
                    case ToolCallChunk(payload=tool_call):
                        if in_thinking:
                            chat.end_thinking()
                            in_thinking = False
                        chat.end_assistant_message()
                        chat.complete_tool_call(tool_call)
                    case ToolResultChunk(payload=tool_result):
                        chat.set_tool_result(tool_result.tool_call_id, tool_result.result)
                    case MessageChunk(payload=message) if message.role.value == "system":
                        chat.add_system_message(message.content)
                    case _:
                        pass

            if in_thinking:
                chat.end_thinking()
            chat.end_assistant_message()
        except Exception as exc:
            if in_thinking:
                chat.end_thinking()
            chat.end_assistant_message()
            chat.add_system_message(f"error: {type(exc).__name__}: {exc}")
        finally:
            status = self._app.query_one("#status-line", StatusBar)
            status.set_tokens(self._app.agent.total_tokens, self._app.agent.context_max_tokens)

    def clear_chat(self) -> None:
        """Clear the chat view and show confirmation."""
        chat = self._app.query_one("#chat-view", ChatView)
        chat.clear_chat()
        chat.add_system_message("cleared")

    def render_new_session(self) -> None:
        """Render the UI state after starting a new session."""
        chat = self._app.query_one("#chat-view", ChatView)
        chat.clear_chat()
        self.render_banner()
        chat.add_system_message("new session started")

        status = self._app.query_one("#status-line", StatusBar)
        status.set_session(
            self._app.agent.session_id,
            self._app.agent.session_parent_id,
        )
        status.set_tokens(self._app.agent.total_tokens, self._app.agent.context_max_tokens)
        status.set_extension_status(None)
