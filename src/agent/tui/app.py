"""Main Textual TUI application."""

import asyncio
from typing import TYPE_CHECKING

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.reactive import reactive
from textual.theme import Theme
from textual.widgets import Static

from agent.core.agent import Agent
from agent.core.config import Config, ThinkingLevel
from agent.core.message import Message, ThinkingContent, ToolCall, ToolCallStart, ToolResult
from agent.prompts.loader import PromptTemplateLoader
from agent.skills.loader import SkillLoader
from agent.tui.chat import ChatView, MessageWidget, ThinkingWidget
from agent.tui.context_modal import ContextModal
from agent.tui.input import PromptInput
from agent.tui.model_modal import ModelModal
from agent.tui.status import StatusBar

if TYPE_CHECKING:
    from textual.timer import Timer

    from agent.core.session import Session

# Startup banner
BANNER = """
█▀▄▀█ █▄█   █▀█ █ █ █ █▄ █   ▄▀█ █▀▀ █▀▀ █▄ █ ▀█▀
█ ▀ █  █    █▄█ ▀▄▀▄▀ █ ▀█   █▀█ █▄█ ██▄ █ ▀█  █
""".strip()

# Minimal Nord-inspired theme
MINIMAL_THEME = Theme(
    name="minimal",
    primary="#88c0d0",
    secondary="#81a1c1",
    accent="#5e81ac",
    foreground="#d8dee9",
    background="#2e3440",
    surface="#3b4252",
    panel="#434c5e",
    success="#a3be8c",
    warning="#ebcb8b",
    error="#bf616a",
    dark=True,
)


class AgentApp(App[None]):
    """Minimal TUI application for the coding agent."""

    CSS_PATH = "styles.tcss"
    TITLE = "my-own-coding-agent"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("escape", "escape_pressed", "Cancel/Focus", show=False),
    ]

    is_processing = reactive(False)

    def __init__(self, config: Config, session: Session | None = None) -> None:
        super().__init__()
        self.config = config
        self._session = session
        self._agent: Agent | None = None
        self._spinner_timer: Timer | None = None

        self._cancel_event: asyncio.Event | None = None

        # Create loaders for autocomplete (shared with agent)
        self._skill_loader = SkillLoader.with_defaults(
            extra_dirs=config.skills_dirs,
        )
        self._template_loader = PromptTemplateLoader.with_defaults(
            extra_dirs=config.prompt_template_dirs,
        )

    @property
    def agent(self) -> Agent:
        """Get the agent, creating it if needed."""
        if self._agent is None:
            self._agent = Agent(
                self.config,
                self._session,
                skill_loader=self._skill_loader,
                template_loader=self._template_loader,
            )
        return self._agent

    def watch_is_processing(self, processing: bool) -> None:
        """React to processing state changes."""
        if processing:
            self._spinner_timer = self.set_interval(0.2, self._animate_spinner)
        elif self._spinner_timer:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def _animate_spinner(self) -> None:
        """Animate the spinner in the chat waiting indicator."""
        chat = self.query_one("#chat-view", ChatView)
        chat.advance_waiting()

    def compose(self) -> ComposeResult:
        """Compose the UI."""
        yield ChatView(id="chat-view")
        with Vertical(id="input-container"):
            yield PromptInput(
                skill_loader=self._skill_loader,
                template_loader=self._template_loader,
                id="prompt-input",
            )
        yield StatusBar(id="status-line")

    async def on_mount(self) -> None:
        """Initialize on mount."""
        # Register and apply theme
        self.register_theme(MINIMAL_THEME)
        self.theme = "minimal"

        # Show banner
        chat = self.query_one("#chat-view", ChatView)
        chat.mount(
            Static(
                f"{BANNER}\n\nmy-own-coding-agent | {self.config.model}", classes="message-system"
            )
        )

        # Update status bar
        status = self.query_one("#status-line", StatusBar)
        status.set_model(self.config.model)
        status.set_thinking(self.config.thinking_level)

        # Focus input
        self.query_one("#prompt-input", PromptInput).focus()

        # Load extensions if configured
        if self.config.extensions:
            errors = await self.agent.load_extensions()
            for error in errors:
                chat.add_system_message(f"extension error: {error}")
            commands = sorted(self.agent.extension_api.get_commands().keys())
            self.query_one("#prompt-input", PromptInput).set_extension_commands(commands)

        # Show existing messages if resuming a session
        if self._session and self._session.messages:
            for msg in self._session.messages:
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

    @on(PromptInput.Submitted, "#prompt-input")
    async def on_input_submitted(self, event: PromptInput.Submitted) -> None:
        """Handle user input submission."""
        if self.is_processing:
            return

        prompt = event.value.strip()
        if not prompt:
            return

        self.query_one("#prompt-input", PromptInput).clear()
        chat = self.query_one("#chat-view", ChatView)

        # Handle commands
        if prompt.lower() == "/clear":
            self.action_clear()
            return
        if prompt.lower() == "/new":
            self.action_new()
            return
        if prompt.lower() == "/quit":
            self.exit()
            return
        if prompt.lower() == "/help":
            chat.add_system_message(
                "ctrl+c quit | ctrl+l clear | /clear | /new | /context | /help | /model | /quit"
            )
            return
        if prompt.lower() == "/context":
            self.push_screen(ContextModal(self.agent, self.config))
            return
        if prompt.lower() == "/model":
            self.push_screen(
                ModelModal(
                    self.config,
                    self.agent.provider,
                    self._on_model_modal_change,
                )
            )
            return
        if prompt.lower().startswith("/model "):
            model_name = prompt[7:].strip()
            if model_name:
                self._switch_model(model_name)
            return

        # Add user message (special handling for $skill-name)
        if not self._render_skill_invocation(prompt):
            chat.add_user_message(prompt)

        # Start processing - show thinking indicator if thinking is enabled
        self.is_processing = True
        thinking_enabled = (
            self.config.thinking_level.value != "off" and self.agent.provider.supports_thinking()
        )
        chat.start_assistant_message(thinking=thinking_enabled)

        # Run agent in background
        self._run_agent(prompt)

    def _render_skill_invocation(self, prompt: str) -> bool:
        """Render a skill invocation block if prompt is $skill-name."""
        if not prompt.startswith("$"):
            return False

        space_index = prompt.find(" ")
        skill_name = prompt[1:] if space_index == -1 else prompt[1:space_index]
        args = "" if space_index == -1 else prompt[space_index + 1 :].strip()

        if not skill_name:
            return False

        skill = self._skill_loader.get(skill_name)
        if not skill:
            return False

        try:
            body = skill.read_body()
        except Exception:
            return False

        content = f"References are relative to {skill.base_dir}.\n\n{body}"
        chat = self.query_one("#chat-view", ChatView)
        chat.add_skill_invocation(skill.name, content, str(skill.readme_path), args or None)
        return True

    def _run_agent(self, prompt: str) -> None:
        """Run the agent loop."""
        self._cancel_event = asyncio.Event()
        asyncio.create_task(self._agent_worker(prompt))

    async def _agent_worker(self, prompt: str) -> None:
        """Execute agent and handle events."""
        chat = self.query_one("#chat-view", ChatView)
        in_thinking = False

        try:
            async for chunk in self.agent.run(prompt, cancel_event=self._cancel_event):
                # Check if cancelled between chunks
                if self._cancel_event and self._cancel_event.is_set():
                    break

                if isinstance(chunk, ThinkingContent):
                    if not in_thinking:
                        await chat.start_thinking()
                        in_thinking = True
                    chat.append_to_thinking(chunk.text)
                elif isinstance(chunk, str):
                    if in_thinking:
                        chat.end_thinking()
                        in_thinking = False
                    await chat.append_to_assistant(chunk)
                elif isinstance(chunk, ToolCallStart):
                    # Tool call starting - show early indicator
                    if in_thinking:
                        chat.end_thinking()
                        in_thinking = False
                    chat.end_assistant_message()
                    await chat.start_tool_call(chunk.id, chunk.name)
                elif isinstance(chunk, ToolCall):
                    # Tool call complete with arguments
                    if in_thinking:
                        chat.end_thinking()
                        in_thinking = False
                    chat.end_assistant_message()
                    chat.complete_tool_call(chunk)
                elif isinstance(chunk, ToolResult):
                    chat.set_tool_result(chunk.tool_call_id, chunk.result)
                elif isinstance(chunk, Message):
                    if chunk.role.value == "system":
                        chat.add_system_message(chunk.content)

            if in_thinking:
                chat.end_thinking()
            chat.end_assistant_message()

        except Exception as e:
            if in_thinking:
                chat.end_thinking()
            chat.end_assistant_message()
            chat.add_system_message(f"error: {type(e).__name__}: {e}")

        finally:
            self.is_processing = False
            self._cancel_event = None
            # Update token count in status bar
            status = self.query_one("#status-line", StatusBar)
            status.set_tokens(self.agent.total_tokens, self.config.context_max_tokens)
            self.query_one("#prompt-input", PromptInput).focus()

    def action_clear(self) -> None:
        """Clear chat history."""
        chat = self.query_one("#chat-view", ChatView)
        chat.clear_chat()
        chat.add_system_message("cleared")

    def action_new(self) -> None:
        """Start a new session."""
        self.agent.new_session()
        self._session = self.agent.session
        chat = self.query_one("#chat-view", ChatView)
        chat.clear_chat()
        chat.add_system_message("new session started")
        status = self.query_one("#status-line", StatusBar)
        status.set_tokens(self.agent.total_tokens, self.config.context_max_tokens)

    def _switch_model(self, model_name: str) -> None:
        """Switch to a different model."""
        chat = self.query_one("#chat-view", ChatView)
        status = self.query_one("#status-line", StatusBar)

        # Update provider
        self.agent.provider.model = model_name
        # Reset encoder for new model if provider has one
        if hasattr(self.agent.provider, "_encoder"):
            self.agent.provider._encoder = None

        # Update UI
        status.set_model(model_name)
        chat.add_system_message(f"switched to {model_name}")

    def _switch_thinking(self, level_name: str) -> None:
        """Switch thinking level."""
        chat = self.query_one("#chat-view", ChatView)
        status = self.query_one("#status-line", StatusBar)

        # Check if model supports thinking
        if not self.agent.provider.supports_thinking():
            chat.add_system_message(f"model {self.agent.provider.model} does not support thinking")
            chat.add_system_message("supported: claude-*, o1-*, o3-*, gpt-5-*")
            return

        try:
            level = ThinkingLevel(level_name)
        except ValueError:
            valid = ", ".join(t.value for t in ThinkingLevel)
            chat.add_system_message(f"invalid thinking level: {level_name}")
            chat.add_system_message(f"valid levels: {valid}")
            return

        # Update config (agent reads this each turn)
        self.config.thinking_level = level

        # Update status bar
        status.set_thinking(level)
        chat.add_system_message(f"thinking level: {level.value}")

    def _on_model_modal_change(self, model: str | None, thinking: ThinkingLevel | None) -> None:
        """Callback when model modal changes settings."""
        if model:
            self._switch_model(model)

        if thinking:
            self._switch_thinking(thinking.value)

    def action_escape_pressed(self) -> None:
        """Handle escape key - cancel processing or focus input."""
        if self.is_processing:
            self._cancel_agent()
        else:
            self.query_one("#prompt-input", PromptInput).focus()

    def _cancel_agent(self) -> None:
        """Cancel the running agent."""
        if self._cancel_event:
            self._cancel_event.set()  # Signal cancellation to provider

        # Clean up UI and show feedback
        chat = self.query_one("#chat-view", ChatView)
        chat.end_assistant_message()
        chat.add_system_message("interrupted")

    async def on_unmount(self) -> None:
        """Clean up on exit."""
        if self._agent:
            await self._agent.close()
