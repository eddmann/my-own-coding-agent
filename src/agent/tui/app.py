"""Main Textual TUI application."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.theme import Theme
from textual.widgets import Static

from agent.tui.chat import ChatView
from agent.tui.compose import TUILoaders, TUIRuntime, build_tui_loaders, build_tui_runtime
from agent.tui.controller import TUIController
from agent.tui.extension_bridge import TUIExtensionBridge
from agent.tui.input import PromptInput
from agent.tui.renderer import TUIRenderer
from agent.tui.status import StatusBar

if TYPE_CHECKING:
    from textual.timer import Timer

    from agent.config import Config
    from agent.extensions.host import ExtensionHost
    from agent.llm.provider import LLMProvider
    from agent.runtime.agent import Agent
    from agent.runtime.session import Session

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

    def __init__(
        self,
        config: Config,
        provider: LLMProvider,
        session: Session | None = None,
    ) -> None:
        super().__init__()
        self._bootstrap_config = config
        self._session = session
        self._provider = provider
        self._runtime: TUIRuntime | None = None
        self._loaders: TUILoaders = build_tui_loaders(config)
        self._controller = TUIController(self)
        self._extension_bridge = TUIExtensionBridge(self)
        self._renderer = TUIRenderer(self, loaders=self._loaders)
        self._spinner_timer: Timer | None = None
        self._extension_widget_timer: Timer | None = None

        self._cancel_event: asyncio.Event | None = None

    @property
    def runtime(self) -> TUIRuntime:
        """Get the runtime stack, creating it if needed."""
        if self._runtime is None:
            self._runtime = build_tui_runtime(
                self._bootstrap_config,
                self._provider,
                self._session,
                loaders=self._loaders,
                extension_bridge=self._extension_bridge,
            )
        return self._runtime

    @property
    def agent(self) -> Agent:
        """Get the agent, creating it if needed."""
        return self.runtime.agent

    @property
    def extension_host(self) -> ExtensionHost:
        """Get the extension host, creating the agent if needed."""
        return self.runtime.extension_host

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
        with Horizontal(id="main-container"):
            yield ChatView(id="chat-view")
            yield Static("", id="extension-right-panel", classes="extension-slot hidden")
        yield Static("", id="extension-footer", classes="extension-slot hidden")
        with Vertical(id="input-container"):
            yield PromptInput(
                skill_loader=self._loaders.skill_loader,
                template_loader=self._loaders.template_loader,
                id="prompt-input",
            )
        yield StatusBar(id="status-line")

    async def on_mount(self) -> None:
        """Initialize on mount."""
        # Ensure agent is created so session model restore applies before UI renders
        _ = self.agent
        if self._session is None:
            self._session = self.agent.session

        # Register and apply theme
        self.register_theme(MINIMAL_THEME)
        self.theme = "minimal"

        # Show banner
        self._renderer.render_banner()

        # Update status bar
        status = self.query_one("#status-line", StatusBar)
        status.set_model(self.agent.model_name)
        status.set_thinking(self.agent.thinking_level)
        status.set_session(
            self.agent.session_id,
            self.agent.session_parent_id,
        )
        status.set_extension_status(None)

        # Focus input
        self.query_one("#prompt-input", PromptInput).focus()

        # Load extensions if configured
        if self._bootstrap_config.extensions:
            errors = await self.extension_host.load_extensions()
            chat = self.query_one("#chat-view", ChatView)
            for error in errors:
                chat.add_system_message(f"extension error: {error}")
            commands = self.extension_host.command_names()
            self.query_one("#prompt-input", PromptInput).set_extension_commands(commands)

        # Show existing messages if resuming a session
        if self._session and self._session.messages:
            self._renderer.render_session_messages(self._session)

        self._extension_widget_timer = self.set_interval(
            0.2,
            self._extension_bridge.render_widgets,
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

        if await self._controller.handle_prompt_command(prompt):
            return

        # Add user message (special handling for $skill-name)
        if not self._renderer.render_skill_invocation(prompt):
            chat.add_user_message(prompt)

        # Start processing - show thinking indicator if thinking is enabled
        self.is_processing = True
        thinking_enabled = (
            self.agent.thinking_level.value != "off" and self.agent.supports_thinking()
        )
        chat.start_assistant_message(thinking=thinking_enabled)

        # Run agent in background
        self._run_agent(prompt)

    def _run_agent(self, prompt: str) -> None:
        """Run the agent loop."""
        self._cancel_event = asyncio.Event()
        asyncio.create_task(self._agent_worker(prompt))

    async def _agent_worker(self, prompt: str) -> None:
        """Execute agent and handle events."""
        try:
            await self._renderer.render_agent_run(prompt, self._cancel_event)
        finally:
            self.is_processing = False
            self._cancel_event = None
            self.query_one("#prompt-input", PromptInput).focus()

    def action_clear(self) -> None:
        """Clear chat history."""
        self._controller.action_clear()

    async def action_new(self) -> None:
        """Start a new session."""
        await self._controller.action_new()

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
        if self._extension_widget_timer:
            self._extension_widget_timer.stop()
        if self._runtime:
            await self._runtime.agent.close()

    async def push_extension_screen(self, screen: Any) -> object | None:
        """Push a modal screen and await its dismissal without Textual workers."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object | None] = loop.create_future()

        def _done(result: object | None) -> None:
            if not future.done():
                future.set_result(result)

        self.push_screen(screen, callback=_done)
        return await future
