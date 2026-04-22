"""Extension API for registering handlers and custom functionality."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

from agent.runtime.settings import (
    ThinkingLevel,
    clamp_thinking_level,
    get_available_thinking_levels,
)

if TYPE_CHECKING:
    from agent.extensions.host import ExtensionHost
    from agent.runtime.agent import Agent
    from agent.runtime.message import Message
    from agent.runtime.session import Session, SessionEntry
    from agent.tools.base import BaseTool

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@dataclass(slots=True, frozen=True)
class ViewControl:
    """Host-rendered control for a presented view."""

    kind: Literal["input", "select", "button"]
    name: str
    label: str = ""
    placeholder: str | None = None
    options: tuple[str, ...] = ()
    primary: bool = False


class PresentedView(Protocol[T_co]):
    """Protocol for temporary interactive UI flows."""

    def render(self) -> str: ...

    def controls(self) -> list[ViewControl]: ...

    def handle_action(self, action: str, value: str | None = None) -> None: ...

    def is_done(self) -> bool: ...

    def result(self) -> T | None: ...


class WidgetView(Protocol):
    """Protocol for persistent widget content."""

    def render(self) -> str: ...


@dataclass(slots=True, frozen=True)
class ExtensionUIBindings:
    """Optional host UI callbacks exposed to extensions."""

    notify: Callable[[str, str], Any] | None = None
    set_status: Callable[[str | None], Any] | None = None
    input: Callable[[str, str | None], Any] | None = None
    confirm: Callable[[str], Any] | None = None
    select: Callable[[str, list[str]], Any] | None = None
    present: Callable[[PresentedView[Any]], Any] | None = None
    set_widget: Callable[[str, WidgetView | None], Any] | None = None

    def is_bound(self) -> bool:
        """Whether any UI capability is available."""
        return any(
            callback is not None
            for callback in (
                self.notify,
                self.set_status,
                self.input,
                self.confirm,
                self.select,
                self.present,
                self.set_widget,
            )
        )


class RuntimeAPI:
    """Runtime control surface exposed to extensions."""

    __slots__ = ("_host",)

    def __init__(self, host: ExtensionHost) -> None:
        self._host = host

    def is_idle(self) -> bool:
        """Check if the agent is idle (not processing)."""
        return not self._host.agent.is_processing

    def abort(self) -> None:
        """Request the current run to abort, if one is active."""
        run_state = self._host.current_run_state
        if run_state is None:
            return
        run_state.request_abort()

    @property
    def aborted(self) -> bool:
        """Check if abort has been requested for the current run."""
        run_state = self._host.current_run_state
        if run_state is None:
            return False
        return run_state.abort_requested

    async def send_user_message(self, text: str) -> None:
        """Queue a user message for processing after the current step completes."""
        run_state = self._host.current_run_state
        if run_state is None:
            return
        run_state.queue_user_message(text)

    def get_system_prompt(self) -> str:
        """Get the current system prompt content."""
        return self._host.agent.get_system_prompt()


class SessionAPI:
    """Session and branching surface exposed to extensions."""

    __slots__ = ("_agent", "_session")

    def __init__(self, agent: Agent, session: Session) -> None:
        self._agent = agent
        self._session = session

    @property
    def id(self) -> str:
        return self._session.metadata.id

    @property
    def parent_id(self) -> str | None:
        return self._session.metadata.parent_session_id

    def messages(self) -> list[Message]:
        """Get a copy of the current session messages."""
        return list(self._session.messages)

    def entries(self) -> list[SessionEntry]:
        """Get a copy of the current session entries."""
        return list(self._session.entries)

    async def fork(self, from_message_id: str) -> str:
        """Fork the active session from a message and switch to the fork."""
        new_session = self._agent.fork_session(from_message_id, session=self._session)
        await self._agent.load_session(new_session)
        return new_session.metadata.id

    def set_leaf(self, entry_id: str) -> None:
        """Switch the active branch leaf."""
        self._agent.set_leaf(entry_id)

    async def new(self) -> str:
        """Start a fresh session and switch to it."""
        await self._agent.new_session()
        return self._agent.session_id


class ModelAPI:
    """Model and reasoning controls exposed to extensions."""

    __slots__ = ("_agent",)

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def get(self) -> str:
        return self._agent.model_name

    def set(self, model: str) -> None:
        self._agent.set_model(model, source="extension")

    def get_thinking_level(self) -> str:
        return self._agent.thinking_level.value

    def set_thinking_level(self, level: str) -> None:
        requested = ThinkingLevel(level)
        available = get_available_thinking_levels(
            self._agent.model_name,
            provider=self._agent.provider_name,
        )
        self._agent.set_thinking_level(clamp_thinking_level(requested, available))


class ToolsAPI:
    """Tool inspection and activation controls exposed to extensions."""

    __slots__ = ("_agent",)

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def available(self) -> list[str]:
        return self._agent.list_tools()

    def active(self) -> list[str]:
        return self._agent.list_active_tools()

    def set_active(self, names: list[str]) -> None:
        self._agent.set_active_tools(names)

    def register(self, tool: BaseTool[Any]) -> None:
        """Register a tool into the live tool registry for future turns."""
        self._agent.register_tool(tool)


class UIAPI:
    """TUI-oriented UI surface exposed to extensions."""

    __slots__ = (
        "_confirm",
        "_input",
        "_notify",
        "_present",
        "_select",
        "_set_status",
        "_set_widget",
    )

    def __init__(
        self,
        bindings: ExtensionUIBindings,
    ) -> None:
        self._notify = bindings.notify
        self._set_status = bindings.set_status
        self._input = bindings.input
        self._confirm = bindings.confirm
        self._select = bindings.select
        self._present = bindings.present
        self._set_widget = bindings.set_widget

    def notify(self, message: str, level: str = "info") -> None:
        """Emit a host notification."""
        if self._notify is None:
            return
        result = self._notify(message, level)
        if asyncio.iscoroutine(result):
            asyncio.create_task(result)

    def set_status(self, text: str | None) -> None:
        """Update the host status line, if available."""
        if self._set_status is None:
            return
        result = self._set_status(text)
        if asyncio.iscoroutine(result):
            asyncio.create_task(result)

    async def input(self, prompt: str, default: str | None = None) -> str | None:
        """Prompt for freeform text input."""
        if self._input is None:
            return default
        result = self._input(prompt, default)
        if asyncio.iscoroutine(result):
            return cast("str | None", await result)
        return cast("str | None", result)

    async def confirm(self, prompt: str) -> bool:
        """Prompt for confirmation."""
        if self._confirm is None:
            return False
        result = self._confirm(prompt)
        if asyncio.iscoroutine(result):
            return bool(await result)
        return bool(result)

    async def select(self, prompt: str, options: list[str]) -> str | None:
        """Prompt the user to select an option."""
        if self._select is None:
            return None
        result = self._select(prompt, options)
        if asyncio.iscoroutine(result):
            return cast("str | None", await result)
        return cast("str | None", result)

    async def present(self, view: PresentedView[T]) -> T | None:
        """Present a custom temporary interactive view."""
        if self._present is None:
            return None
        result = self._present(view)
        if asyncio.iscoroutine(result):
            return cast("T | None", await result)
        return cast("T | None", result)

    def set_widget(
        self,
        slot: Literal["footer", "right_panel"],
        view: WidgetView | None,
    ) -> None:
        """Set or clear a persistent widget in a named slot."""
        if self._set_widget is None:
            return
        result = self._set_widget(slot, view)
        if asyncio.iscoroutine(result):
            asyncio.create_task(result)


class ExtensionContext:
    """Context passed to extension handlers.

    Provides safe access to agent state without exposing internals.
    """

    __slots__ = ("cwd", "config", "runtime", "session", "model", "tools", "ui", "_agent")

    def __init__(
        self,
        host: ExtensionHost,
        session: Session | None = None,
    ) -> None:
        self._agent = host.agent
        active_session = session or host.agent.session
        self.cwd = host.agent.cwd
        self.config = MappingProxyType(host.config_snapshot())
        self.runtime = RuntimeAPI(host)
        self.session = SessionAPI(host.agent, active_session)
        self.model = ModelAPI(host.agent)
        self.tools = ToolsAPI(host.agent)
        bindings = host.ui_bindings
        self.ui = UIAPI(bindings) if bindings is not None and bindings.is_bound() else None

    def is_idle(self) -> bool:
        """Backward-compatible alias for ctx.runtime.is_idle()."""
        return self.runtime.is_idle()

    def abort(self) -> None:
        """Backward-compatible alias for ctx.runtime.abort()."""
        self.runtime.abort()

    @property
    def aborted(self) -> bool:
        """Backward-compatible alias for ctx.runtime.aborted."""
        return self.runtime.aborted

    def get_messages(self) -> list[Message]:
        """Backward-compatible alias for ctx.session.messages()."""
        return self.session.messages()

    def get_config(self) -> dict[str, Any]:
        """Backward-compatible config snapshot."""
        return dict(self.config)


# Type alias for event handlers
EventHandler = Callable[[Any, ExtensionContext], Any]
CommandHandler = Callable[[str, ExtensionContext], str | None]


class ExtensionAPI:
    """API for extensions to register handlers and custom functionality.

    Extensions use this API to:
    - Subscribe to events (tool_call, tool_result, context, input)
    - Register custom tools
    - Register slash commands

    Example extension:
        def setup(api: ExtensionAPI):
            # Block dangerous commands
            async def block_rm(event, ctx):
                if "rm " in str(event.input):
                    return ToolCallResult(block=True, reason="rm blocked")
            api.on("tool_call", block_rm)

            # Register a custom command
            api.register_command("status", lambda args, ctx: "All systems operational")
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._tools: dict[str, BaseTool[Any]] = {}
        self._commands: dict[str, CommandHandler] = {}

    def on(self, event_type: str, handler: EventHandler) -> Callable[[], None]:
        """Subscribe to an event type.

        Args:
            event_type: A supported extension event name such as
                "input", "context", "tool_call", "tool_result",
                "session_start", "session_end", "turn_start",
                "turn_end", "agent_start", "agent_end",
                "model_select", or "compaction".
                Internal runtime events may also be observable, but are
                not part of the stable public extension contract.
            handler: Async or sync function receiving (event, context)

        Returns:
            Unsubscribe function
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

        return unsubscribe

    def register_tool(self, tool: BaseTool[Any]) -> None:
        """Register a custom tool.

        Args:
            tool: Tool instance implementing BaseTool
        """
        self._tools[tool.name] = tool

    def register_command(self, name: str, handler: CommandHandler) -> None:
        """Register a slash command.

        Args:
            name: Command name (without the slash)
            handler: Function receiving (args_string, context), returns response or None
        """
        self._commands[name] = handler

    def get_handlers(self, event_type: str) -> list[EventHandler]:
        """Get all handlers for an event type."""
        return self._handlers.get(event_type, [])

    def get_tools(self) -> dict[str, BaseTool[Any]]:
        """Get all registered custom tools."""
        return self._tools.copy()

    def get_commands(self) -> dict[str, CommandHandler]:
        """Get all registered commands."""
        return self._commands.copy()
