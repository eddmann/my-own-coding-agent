"""Extension API for registering handlers and custom functionality."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.core.agent import Agent
    from agent.core.message import Message
    from agent.tools.base import BaseTool


class ExtensionContext:
    """Context passed to extension handlers.

    Provides safe access to agent state without exposing internals.
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self._aborted = False

    def is_idle(self) -> bool:
        """Check if the agent is idle (not processing)."""
        # Check if we're in an agent loop
        return not hasattr(self._agent, "_in_loop") or not self._agent._in_loop

    def abort(self) -> None:
        """Request the agent to abort the current operation."""
        self._aborted = True

    @property
    def aborted(self) -> bool:
        """Check if abort has been requested."""
        return self._aborted

    def get_messages(self) -> list[Message]:
        """Get a copy of the current message history."""
        return list(self._agent.session.messages)

    def get_config(self) -> dict[str, Any]:
        """Get agent configuration as a dictionary."""
        return {
            "provider": self._agent.config.provider,
            "model": self._agent.config.model,
            "context_max_tokens": self._agent.config.context_max_tokens,
            "max_output_tokens": self._agent.config.max_output_tokens,
            "temperature": self._agent.config.temperature,
        }


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
            event_type: One of "tool_call", "tool_result", "context", "input",
                or any AgentEvent type
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
