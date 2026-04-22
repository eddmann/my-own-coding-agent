"""Extension runner that invokes handlers and manages event flow."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any

from agent.extensions.api import ExtensionAPI, ExtensionContext
from agent.extensions.types import (
    ContextEvent,
    ContextModification,
    InputEvent,
    InputResult,
    ToolCallEvent,
    ToolCallResult,
    ToolResultEvent,
    ToolResultModification,
)

if TYPE_CHECKING:
    from agent.extensions.host import ExtensionHost
    from agent.runtime.events import AgentEvent
    from agent.runtime.message import Message
    from agent.runtime.session import Session


class ExtensionRunner:
    """Runs extensions and manages event flow.

    The runner:
    - Holds the ExtensionAPI with registered handlers
    - Creates ExtensionContext for each invocation
    - Calls handlers in registration order
    - Handles async and sync handlers uniformly
    """

    def __init__(self, api: ExtensionAPI, host: ExtensionHost) -> None:
        self.api = api
        self.host = host

    async def _call_handler(self, handler: Any, event: Any, ctx: ExtensionContext) -> Any:
        """Call a handler, handling both async and sync functions."""
        result = handler(event, ctx)
        if inspect.iscoroutine(result):
            return await result
        return result

    async def emit_tool_call(
        self,
        event: ToolCallEvent,
    ) -> ToolCallResult | None:
        """Emit tool_call event - returns first blocking result.

        Extensions can return ToolCallResult(block=True) to prevent execution.
        """
        ctx = ExtensionContext(self.host)
        for handler in self.api.get_handlers("tool_call"):
            try:
                result = await self._call_handler(handler, event, ctx)
                if isinstance(result, ToolCallResult) and result.block:
                    return result  # Early exit on block
            except Exception:
                # Log but continue with other handlers
                pass
        return None

    async def emit_tool_result(
        self,
        event: ToolResultEvent,
    ) -> ToolResultModification | None:
        """Emit tool_result event - allows modification of results.

        Extensions can return ToolResultModification to change content.
        """
        ctx = ExtensionContext(self.host)
        modification: ToolResultModification | None = None

        for handler in self.api.get_handlers("tool_result"):
            try:
                result = await self._call_handler(handler, event, ctx)
                if isinstance(result, ToolResultModification):
                    # Apply modifications (last one wins for each field)
                    if modification is None:
                        modification = result
                    else:
                        if result.content is not None:
                            modification.content = result.content
                        if result.is_error is not None:
                            modification.is_error = result.is_error
            except Exception:
                pass

        return modification

    async def emit_context(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """Emit context event - allows message modification before LLM call.

        Extensions can modify the message list sent to the LLM.
        """
        ctx = ExtensionContext(self.host)
        current = list(messages)  # Copy to avoid mutating original

        for handler in self.api.get_handlers("context"):
            try:
                event = ContextEvent(messages=current)
                result = await self._call_handler(handler, event, ctx)
                if isinstance(result, ContextModification) and result.messages is not None:
                    current = result.messages
            except Exception:
                pass

        return current

    async def emit_input(
        self,
        text: str,
        source: str = "interactive",
    ) -> InputResult | None:
        """Emit input event - allows transformation or blocking of user input.

        Extensions can modify the input text or block it entirely.
        """
        ctx = ExtensionContext(self.host)
        event = InputEvent(text=text, source=source)
        result_text = text

        for handler in self.api.get_handlers("input"):
            try:
                result = await self._call_handler(handler, event, ctx)
                if isinstance(result, InputResult):
                    if result.block:
                        return result  # Early exit on block
                    if result.handled:
                        return result
                    if result.text is not None:
                        result_text = result.text
                        event = InputEvent(text=result_text, source=source)
            except Exception:
                pass

        if result_text != text:
            return InputResult(text=result_text)
        return None

    async def emit_agent_event(
        self,
        event: AgentEvent,
        *,
        session: Session | None = None,
    ) -> None:
        """Emit a general agent event to all subscribers.

        These events are informational - handlers cannot modify behavior.
        """
        ctx = ExtensionContext(self.host, session=session)
        event_type = event.type

        handlers = self.api.get_handlers(event_type)
        if not handlers:
            return

        # Run all handlers concurrently for informational events
        tasks = [self._call_handler(h, event, ctx) for h in handlers]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def execute_command(
        self,
        name: str,
        args: str,
    ) -> str | None:
        """Execute a registered slash command.

        Args:
            name: Command name (without slash)
            args: Arguments string

        Returns:
            Command response or None if command not found
        """
        commands = self.api.get_commands()
        if name not in commands:
            return None

        ctx = ExtensionContext(self.host)
        handler = commands[name]

        result = handler(args, ctx)
        if inspect.iscoroutine(result):
            return await result
        return result
