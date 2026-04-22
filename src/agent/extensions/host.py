"""Extension host implementing the neutral agent hook boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.extensions.api import ExtensionAPI, ExtensionUIBindings
from agent.extensions.loader import ExtensionLoader
from agent.extensions.runner import ExtensionRunner
from agent.extensions.runtime import ExtensionRunState, ExtensionRuntime
from agent.extensions.types import ToolCallEvent, ToolResultEvent
from agent.prompts.parser import parse_command
from agent.runtime.events import SessionStartEvent
from agent.runtime.hooks import (
    AgentHooks,
    InputResolution,
    ToolAuthorization,
    ToolCallRequest,
    ToolResultData,
    ToolResultUpdate,
)

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

    from agent.runtime.agent import Agent
    from agent.runtime.events import AgentEvent
    from agent.runtime.hooks import RunControl
    from agent.runtime.message import Message
    from agent.runtime.session import Session


class ExtensionHost(AgentHooks):
    """Hook-host implementation backed by the extension system."""

    __slots__ = ("agent", "api", "runtime", "runner")

    def __init__(self, agent: Agent, paths: list[Path] | None = None) -> None:
        self.agent = agent
        self.api = ExtensionAPI()
        self.runtime = ExtensionRuntime(agent, paths=paths)
        self.runner = ExtensionRunner(self.api, self)

    @property
    def ui_bindings(self) -> ExtensionUIBindings | None:
        """Optional UI bindings exposed to extension contexts."""
        return self.runtime.ui_bindings

    @property
    def current_run_state(self) -> ExtensionRunState | None:
        """Return the currently active extension run state, if any."""
        return self.runtime.current_run_state

    def bind_ui(self, bindings: ExtensionUIBindings | None) -> None:
        """Bind optional UI callbacks for extensions."""
        self.runtime.bind_ui(bindings)

    def command_names(self) -> list[str]:
        """List registered extension command names."""
        return sorted(self.api.get_commands().keys())

    def config_snapshot(self) -> dict[str, object]:
        """Build the config snapshot exposed to extension contexts."""
        return self.runtime.config_snapshot()

    async def load_extensions(self, paths: list[Path] | None = None) -> list[str]:
        """Load configured extensions into the host."""
        extension_paths = self.runtime.extension_paths(paths)
        if not extension_paths:
            return []

        errors = await ExtensionLoader.load_multiple(extension_paths, self.api)
        self.runtime.register_tools(self.api.get_tools().values())

        self.agent.refresh_system_prompt()
        await self.on_event(
            SessionStartEvent(
                session_id=self.agent.session_id,
                parent_session_id=self.agent.session_parent_id,
            ),
            session=self.agent.session,
        )
        return errors

    async def resolve_input(
        self,
        text: str,
        *,
        source: str = "interactive",
    ) -> InputResolution | None:
        input_result = await self.runner.emit_input(text, source=source)
        if input_result and input_result.block:
            return InputResolution(block=True, reason=input_result.reason)
        if input_result and input_result.handled:
            return InputResolution(
                handled=True,
                handled_output=input_result.handled_output,
            )

        resolved_text = (
            input_result.text if input_result and input_result.text is not None else text
        )
        command = parse_command(resolved_text)
        if command and command.template_name in self.api.get_commands():
            output = await self.runner.execute_command(command.template_name, command.raw_args)
            handled_output = None
            if output is not None:
                handled_output = output.strip() or "(no output)"
            return InputResolution(handled=True, handled_output=handled_output)

        if resolved_text != text:
            return InputResolution(text=resolved_text)
        return None

    async def prepare_context(self, messages: list[Message]) -> list[Message]:
        return await self.runner.emit_context(messages)

    async def authorize_tool_call(
        self,
        request: ToolCallRequest,
    ) -> ToolAuthorization | None:
        result = await self.runner.emit_tool_call(
            ToolCallEvent(
                tool_name=request.tool_name,
                tool_call_id=request.tool_call_id,
                input=request.input,
            )
        )
        if result is None:
            return None
        return ToolAuthorization(block=result.block, reason=result.reason)

    async def process_tool_result(
        self,
        result: ToolResultData,
    ) -> ToolResultUpdate | None:
        modification = await self.runner.emit_tool_result(
            ToolResultEvent(
                tool_name=result.tool_name,
                tool_call_id=result.tool_call_id,
                content=result.content,
                is_error=result.is_error,
            )
        )
        if modification is None:
            return None
        return ToolResultUpdate(content=modification.content, is_error=modification.is_error)

    async def on_event(
        self,
        event: AgentEvent,
        *,
        session: Session | None = None,
    ) -> None:
        await self.runner.emit_agent_event(event, session=session)

    def run_scope(self) -> AbstractAsyncContextManager[RunControl]:
        return self.runtime.run_scope()
