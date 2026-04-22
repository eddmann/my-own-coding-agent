"""Neutral integration hooks for the agent runtime."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agent.runtime.events import AgentEvent
    from agent.runtime.message import Message
    from agent.runtime.session import Session


@dataclass(slots=True)
class InputResolution:
    """Result of resolving raw user input before the agent loop proceeds."""

    text: str | None = None
    block: bool = False
    reason: str | None = None
    handled: bool = False
    handled_output: str | None = None


@dataclass(slots=True)
class ToolCallRequest:
    """Neutral tool-call request passed through hook hosts."""

    tool_name: str = ""
    tool_call_id: str = ""
    input: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ToolAuthorization:
    """Decision returned before a tool call executes."""

    block: bool = False
    reason: str | None = None


@dataclass(slots=True)
class ToolResultData:
    """Neutral tool-result payload passed through hook hosts."""

    tool_name: str = ""
    tool_call_id: str = ""
    content: str = ""
    is_error: bool = False


@dataclass(slots=True)
class ToolResultUpdate:
    """Mutation returned after a tool call executes."""

    content: str | None = None
    is_error: bool | None = None


class RunControl(Protocol):
    """Mutable control state for one active runtime invocation."""

    def reset(self) -> None: ...

    @property
    def aborted(self) -> bool: ...

    def drain_user_messages(self) -> list[str]: ...


@dataclass(slots=True)
class NullRunControl:
    """No-op run control used when no hook host is attached."""

    def reset(self) -> None:
        return None

    @property
    def aborted(self) -> bool:
        return False

    def drain_user_messages(self) -> list[str]:
        return []


class AgentHooks(Protocol):
    """Neutral integration seam for the core agent runtime."""

    async def resolve_input(
        self,
        text: str,
        *,
        source: str = "interactive",
    ) -> InputResolution | None: ...

    async def prepare_context(self, messages: list[Message]) -> list[Message]: ...

    async def authorize_tool_call(
        self,
        request: ToolCallRequest,
    ) -> ToolAuthorization | None: ...

    async def process_tool_result(
        self,
        result: ToolResultData,
    ) -> ToolResultUpdate | None: ...

    async def on_event(
        self,
        event: AgentEvent,
        *,
        session: Session | None = None,
    ) -> None: ...

    def run_scope(self) -> AbstractAsyncContextManager[RunControl]: ...


class NullHooks:
    """No-op hook host used when the runtime has no integrations attached."""

    __slots__ = ("_run_control",)

    def __init__(self) -> None:
        self._run_control = NullRunControl()

    async def resolve_input(
        self,
        text: str,
        *,
        source: str = "interactive",
    ) -> InputResolution | None:
        return None

    async def prepare_context(self, messages: list[Message]) -> list[Message]:
        return list(messages)

    async def authorize_tool_call(
        self,
        request: ToolCallRequest,
    ) -> ToolAuthorization | None:
        return None

    async def process_tool_result(
        self,
        result: ToolResultData,
    ) -> ToolResultUpdate | None:
        return None

    async def on_event(
        self,
        event: AgentEvent,
        *,
        session: Session | None = None,
    ) -> None:
        return None

    @asynccontextmanager
    async def run_scope(self) -> AsyncIterator[RunControl]:
        yield self._run_control
