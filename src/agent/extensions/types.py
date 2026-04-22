"""Extension event types for hooks and modifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent.runtime.hooks import (
    InputResolution,
    ToolAuthorization,
    ToolCallRequest,
    ToolResultData,
    ToolResultUpdate,
)

if TYPE_CHECKING:
    from agent.runtime.message import Message


# Stable extension event contract. These names are part of the supported API.
PUBLIC_EXTENSION_EVENTS: frozenset[str] = frozenset(
    {
        "input",
        "context",
        "tool_call",
        "tool_result",
        "session_start",
        "session_end",
        "turn_start",
        "turn_end",
        "agent_start",
        "agent_end",
        "model_select",
        "compaction",
    }
)

# Internal runtime/rendering events. Extensions may technically observe them,
# but they are not part of the stable public API and may change without notice.
INTERNAL_EXTENSION_EVENTS: frozenset[str] = frozenset(
    {
        "message_start",
        "message_update",
        "message_end",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_end",
    }
)


# --- Tool Hook Events ---


@dataclass(slots=True)
class ToolCallEvent(ToolCallRequest):
    """Fired before tool execution - extensions can block."""

    type: str = field(default="tool_call", init=False)


@dataclass(slots=True)
class ToolCallResult(ToolAuthorization):
    """Return from tool_call handler to block execution."""


@dataclass(slots=True)
class ToolResultEvent(ToolResultData):
    """Fired after tool execution - extensions can modify result."""

    type: str = field(default="tool_result", init=False)


@dataclass(slots=True)
class ToolResultModification(ToolResultUpdate):
    """Return from tool_result handler to modify the result."""


# --- Context Hook Events ---


@dataclass(slots=True)
class ContextEvent:
    """Fired before LLM call - extensions can modify messages."""

    messages: list[Message] = field(default_factory=list)
    type: str = field(default="context", init=False)


@dataclass(slots=True)
class ContextModification:
    """Return from context handler to modify messages."""

    messages: list[Message] | None = None


# --- Input Hook Events ---


@dataclass(slots=True)
class InputEvent:
    """Fired when user input is received - can transform or block."""

    text: str = ""
    source: str = "interactive"  # "interactive" | "extension" | "api"
    type: str = field(default="input", init=False)


@dataclass(slots=True)
class InputResult(InputResolution):
    """Return from input handler to transform, block, or handle input."""
