"""Extension event types for hooks and modifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.core.message import Message


# --- Tool Hook Events ---


@dataclass(slots=True)
class ToolCallEvent:
    """Fired before tool execution - extensions can block."""

    tool_name: str = ""
    tool_call_id: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    type: str = field(default="tool_call", init=False)


@dataclass(slots=True)
class ToolCallResult:
    """Return from tool_call handler to block execution."""

    block: bool = False
    reason: str | None = None


@dataclass(slots=True)
class ToolResultEvent:
    """Fired after tool execution - extensions can modify result."""

    tool_name: str = ""
    tool_call_id: str = ""
    content: str = ""
    is_error: bool = False
    type: str = field(default="tool_result", init=False)


@dataclass(slots=True)
class ToolResultModification:
    """Return from tool_result handler to modify the result."""

    content: str | None = None
    is_error: bool | None = None


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
class InputResult:
    """Return from input handler to transform or block input."""

    text: str | None = None  # Modified text, or None to keep original
    block: bool = False
    reason: str | None = None
