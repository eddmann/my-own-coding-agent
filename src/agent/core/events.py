"""Event types for the agent event system.

Events are emitted throughout the agent lifecycle to enable:
- Extensions to hook into agent behavior
- TUI to react to state changes
- Logging and observability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.core.message import Message, ToolResult


# --- Agent Lifecycle Events ---


@dataclass(slots=True)
class AgentStartEvent:
    """Emitted when the agent starts processing."""

    type: str = field(default="agent_start", init=False)


@dataclass(slots=True)
class AgentEndEvent:
    """Emitted when the agent completes processing."""

    messages: list[Message] = field(default_factory=list)
    type: str = field(default="agent_end", init=False)


# --- Turn Events ---


@dataclass(slots=True)
class TurnStartEvent:
    """Emitted at the start of each agent turn (before LLM call)."""

    turn_number: int = 0
    type: str = field(default="turn_start", init=False)


@dataclass(slots=True)
class TurnEndEvent:
    """Emitted at the end of each agent turn (after tool execution)."""

    message: Message | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    type: str = field(default="turn_end", init=False)


# --- Message Events ---


@dataclass(slots=True)
class MessageStartEvent:
    """Emitted when the assistant starts generating a message."""

    message_id: str = ""
    type: str = field(default="message_start", init=False)


@dataclass(slots=True)
class MessageUpdateEvent:
    """Emitted during message streaming with new content."""

    delta: str = ""
    message_id: str = ""
    type: str = field(default="message_update", init=False)


@dataclass(slots=True)
class MessageEndEvent:
    """Emitted when the assistant finishes generating a message."""

    message: Message | None = None
    type: str = field(default="message_end", init=False)


# --- Tool Events ---


@dataclass(slots=True)
class ToolExecutionStartEvent:
    """Emitted before tool execution begins."""

    tool_call_id: str = ""
    tool_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    type: str = field(default="tool_execution_start", init=False)


@dataclass(slots=True)
class ToolExecutionUpdateEvent:
    """Emitted during tool execution with progress updates."""

    tool_call_id: str = ""
    tool_name: str = ""
    partial_result: Any = None
    type: str = field(default="tool_execution_update", init=False)


@dataclass(slots=True)
class ToolExecutionEndEvent:
    """Emitted when tool execution completes."""

    tool_call_id: str = ""
    tool_name: str = ""
    result: Any = None
    is_error: bool = False
    type: str = field(default="tool_execution_end", init=False)


# --- Thinking Events ---


@dataclass(slots=True)
class ThinkingStartEvent:
    """Emitted when the model starts a thinking/reasoning block."""

    type: str = field(default="thinking_start", init=False)


@dataclass(slots=True)
class ThinkingDeltaEvent:
    """Emitted during thinking with new content."""

    delta: str = ""
    type: str = field(default="thinking_delta", init=False)


@dataclass(slots=True)
class ThinkingEndEvent:
    """Emitted when thinking/reasoning completes."""

    content: str = ""
    type: str = field(default="thinking_end", init=False)


# --- Context Events ---


@dataclass(slots=True)
class ContextCompactionEvent:
    """Emitted when context is compacted."""

    original_tokens: int = 0
    compacted_tokens: int = 0
    type: str = field(default="context_compaction", init=False)


# --- Model Events ---


@dataclass(slots=True)
class ModelSelectEvent:
    """Emitted when the active model changes."""

    provider: str = ""
    model: str = ""
    previous_provider: str | None = None
    previous_model: str | None = None
    source: str = "set"  # "set" | "cycle" | "restore"
    type: str = field(default="model_select", init=False)


# Union type for all events
AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ContextCompactionEvent
    | ModelSelectEvent
)
