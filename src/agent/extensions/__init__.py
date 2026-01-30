"""Extension system for runtime customization.

Extensions can:
- Hook into agent lifecycle events
- Block or modify tool calls
- Transform messages before LLM calls
- Register custom tools and commands
"""

from agent.extensions.api import ExtensionAPI, ExtensionContext
from agent.extensions.loader import ExtensionLoader
from agent.extensions.runner import ExtensionRunner
from agent.extensions.types import (
    ContextEvent,
    InputEvent,
    ToolCallEvent,
    ToolCallResult,
    ToolResultEvent,
    ToolResultModification,
)

__all__ = [
    "ExtensionAPI",
    "ExtensionContext",
    "ExtensionLoader",
    "ExtensionRunner",
    "ContextEvent",
    "InputEvent",
    "ToolCallEvent",
    "ToolCallResult",
    "ToolResultEvent",
    "ToolResultModification",
]
