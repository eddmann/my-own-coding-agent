"""Core runtime settings and thinking-level policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class ThinkingLevel(StrEnum):
    """Thinking/reasoning effort level for supported models."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"  # Extended high - very large budget, select models only


# Standard thinking levels (most reasoning models)
THINKING_LEVELS: list[ThinkingLevel] = [
    ThinkingLevel.OFF,
    ThinkingLevel.MINIMAL,
    ThinkingLevel.LOW,
    ThinkingLevel.MEDIUM,
    ThinkingLevel.HIGH,
]

# Extended levels for models that support xhigh
THINKING_LEVELS_WITH_XHIGH: list[ThinkingLevel] = [
    *THINKING_LEVELS,
    ThinkingLevel.XHIGH,
]

# Default token budgets for thinking levels (used by Anthropic)
THINKING_BUDGETS: dict[ThinkingLevel, int] = {
    ThinkingLevel.MINIMAL: 1024,
    ThinkingLevel.LOW: 2048,
    ThinkingLevel.MEDIUM: 8192,
    ThinkingLevel.HIGH: 16384,
    ThinkingLevel.XHIGH: 32768,
}


def get_available_thinking_levels(
    model_id: str,
    provider: str | None = None,
) -> list[ThinkingLevel]:
    """Get available thinking levels for a model."""
    from agent.llm.models import supports_reasoning, supports_xhigh

    if not supports_reasoning(model_id, provider=provider):
        return [ThinkingLevel.OFF]

    if supports_xhigh(model_id, provider=provider):
        return THINKING_LEVELS_WITH_XHIGH

    return THINKING_LEVELS


def clamp_thinking_level(level: ThinkingLevel, available: list[ThinkingLevel]) -> ThinkingLevel:
    """Clamp a thinking level to the available levels."""
    if level in available:
        return level

    all_levels = list(ThinkingLevel)
    level_idx = all_levels.index(level)

    for check_level in reversed(all_levels[: level_idx + 1]):
        if check_level in available:
            return check_level

    return available[0] if available else ThinkingLevel.OFF


@dataclass(slots=True)
class AgentSettings:
    """Core runtime settings required by the Agent."""

    context_max_tokens: int = 128000
    max_output_tokens: int = 8192
    temperature: float = 0.7
    thinking_level: ThinkingLevel = ThinkingLevel.OFF
    session_dir: Path = field(default_factory=lambda: Path.home() / ".agent" / "sessions")
    skills_dirs: list[Path] = field(default_factory=list)
    extensions: list[Path] = field(default_factory=list)
    prompt_template_dirs: list[Path] = field(default_factory=list)
    context_file_paths: list[Path] = field(default_factory=list)
    custom_system_prompt: str | None = None
    append_system_prompt: str | None = None
