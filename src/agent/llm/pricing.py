"""Model pricing for cost calculation (per 1M tokens)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.llm.events import Usage


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Pricing per million tokens for a model."""

    input: float  # $/MTok
    output: float  # $/MTok
    cache_read: float = 0.0  # $/MTok (prompt caching)
    cache_write: float = 0.0  # $/MTok (prompt caching)


# Anthropic pricing (as of 2026)
ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    # Opus 4.5
    "claude-opus-4-5": ModelPricing(5.0, 25.0, 0.5, 6.25),
    # Opus 4
    "claude-opus-4": ModelPricing(15.0, 75.0, 1.5, 18.75),
    # Sonnet 4.5
    "claude-sonnet-4-5": ModelPricing(3.0, 15.0, 0.3, 3.75),
    # Sonnet 4
    "claude-sonnet-4": ModelPricing(3.0, 15.0, 0.3, 3.75),
    # Haiku 4.5
    "claude-haiku-4-5": ModelPricing(1.0, 5.0, 0.1, 1.25),
    # Haiku 3.5
    "claude-3-5-haiku": ModelPricing(0.8, 4.0, 0.08, 1.0),
}

# OpenAI pricing (as of 2026)
OPENAI_PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(2.5, 10.0, 1.25, 2.5),
    "gpt-4o-mini": ModelPricing(0.15, 0.6, 0.075, 0.15),
    "o1": ModelPricing(15.0, 60.0, 7.5, 15.0),
    "o3": ModelPricing(2.0, 8.0, 1.0, 2.0),
    "o3-mini": ModelPricing(1.1, 4.4, 0.55, 1.1),
}


def get_pricing(model: str, provider: str) -> ModelPricing | None:
    """Get pricing for a model, with fuzzy matching.

    Args:
        model: Model name (e.g., "claude-sonnet-4-20250514")
        provider: Provider name ("anthropic" or "openai")

    Returns:
        ModelPricing if found, None otherwise
    """
    tables = ANTHROPIC_PRICING if provider == "anthropic" else OPENAI_PRICING

    # Exact match
    if model in tables:
        return tables[model]

    # Fuzzy match (e.g., "claude-sonnet-4-20250514" -> "claude-sonnet-4")
    for key in tables:
        if key in model or model.startswith(key):
            return tables[key]

    return None


def calculate_cost(usage: Usage, pricing: ModelPricing | None) -> float:
    """Calculate total cost from usage and pricing.

    Args:
        usage: Token usage from a request
        pricing: Pricing for the model

    Returns:
        Total cost in dollars
    """
    if not pricing:
        return 0.0
    return (
        (usage.input / 1_000_000) * pricing.input
        + (usage.output / 1_000_000) * pricing.output
        + (usage.cache_read / 1_000_000) * pricing.cache_read
        + (usage.cache_write / 1_000_000) * pricing.cache_write
    )
