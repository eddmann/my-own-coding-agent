"""Model registry with capability flags.

Defines known models and their capabilities, aligned with the internal registry.
This avoids brittle name-prefix matching for capability detection.
"""

from dataclasses import dataclass
from typing import Literal

Provider = Literal["anthropic", "openai", "openai-compat"]


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Model capability information."""

    id: str
    provider: Provider
    reasoning: bool = False  # Supports thinking/reasoning
    xhigh: bool = False  # Supports xhigh thinking level (very high budget)
    max_output_tokens: int = 8192


# Known models with their capabilities
# Models not in this registry will use provider defaults
MODELS: dict[str, ModelInfo] = {
    # =========================================================================
    # ANTHROPIC CLAUDE MODELS
    # =========================================================================
    # Claude 3 models - NO extended thinking
    "claude-3-haiku-20240307": ModelInfo(
        id="claude-3-haiku-20240307",
        provider="anthropic",
        reasoning=False,
        max_output_tokens=4096,
    ),
    "claude-3-sonnet-20240229": ModelInfo(
        id="claude-3-sonnet-20240229",
        provider="anthropic",
        reasoning=False,
        max_output_tokens=4096,
    ),
    "claude-3-opus-20240229": ModelInfo(
        id="claude-3-opus-20240229",
        provider="anthropic",
        reasoning=False,
        max_output_tokens=4096,
    ),
    # Claude 3.5 models - NO extended thinking
    "claude-3-5-haiku-20241022": ModelInfo(
        id="claude-3-5-haiku-20241022",
        provider="anthropic",
        reasoning=False,
        max_output_tokens=8192,
    ),
    "claude-3-5-haiku-latest": ModelInfo(
        id="claude-3-5-haiku-latest",
        provider="anthropic",
        reasoning=False,
        max_output_tokens=8192,
    ),
    "claude-3-5-sonnet-20240620": ModelInfo(
        id="claude-3-5-sonnet-20240620",
        provider="anthropic",
        reasoning=False,
        max_output_tokens=8192,
    ),
    "claude-3-5-sonnet-20241022": ModelInfo(
        id="claude-3-5-sonnet-20241022",
        provider="anthropic",
        reasoning=False,
        max_output_tokens=8192,
    ),
    # Claude 3.7 models - HAS extended thinking
    "claude-3-7-sonnet-20250219": ModelInfo(
        id="claude-3-7-sonnet-20250219",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=16384,
    ),
    "claude-3-7-sonnet-latest": ModelInfo(
        id="claude-3-7-sonnet-latest",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=16384,
    ),
    # Claude 4 Haiku - HAS extended thinking
    "claude-haiku-4-5": ModelInfo(
        id="claude-haiku-4-5",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    "claude-haiku-4-5-20251001": ModelInfo(
        id="claude-haiku-4-5-20251001",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    # Claude 4 Sonnet - HAS extended thinking
    "claude-sonnet-4-0": ModelInfo(
        id="claude-sonnet-4-0",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    "claude-sonnet-4-20250514": ModelInfo(
        id="claude-sonnet-4-20250514",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    "claude-sonnet-4-5": ModelInfo(
        id="claude-sonnet-4-5",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    "claude-sonnet-4-5-20250929": ModelInfo(
        id="claude-sonnet-4-5-20250929",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    # Aliases
    "claude-sonnet-4": ModelInfo(
        id="claude-sonnet-4",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    # Claude 4 Opus - HAS extended thinking
    "claude-opus-4-0": ModelInfo(
        id="claude-opus-4-0",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    "claude-opus-4-20250514": ModelInfo(
        id="claude-opus-4-20250514",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    "claude-opus-4-1": ModelInfo(
        id="claude-opus-4-1",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    "claude-opus-4-1-20250805": ModelInfo(
        id="claude-opus-4-1-20250805",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    "claude-opus-4-5": ModelInfo(
        id="claude-opus-4-5",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    "claude-opus-4-5-20251101": ModelInfo(
        id="claude-opus-4-5-20251101",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    "claude-opus-4-6": ModelInfo(
        id="claude-opus-4-6",
        provider="anthropic",
        reasoning=True,
        xhigh=True,  # Anthropic adaptive thinking supports "max" effort
        max_output_tokens=128000,
    ),
    # Aliases
    "claude-opus-4": ModelInfo(
        id="claude-opus-4",
        provider="anthropic",
        reasoning=True,
        max_output_tokens=64000,
    ),
    # =========================================================================
    # OPENAI MODELS
    # =========================================================================
    # GPT-4 family - NO reasoning
    "gpt-4": ModelInfo(
        id="gpt-4",
        provider="openai",
        reasoning=False,
        max_output_tokens=8192,
    ),
    "gpt-4-turbo": ModelInfo(
        id="gpt-4-turbo",
        provider="openai",
        reasoning=False,
        max_output_tokens=4096,
    ),
    # GPT-4.1 family - NO reasoning
    "gpt-4.1": ModelInfo(
        id="gpt-4.1",
        provider="openai",
        reasoning=False,
        max_output_tokens=16384,
    ),
    "gpt-4.1-mini": ModelInfo(
        id="gpt-4.1-mini",
        provider="openai",
        reasoning=False,
        max_output_tokens=16384,
    ),
    "gpt-4.1-nano": ModelInfo(
        id="gpt-4.1-nano",
        provider="openai",
        reasoning=False,
        max_output_tokens=16384,
    ),
    # GPT-4o family - NO reasoning
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        provider="openai",
        reasoning=False,
        max_output_tokens=16384,
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        reasoning=False,
        max_output_tokens=16384,
    ),
    "gpt-4o-2024-05-13": ModelInfo(
        id="gpt-4o-2024-05-13",
        provider="openai",
        reasoning=False,
        max_output_tokens=4096,
    ),
    "gpt-4o-2024-08-06": ModelInfo(
        id="gpt-4o-2024-08-06",
        provider="openai",
        reasoning=False,
        max_output_tokens=16384,
    ),
    "gpt-4o-2024-11-20": ModelInfo(
        id="gpt-4o-2024-11-20",
        provider="openai",
        reasoning=False,
        max_output_tokens=16384,
    ),
    # GPT-5 family - HAS reasoning
    "gpt-5": ModelInfo(
        id="gpt-5",
        provider="openai",
        reasoning=True,
        max_output_tokens=32768,
    ),
    "gpt-5-mini": ModelInfo(
        id="gpt-5-mini",
        provider="openai",
        reasoning=True,
        max_output_tokens=16384,
    ),
    "gpt-5-nano": ModelInfo(
        id="gpt-5-nano",
        provider="openai",
        reasoning=True,
        max_output_tokens=16384,
    ),
    "gpt-5-pro": ModelInfo(
        id="gpt-5-pro",
        provider="openai",
        reasoning=True,
        max_output_tokens=65536,
    ),
    "gpt-5-codex": ModelInfo(
        id="gpt-5-codex",
        provider="openai",
        reasoning=True,
        max_output_tokens=32768,
    ),
    # Special case: gpt-5-chat-latest has NO reasoning
    "gpt-5-chat-latest": ModelInfo(
        id="gpt-5-chat-latest",
        provider="openai",
        reasoning=False,
        max_output_tokens=32768,
    ),
    # GPT-5.1 family - HAS reasoning
    "gpt-5.1": ModelInfo(
        id="gpt-5.1",
        provider="openai",
        reasoning=True,
        max_output_tokens=32768,
    ),
    "gpt-5.1-chat-latest": ModelInfo(
        id="gpt-5.1-chat-latest",
        provider="openai",
        reasoning=True,
        max_output_tokens=32768,
    ),
    "gpt-5.1-codex": ModelInfo(
        id="gpt-5.1-codex",
        provider="openai",
        reasoning=True,
        max_output_tokens=32768,
    ),
    "gpt-5.1-codex-mini": ModelInfo(
        id="gpt-5.1-codex-mini",
        provider="openai",
        reasoning=True,
        max_output_tokens=32768,
    ),
    "gpt-5.1-codex-max": ModelInfo(
        id="gpt-5.1-codex-max",
        provider="openai",
        reasoning=True,
        xhigh=True,  # Supports xhigh
        max_output_tokens=65536,
    ),
    # GPT-5.2 family - HAS reasoning + xhigh
    "gpt-5.2": ModelInfo(
        id="gpt-5.2",
        provider="openai",
        reasoning=True,
        xhigh=True,  # Supports xhigh
        max_output_tokens=32768,
    ),
    "gpt-5.2-chat-latest": ModelInfo(
        id="gpt-5.2-chat-latest",
        provider="openai",
        reasoning=True,
        max_output_tokens=32768,
    ),
    "gpt-5.2-codex": ModelInfo(
        id="gpt-5.2-codex",
        provider="openai",
        reasoning=True,
        xhigh=True,  # Supports xhigh
        max_output_tokens=65536,
    ),
    "gpt-5.2-pro": ModelInfo(
        id="gpt-5.2-pro",
        provider="openai",
        reasoning=True,
        max_output_tokens=65536,
    ),
    # GPT-5.3 family - HAS reasoning + xhigh
    "gpt-5.3": ModelInfo(
        id="gpt-5.3",
        provider="openai",
        reasoning=True,
        xhigh=True,  # Supports xhigh
        max_output_tokens=128000,
    ),
    "gpt-5.3-codex": ModelInfo(
        id="gpt-5.3-codex",
        provider="openai",
        reasoning=True,
        xhigh=True,  # Supports xhigh
        max_output_tokens=128000,
    ),
    # O1 family - HAS reasoning
    "o1": ModelInfo(
        id="o1",
        provider="openai",
        reasoning=True,
        max_output_tokens=100000,
    ),
    "o1-pro": ModelInfo(
        id="o1-pro",
        provider="openai",
        reasoning=True,
        max_output_tokens=100000,
    ),
    "o1-preview": ModelInfo(
        id="o1-preview",
        provider="openai",
        reasoning=True,
        max_output_tokens=32768,
    ),
    "o1-mini": ModelInfo(
        id="o1-mini",
        provider="openai",
        reasoning=True,
        max_output_tokens=65536,
    ),
    # O3 family - HAS reasoning
    "o3": ModelInfo(
        id="o3",
        provider="openai",
        reasoning=True,
        max_output_tokens=100000,
    ),
    "o3-mini": ModelInfo(
        id="o3-mini",
        provider="openai",
        reasoning=True,
        max_output_tokens=65536,
    ),
    "o3-pro": ModelInfo(
        id="o3-pro",
        provider="openai",
        reasoning=True,
        max_output_tokens=100000,
    ),
    "o3-deep-research": ModelInfo(
        id="o3-deep-research",
        provider="openai",
        reasoning=True,
        max_output_tokens=100000,
    ),
    # O4 family - HAS reasoning
    "o4-mini": ModelInfo(
        id="o4-mini",
        provider="openai",
        reasoning=True,
        max_output_tokens=65536,
    ),
    "o4-mini-deep-research": ModelInfo(
        id="o4-mini-deep-research",
        provider="openai",
        reasoning=True,
        max_output_tokens=65536,
    ),
    # OpenAI coding models - HAS reasoning
    "codex-mini-latest": ModelInfo(
        id="codex-mini-latest",
        provider="openai",
        reasoning=True,
        max_output_tokens=32768,
    ),
}


# Models that support xhigh thinking level.
XHIGH_MODELS = {"gpt-5.1-codex-max", "gpt-5.2", "gpt-5.2-codex", "gpt-5.3", "gpt-5.3-codex"}


def resolve_capability_provider(provider: str | None) -> Provider | None:
    """Map config/runtime provider names to capability provider families."""
    if provider == "anthropic":
        return "anthropic"
    if provider in {"openai", "openai-codex"}:
        return "openai"
    if provider:
        # Everything else currently routes through OpenAI-compatible transport.
        return "openai-compat"
    return None


def _normalize_capability_model_id(model_id: str) -> tuple[str, str | None]:
    """Normalize model IDs for capability checks.

    Returns:
        Tuple of (normalized_model_id, optional_provider_prefix)
    """
    normalized = model_id.strip().lower()
    prefix: str | None = None
    if "/" in normalized:
        prefix, normalized = normalized.split("/", 1)

    # Normalize common alias variants to registry keys.
    normalized = normalized.replace("claude-opus-4.6", "claude-opus-4-6")
    return normalized, prefix


def _effective_capability_provider(
    provider: str | None,
    model_prefix: str | None,
) -> Provider | None:
    capability_provider = resolve_capability_provider(provider)
    if capability_provider != "openai-compat":
        return capability_provider

    # OpenAI-compatible routes often encode upstream provider in the model ID.
    if model_prefix == "openai":
        return "openai"
    if model_prefix == "anthropic":
        return "anthropic"
    return capability_provider


def _looks_like_opus_46(model_id: str) -> bool:
    return "opus-4-6" in model_id or "opus-4.6" in model_id


def model_supports_xhigh(model_id: str) -> bool:
    """Return model-level xhigh capability without provider policy."""
    normalized_model, _ = _normalize_capability_model_id(model_id)
    if normalized_model in XHIGH_MODELS:
        return True
    info = MODELS.get(normalized_model)
    return bool(info and info.xhigh)


def provider_allows_xhigh(model_id: str, provider: str | None = None) -> bool:
    """Apply provider/API policy on top of model-level xhigh capability."""
    normalized_model, _ = _normalize_capability_model_id(model_id)
    capability_provider = resolve_capability_provider(provider)

    # Anthropic adaptive effort "max" is only available via native Anthropic API.
    if _looks_like_opus_46(normalized_model):
        return capability_provider == "anthropic"

    return model_supports_xhigh(normalized_model)


def get_model_info(model_id: str) -> ModelInfo | None:
    """Get model info from registry with normalized lookup."""
    info = MODELS.get(model_id)
    if info is not None:
        return info

    normalized_model, _ = _normalize_capability_model_id(model_id)
    return MODELS.get(normalized_model)


def supports_reasoning(model_id: str, provider: str | None = None) -> bool:
    """Check if a model supports reasoning/thinking.

    First checks the registry. If not found, falls back to
    name-based heuristics for known model families.

    Args:
        model_id: The model ID to check
        provider: Optional provider hint for fallback detection

    Returns:
        True if model supports reasoning
    """
    normalized_model, model_prefix = _normalize_capability_model_id(model_id)
    effective_provider = _effective_capability_provider(provider, model_prefix)

    # Check registry first
    info = get_model_info(normalized_model)
    if info is not None:
        return info.reasoning

    # Fallback: name-based heuristics for models not in registry
    model_lower = normalized_model

    # Anthropic: claude-3.7+, claude-4+ support reasoning
    if effective_provider == "anthropic" or "claude" in model_lower:
        # Claude 3.7+ supports extended thinking
        if "claude-3-7" in model_lower or "claude-3.7" in model_lower:
            return True
        # Claude 4+ models support extended thinking
        if "claude-4" in model_lower or "claude-haiku-4" in model_lower:
            return True
        # Older models don't
        return "claude-sonnet-4" in model_lower or "claude-opus-4" in model_lower

    # OpenAI: o1/o3/o4 and gpt-5+ support reasoning
    if effective_provider in ("openai", "openai-compat"):
        if model_lower.startswith(("o1", "o3", "o4")):
            return True
        # gpt-5+ supports reasoning (except gpt-5-chat-latest)
        if model_lower.startswith("gpt-5"):
            return "chat-latest" not in model_lower
        return bool(model_lower.startswith("codex"))

    # Default: no reasoning support
    return False


def supports_xhigh(model_id: str, provider: str | None = None) -> bool:
    """Check if a model supports xhigh thinking level.

    Currently only certain OpenAI models support this,
    based on the explicit xhigh registry.

    Args:
        model_id: The model ID to check

    Returns:
        True if model supports xhigh thinking
    """
    if provider is None:
        return model_supports_xhigh(model_id)
    return provider_allows_xhigh(model_id, provider)
