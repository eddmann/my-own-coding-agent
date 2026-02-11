"""Shared default model configuration for providers."""

DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
DEFAULT_OPENAI_CODEX_MODEL = "gpt-5.3-codex"

DEFAULT_MODEL_BY_PROVIDER = {
    "openai": DEFAULT_OPENAI_MODEL,
    "anthropic": DEFAULT_ANTHROPIC_MODEL,
    "openai-codex": DEFAULT_OPENAI_CODEX_MODEL,
}


def default_model_for_provider(provider: str) -> str | None:
    """Return default model for a provider family."""
    return DEFAULT_MODEL_BY_PROVIDER.get(provider)
