"""Behavior tests for model capability policy."""

from agent.core.settings import ThinkingLevel, get_available_thinking_levels
from agent.llm.models import (
    is_model_valid_for_provider,
    resolve_capability_provider,
    supports_reasoning,
    supports_xhigh,
)


class TestModelCapabilities:
    """Tests for model capability policy and normalization."""

    def test_supports_xhigh_for_gpt_53_family(self) -> None:
        assert supports_xhigh("gpt-5.3", provider="openai")
        assert supports_xhigh("gpt-5.3-codex", provider="openai")

    def test_supports_xhigh_for_opus_46_only_on_anthropic(self) -> None:
        assert supports_xhigh("claude-opus-4-6", provider="anthropic")
        assert not supports_xhigh("claude-opus-4-6", provider="openai-compat")
        assert not supports_xhigh("anthropic/claude-opus-4.6", provider="openai-compat")

    def test_openai_compat_prefix_supports_reasoning_and_xhigh_for_openai_models(self) -> None:
        assert supports_reasoning("openai/gpt-5.3-codex", provider="openai-compat")
        assert supports_xhigh("openai/gpt-5.3-codex", provider="openai-compat")

    def test_provider_resolution_for_capability_checks(self) -> None:
        assert resolve_capability_provider("openai") == "openai"
        assert resolve_capability_provider("openai-codex") == "openai"
        assert resolve_capability_provider("anthropic") == "anthropic"
        assert resolve_capability_provider("openrouter") == "openai-compat"
        assert resolve_capability_provider(None) is None

    def test_available_thinking_levels_are_provider_aware(self) -> None:
        anthropic_levels = get_available_thinking_levels("claude-opus-4-6", provider="anthropic")
        compat_levels = get_available_thinking_levels(
            "anthropic/claude-opus-4.6",
            provider="openai-compat",
        )

        assert ThinkingLevel.XHIGH in anthropic_levels
        assert ThinkingLevel.XHIGH not in compat_levels

    def test_model_validation_is_provider_aware(self) -> None:
        assert is_model_valid_for_provider("claude-sonnet-4-5", "anthropic")
        assert not is_model_valid_for_provider("claude-sonnet-4-5", "openai")
        assert is_model_valid_for_provider("gpt-5-codex", "openai-codex")
        assert not is_model_valid_for_provider("gpt-4o", "openai-codex")
