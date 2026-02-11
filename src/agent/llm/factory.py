"""Factory for constructing LLM providers from runtime config."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from agent.llm.defaults import default_model_for_provider

if TYPE_CHECKING:
    from agent.llm.provider import LLMProvider


class ProviderOverride(Protocol):
    """Shape for provider override values supplied by delivery config."""

    base_url: str
    model: str | None
    api_key: str | None


ProviderOverrides = Mapping[str, ProviderOverride]


@dataclass(slots=True)
class ResolvedProviderConfig:
    """Resolved settings for constructing a provider implementation."""

    base_url: str
    model: str | None
    api_key: str | None = None


def _env_provider_key(provider: str) -> str | None:
    mapping = {
        "openai": "OPENAI_API_KEY",
        "openai-codex": "OPENAI_CODEX_OAUTH_TOKEN",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    env_var = mapping.get(provider)
    if env_var:
        return os.environ.get(env_var)
    return None


def _is_anthropic_key(value: str | None) -> bool:
    return value is not None and value.startswith("sk-ant-")


def _is_openai_oauth_token(value: str | None) -> bool:
    if not value:
        return False
    parts = value.split(".")
    return len(parts) == 3 and all(part for part in parts)


def _resolve_api_key(default: str | None, provider: str) -> str | None:
    env_key = _env_provider_key(provider)
    if env_key:
        return env_key

    agent_key = os.environ.get("AGENT_API_KEY")

    if provider == "anthropic":
        if _is_anthropic_key(agent_key):
            return agent_key
        if _is_anthropic_key(default):
            return default
        return None

    if provider == "openai":
        if agent_key and not _is_anthropic_key(agent_key):
            return agent_key
        if default and not _is_anthropic_key(default):
            return default
        return None

    if provider == "openai-codex":
        if _is_openai_oauth_token(agent_key):
            return agent_key
        if _is_openai_oauth_token(default):
            return default
        return None

    if agent_key:
        return agent_key
    return default


def _resolve_model(provider: str, model: str | None, fallback_model: str | None) -> str | None:
    if model is not None:
        return model
    if fallback_model is not None:
        return fallback_model
    return default_model_for_provider(provider)


def resolve_provider_config(
    *,
    provider: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    provider_overrides: ProviderOverrides | None = None,
) -> ResolvedProviderConfig:
    """Resolve concrete provider config from flat provider bootstrap params."""
    if provider_overrides and provider in provider_overrides:
        override = provider_overrides[provider]
        return ResolvedProviderConfig(
            base_url=base_url or override.base_url,
            model=_resolve_model(provider, model, override.model),
            api_key=_resolve_api_key(api_key or override.api_key, provider),
        )

    default_configs = {
        "openai": ResolvedProviderConfig(
            base_url="https://api.openai.com",
            model=default_model_for_provider("openai"),
            api_key=api_key,
        ),
        "openai-codex": ResolvedProviderConfig(
            base_url="https://chatgpt.com/backend-api",
            model=default_model_for_provider("openai-codex"),
            api_key=api_key,
        ),
        "anthropic": ResolvedProviderConfig(
            base_url="https://api.anthropic.com",
            model=default_model_for_provider("anthropic"),
            api_key=api_key,
        ),
        "ollama": ResolvedProviderConfig(
            base_url="http://localhost:11434",
            model=None,
            api_key="ollama",
        ),
        "openrouter": ResolvedProviderConfig(
            base_url="https://openrouter.ai/api",
            model=None,
            api_key=api_key,
        ),
        "groq": ResolvedProviderConfig(
            base_url="https://api.groq.com/openai/v1",
            model=None,
            api_key=api_key,
        ),
    }

    if provider in default_configs:
        provider_config = default_configs[provider]
        return ResolvedProviderConfig(
            base_url=base_url or provider_config.base_url,
            model=_resolve_model(provider, model, provider_config.model),
            api_key=_resolve_api_key(api_key or provider_config.api_key, provider),
        )

    return ResolvedProviderConfig(
        base_url=base_url or "https://api.openai.com",
        model=_resolve_model(provider, model, None),
        api_key=_resolve_api_key(api_key, provider),
    )


def create_provider(
    *,
    provider: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    temperature: float,
    max_output_tokens: int,
    provider_overrides: ProviderOverrides | None = None,
) -> LLMProvider:
    """Create a concrete provider instance from flat provider bootstrap params."""
    from agent.llm.models import is_model_valid_for_provider

    prov_config = resolve_provider_config(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        provider_overrides=provider_overrides,
    )
    if prov_config.model is None:
        raise ValueError(
            f"Model is required for provider '{provider}'. "
            "Set --model or configure provider-specific model override."
        )

    if not is_model_valid_for_provider(prov_config.model, provider):
        raise ValueError(f"Model '{prov_config.model}' is not valid for provider '{provider}'")

    if provider == "anthropic":
        from agent.llm.anthropic import AnthropicProvider

        return AnthropicProvider(
            api_key=prov_config.api_key or "",
            model=prov_config.model,
            max_tokens=max_output_tokens,
        )

    if provider == "openai":
        from agent.llm.openai import OpenAIProvider

        return OpenAIProvider(
            api_key=prov_config.api_key or "",
            model=prov_config.model,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

    if provider == "openai-codex":
        from agent.llm.openai_codex import OpenAICodexProvider

        return OpenAICodexProvider(
            api_key=prov_config.api_key or "",
            model=prov_config.model,
            temperature=temperature,
            max_tokens=max_output_tokens,
            base_url=prov_config.base_url,
        )

    from agent.llm.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        name=provider,
        base_url=prov_config.base_url,
        api_key=prov_config.api_key or "",
        model=prov_config.model,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
