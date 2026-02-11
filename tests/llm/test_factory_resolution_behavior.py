"""Behavior tests for LLM provider factory resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.config import Config
from agent.llm.factory import create_provider, resolve_provider_config


def resolve_from_config(config: Config):
    return resolve_provider_config(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        provider_overrides=config.provider_overrides(),
    )


def test_openai_codex_provider_prefers_oauth_env_and_default_model(monkeypatch):
    oauth_token = "a.b.c"
    monkeypatch.setenv("OPENAI_CODEX_OAUTH_TOKEN", oauth_token)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = Config(provider="openai-codex", model=None)
    provider_config = resolve_from_config(config)

    assert provider_config.model == "gpt-5.3-codex"
    assert provider_config.api_key == oauth_token


def test_openai_codex_provider_does_not_use_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_CODEX_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-123")

    config = Config(provider="openai-codex", model="gpt-5-codex")
    provider_config = resolve_from_config(config)

    assert provider_config.api_key is None


def test_resolve_provider_config_uses_override_values_for_anthropic_when_model_unset():
    override = SimpleNamespace(
        base_url="https://anthropic.example",
        model="claude-override",
        api_key="sk-ant-override",
    )

    provider_config = resolve_provider_config(
        provider="anthropic",
        model=None,
        api_key=None,
        base_url=None,
        provider_overrides={"anthropic": override},
    )

    assert provider_config.base_url == "https://anthropic.example"
    assert provider_config.model == "claude-override"
    assert provider_config.api_key == "sk-ant-override"


def test_resolve_provider_config_prefers_explicit_model_over_provider_override():
    override = SimpleNamespace(
        base_url="https://anthropic.example",
        model="claude-override",
        api_key="sk-ant-override",
    )

    provider_config = resolve_provider_config(
        provider="anthropic",
        model="claude-sonnet-4-5",
        api_key=None,
        base_url=None,
        provider_overrides={"anthropic": override},
    )

    assert provider_config.model == "claude-sonnet-4-5"


def test_create_provider_rejects_invalid_provider_model_pair():
    with pytest.raises(ValueError, match="not valid for provider"):
        create_provider(
            provider="openai",
            model="claude-sonnet-4-5",
            api_key="sk-test",
            base_url=None,
            temperature=0.7,
            max_output_tokens=4096,
            provider_overrides=None,
        )


def test_create_provider_rejects_invalid_explicit_model_even_with_valid_override():
    override = SimpleNamespace(
        base_url="https://api.openai.com",
        model="gpt-4o",
        api_key="sk-test",
    )

    with pytest.raises(ValueError, match="not valid for provider"):
        create_provider(
            provider="openai",
            model="claude-sonnet-4-5",
            api_key=None,
            base_url=None,
            temperature=0.7,
            max_output_tokens=4096,
            provider_overrides={"openai": override},
        )


def test_create_provider_uses_provider_default_model_when_unset():
    provider = create_provider(
        provider="anthropic",
        model=None,
        api_key="sk-ant-test",
        base_url=None,
        temperature=0.7,
        max_output_tokens=4096,
        provider_overrides=None,
    )

    assert provider.name == "anthropic"
    assert provider.model == "claude-opus-4-6"


def test_resolve_provider_config_uses_openai_default_model_when_unset():
    provider_config = resolve_provider_config(
        provider="openai",
        model=None,
        api_key="sk-test",
        base_url=None,
        provider_overrides=None,
    )

    assert provider_config.model == "gpt-5.2"


def test_create_provider_builds_openai_provider_instance():
    provider = create_provider(
        provider="openai",
        model="gpt-4o",
        api_key="sk-test",
        base_url=None,
        temperature=0.7,
        max_output_tokens=4096,
        provider_overrides=None,
    )

    assert provider.name == "openai"
    assert provider.model == "gpt-4o"


def test_create_provider_builds_openai_compatible_provider_with_override():
    override = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5",
        api_key="sk-openrouter",
    )
    provider = create_provider(
        provider="openrouter",
        model="openai/gpt-5",
        api_key=None,
        base_url=None,
        temperature=0.7,
        max_output_tokens=4096,
        provider_overrides={"openrouter": override},
    )

    assert provider.name == "openrouter"
    assert provider.model == "openai/gpt-5"


def test_resolve_provider_config_prefers_provider_specific_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("AGENT_API_KEY", "sk-agent")

    provider_config = resolve_provider_config(
        provider="openai",
        model=None,
        api_key=None,
        base_url=None,
        provider_overrides=None,
    )

    assert provider_config.api_key == "sk-openai"


def test_resolve_provider_config_openai_ignores_anthropic_shaped_agent_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AGENT_API_KEY", "sk-ant-123")

    provider_config = resolve_provider_config(
        provider="openai",
        model=None,
        api_key="sk-openai-default",
        base_url=None,
        provider_overrides=None,
    )

    assert provider_config.api_key == "sk-openai-default"


def test_resolve_provider_config_anthropic_rejects_non_anthropic_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AGENT_API_KEY", "sk-openai")

    provider_config = resolve_provider_config(
        provider="anthropic",
        model=None,
        api_key="sk-openai-default",
        base_url=None,
        provider_overrides=None,
    )

    assert provider_config.api_key is None


def test_resolve_provider_config_openai_codex_uses_agent_oauth_token(monkeypatch):
    monkeypatch.delenv("OPENAI_CODEX_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("AGENT_API_KEY", "header.payload.signature")

    provider_config = resolve_provider_config(
        provider="openai-codex",
        model=None,
        api_key=None,
        base_url=None,
        provider_overrides=None,
    )

    assert provider_config.api_key == "header.payload.signature"


def test_resolve_provider_config_prefers_explicit_base_url_over_provider_override():
    override = SimpleNamespace(
        base_url="https://override.example",
        model="openai/gpt-5",
        api_key="sk-openrouter",
    )

    provider_config = resolve_provider_config(
        provider="openrouter",
        model="openai/gpt-5",
        api_key=None,
        base_url="https://explicit.example",
        provider_overrides={"openrouter": override},
    )

    assert provider_config.base_url == "https://explicit.example"


def test_resolve_provider_config_uses_default_for_ollama():
    provider_config = resolve_provider_config(
        provider="ollama",
        model="llama3.2",
        api_key=None,
        base_url=None,
        provider_overrides=None,
    )

    assert provider_config.base_url == "http://localhost:11434"
    assert provider_config.model == "llama3.2"
    assert provider_config.api_key == "ollama"


def test_resolve_provider_config_unknown_provider_falls_back_to_openai_base_url():
    provider_config = resolve_provider_config(
        provider="custom-provider",
        model="my-model",
        api_key=None,
        base_url=None,
        provider_overrides=None,
    )

    assert provider_config.base_url == "https://api.openai.com"
    assert provider_config.model == "my-model"


def test_create_provider_requires_model_for_openai_compatible_when_unset():
    with pytest.raises(ValueError, match="Model is required for provider 'openrouter'"):
        create_provider(
            provider="openrouter",
            model=None,
            api_key="sk-openrouter",
            base_url=None,
            temperature=0.7,
            max_output_tokens=4096,
            provider_overrides=None,
        )
