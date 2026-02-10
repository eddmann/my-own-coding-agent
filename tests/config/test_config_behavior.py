"""Behavior tests for config loading and precedence."""

from __future__ import annotations

import textwrap

from agent.config import Config, ProviderConfig
from agent.core.settings import ThinkingLevel


def test_legacy_max_tokens_maps_to_context_and_output_defaults(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    monkeypatch.setenv("AGENT_MAX_TOKENS", "10000")

    config = Config.load()

    assert config.context_max_tokens == 10000
    assert config.max_output_tokens == 8192


def test_config_to_agent_settings_projects_runtime_fields(temp_dir):
    skills_dir = temp_dir / "skills"
    ext_path = temp_dir / "ext.py"
    prompts_dir = temp_dir / "prompts"
    context_path = temp_dir / "AGENTS.md"

    config = Config(
        context_max_tokens=42,
        max_output_tokens=84,
        temperature=0.1,
        thinking_level=ThinkingLevel.LOW,
        session_dir=temp_dir / "sessions",
        skills_dirs=[skills_dir],
        extensions=[ext_path],
        prompt_template_dirs=[prompts_dir],
        context_file_paths=[context_path],
        custom_system_prompt="custom",
        append_system_prompt="append",
    )

    settings = config.to_agent_settings()

    assert settings.context_max_tokens == 42
    assert settings.max_output_tokens == 84
    assert settings.temperature == 0.1
    assert settings.thinking_level == ThinkingLevel.LOW
    assert settings.session_dir == temp_dir / "sessions"
    assert settings.skills_dirs == [skills_dir]
    assert settings.extensions == [ext_path]
    assert settings.prompt_template_dirs == [prompts_dir]
    assert settings.context_file_paths == [context_path]
    assert settings.custom_system_prompt == "custom"
    assert settings.append_system_prompt == "append"


def test_config_provider_overrides_returns_copy_of_mapping():
    config = Config(
        providers={
            "openrouter": ProviderConfig(
                base_url="https://openrouter.ai/api/v1",
                model="openai/gpt-5",
                api_key="sk-123",
            )
        }
    )

    overrides = config.provider_overrides()
    overrides["new-provider"] = ProviderConfig(
        base_url="https://example.com",
        model="test-model",
        api_key=None,
    )

    assert "openrouter" in config.providers
    assert "new-provider" not in config.providers


def test_config_load_reads_yaml_when_toml_missing(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    global_dir = home / ".agent"
    global_dir.mkdir()
    (global_dir / "config.yaml").write_text(
        textwrap.dedent(
            """
            provider: anthropic
            model: claude-sonnet-4-5
            """
        )
    )

    config = Config.load()

    assert config.provider == "anthropic"
    assert config.model == "claude-sonnet-4-5"


def test_config_load_reads_yml_when_other_formats_missing(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    global_dir = home / ".agent"
    global_dir.mkdir()
    (global_dir / "config.yml").write_text(
        textwrap.dedent(
            """
            temperature: 0.33
            """
        )
    )

    config = Config.load()

    assert config.temperature == 0.33


def test_config_invalid_thinking_level_falls_back_to_off(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    global_dir = home / ".agent"
    global_dir.mkdir()
    (global_dir / "config.toml").write_text('thinking_level = "not-a-level"\n')

    config = Config.load()

    assert config.thinking_level == ThinkingLevel.OFF


def test_config_provider_overrides_defaults_missing_provider_fields(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    project_dir = project / ".agent"
    project_dir.mkdir()
    (project_dir / "config.toml").write_text(
        textwrap.dedent(
            """
            [providers.openrouter]
            """
        )
    )

    config = Config.load()
    override = config.providers["openrouter"]

    assert override.base_url == "https://api.openai.com"
    assert override.model == "gpt-4o"
    assert override.api_key is None


def test_config_precedence_is_env_then_project_then_global(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    global_dir = home / ".agent"
    project_dir = project / ".agent"
    global_dir.mkdir()
    project_dir.mkdir()

    (global_dir / "config.toml").write_text(
        textwrap.dedent(
            """
            provider = "openai"
            model = "global-model"
            temperature = 0.1
            """
        )
    )

    (project_dir / "config.toml").write_text(
        textwrap.dedent(
            """
            model = "project-model"
            temperature = 0.2
            """
        )
    )

    monkeypatch.setenv("AGENT_MODEL", "env-model")

    config = Config.load()

    assert config.model == "env-model"
    assert config.temperature == 0.2
    assert config.provider == "openai"


def test_config_uses_provider_env_key_as_api_key_fallback(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    config = Config.load()

    assert config.api_key == "sk-openai"


def test_config_does_not_override_explicit_api_key_with_env_fallback(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    project_dir = project / ".agent"
    project_dir.mkdir()
    (project_dir / "config.toml").write_text(
        textwrap.dedent(
            """
            api_key = "sk-config"
            """
        )
    )

    config = Config.load()

    assert config.api_key == "sk-config"
