"""Behavior tests for config loading and overrides."""

from __future__ import annotations

import textwrap

from agent.core.config import Config


def test_config_loads_with_precedence(temp_dir, monkeypatch):
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
