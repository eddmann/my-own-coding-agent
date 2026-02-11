"""Delivery-layer configuration loading (files + environment + CLI overrides)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent.core.settings import AgentSettings, ThinkingLevel


@dataclass(slots=True)
class ProviderConfig:
    """Top-level provider override values from user config."""

    base_url: str
    model: str | None = None
    api_key: str | None = None


@dataclass(slots=True)
class Config:
    """Resolved app config (delivery concern)."""

    provider: str = "openai"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    context_max_tokens: int = 128000
    max_output_tokens: int = 8192
    temperature: float = 0.7
    thinking_level: ThinkingLevel = ThinkingLevel.OFF
    session_dir: Path = field(default_factory=lambda: Path.home() / ".agent" / "sessions")
    skills_dirs: list[Path] = field(default_factory=list)
    extensions: list[Path] = field(default_factory=list)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    prompt_template_dirs: list[Path] = field(default_factory=list)
    context_file_paths: list[Path] = field(default_factory=list)
    custom_system_prompt: str | None = None
    append_system_prompt: str | None = None

    @classmethod
    def load(cls) -> Config:
        """Load config from files and environment variables."""
        config_data: dict[str, Any] = {}

        global_config_dir = Path.home() / ".agent"
        config_data = cls._merge_config(config_data, cls._load_config_file(global_config_dir))

        project_config_dir = Path.cwd() / ".agent"
        config_data = cls._merge_config(config_data, cls._load_config_file(project_config_dir))

        config_data = cls._apply_env_vars(config_data)
        return cls._from_dict(config_data)

    def to_agent_settings(self) -> AgentSettings:
        """Project delivery config into core runtime settings."""
        return AgentSettings(
            context_max_tokens=self.context_max_tokens,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            thinking_level=self.thinking_level,
            session_dir=self.session_dir,
            skills_dirs=list(self.skills_dirs),
            extensions=list(self.extensions),
            prompt_template_dirs=list(self.prompt_template_dirs),
            context_file_paths=list(self.context_file_paths),
            custom_system_prompt=self.custom_system_prompt,
            append_system_prompt=self.append_system_prompt,
        )

    def provider_overrides(self) -> dict[str, ProviderConfig]:
        """Return provider override values for provider bootstrap."""
        return dict(self.providers)

    @classmethod
    def _load_config_file(cls, config_dir: Path) -> dict[str, Any]:
        """Load config from a directory (TOML or YAML)."""
        toml_path = config_dir / "config.toml"
        yaml_path = config_dir / "config.yaml"
        yml_path = config_dir / "config.yml"

        if toml_path.exists():
            with open(toml_path, "rb") as f:
                return tomllib.load(f)
        if yaml_path.exists():
            with open(yaml_path) as f:
                return yaml.safe_load(f) or {}
        if yml_path.exists():
            with open(yml_path) as f:
                return yaml.safe_load(f) or {}

        return {}

    @classmethod
    def _merge_config(cls, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two config dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    @classmethod
    def _apply_env_vars(cls, config_data: dict[str, Any]) -> dict[str, Any]:
        """Apply AGENT_* environment variables."""
        env_mappings = {
            "AGENT_PROVIDER": "provider",
            "AGENT_MODEL": "model",
            "AGENT_API_KEY": "api_key",
            "AGENT_BASE_URL": "base_url",
            # Legacy: maps to context_max_tokens (output tokens have their own field)
            "AGENT_MAX_TOKENS": "max_tokens",
            "AGENT_CONTEXT_TOKENS": "context_max_tokens",
            "AGENT_MAX_OUTPUT_TOKENS": "max_output_tokens",
            "AGENT_TEMPERATURE": "temperature",
            "AGENT_THINKING": "thinking_level",
        }

        for env_var, config_key in env_mappings.items():
            if value := os.environ.get(env_var):
                if config_key in ("max_tokens", "context_max_tokens", "max_output_tokens"):
                    config_data[config_key] = int(value)
                elif config_key in ("temperature",):
                    config_data[config_key] = float(value)
                else:
                    config_data[config_key] = value

        # Fallback API key sources
        if not config_data.get("api_key"):
            for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
                if value := os.environ.get(env_var):
                    config_data["api_key"] = value
                    break

        return config_data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Config:
        """Create Config from dictionary."""
        providers: dict[str, ProviderConfig] = {}
        if "providers" in data:
            for name, prov_data in data["providers"].items():
                providers[name] = ProviderConfig(
                    base_url=prov_data.get("base_url", "https://api.openai.com"),
                    model=prov_data.get("model"),
                    api_key=prov_data.get("api_key"),
                )

        skills_dirs = [Path(p) for p in data.get("skills_dirs", [])]
        extensions = [Path(p) for p in data.get("extensions", [])]
        prompt_template_dirs = [Path(p) for p in data.get("prompt_template_dirs", [])]
        context_file_paths = [Path(p) for p in data.get("context_file_paths", [])]
        session_dir = Path(data.get("session_dir", Path.home() / ".agent" / "sessions"))

        thinking_str = data.get("thinking_level", "off")
        try:
            thinking_level = ThinkingLevel(thinking_str)
        except ValueError:
            thinking_level = ThinkingLevel.OFF

        legacy_max_tokens = data.get("max_tokens")
        context_max_tokens = data.get("context_max_tokens")
        if context_max_tokens is None:
            context_max_tokens = legacy_max_tokens if legacy_max_tokens is not None else 128000

        max_output_tokens = data.get("max_output_tokens")
        if max_output_tokens is None:
            max_output_tokens = min(8192, context_max_tokens)

        return cls(
            provider=data.get("provider", "openai"),
            model=data.get("model"),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            context_max_tokens=context_max_tokens,
            max_output_tokens=max_output_tokens,
            temperature=data.get("temperature", 0.7),
            thinking_level=thinking_level,
            session_dir=session_dir,
            skills_dirs=skills_dirs,
            extensions=extensions,
            providers=providers,
            prompt_template_dirs=prompt_template_dirs,
            context_file_paths=context_file_paths,
            custom_system_prompt=data.get("custom_system_prompt"),
            append_system_prompt=data.get("append_system_prompt"),
        )
