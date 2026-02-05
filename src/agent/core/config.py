"""Configuration loading from files and environment variables."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


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


def get_available_thinking_levels(model_id: str) -> list[ThinkingLevel]:
    """Get available thinking levels for a model.

    Args:
        model_id: The model ID to check

    Returns:
        List of supported thinking levels
    """
    from agent.llm.models import supports_reasoning, supports_xhigh

    if not supports_reasoning(model_id):
        return [ThinkingLevel.OFF]

    if supports_xhigh(model_id):
        return THINKING_LEVELS_WITH_XHIGH

    return THINKING_LEVELS


def clamp_thinking_level(level: ThinkingLevel, available: list[ThinkingLevel]) -> ThinkingLevel:
    """Clamp a thinking level to the available levels.

    If the requested level isn't available, returns the highest
    available level that's <= the requested level.

    Args:
        level: Requested thinking level
        available: List of available levels

    Returns:
        Valid thinking level from available list
    """
    if level in available:
        return level

    # Get ordered levels
    all_levels = list(ThinkingLevel)
    level_idx = all_levels.index(level)

    # Find highest available level <= requested
    for check_level in reversed(all_levels[: level_idx + 1]):
        if check_level in available:
            return check_level

    # Fallback to first available (should be OFF)
    return available[0] if available else ThinkingLevel.OFF


@dataclass(slots=True)
class ProviderConfig:
    """Configuration for a specific LLM provider."""

    base_url: str
    model: str
    api_key: str | None = None


@dataclass(slots=True)
class Config:
    """Agent configuration with sensible defaults."""

    provider: str = "openai"
    model: str = "gpt-4o"
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
    # Prompt configuration
    prompt_template_dirs: list[Path] = field(default_factory=list)
    context_file_paths: list[Path] = field(default_factory=list)
    custom_system_prompt: str | None = None
    append_system_prompt: str | None = None

    @classmethod
    def load(cls) -> Config:
        """Load config from files and environment variables.

        Priority (highest to lowest):
        1. Environment variables (AGENT_*)
        2. Project config (.agent/config.toml or .agent/config.yaml)
        3. Global config (~/.agent/config.toml or ~/.agent/config.yaml)
        4. Defaults
        """
        config_data: dict[str, Any] = {}

        # Load global config
        global_config_dir = Path.home() / ".agent"
        config_data = cls._merge_config(config_data, cls._load_config_file(global_config_dir))

        # Load project config
        project_config_dir = Path.cwd() / ".agent"
        config_data = cls._merge_config(config_data, cls._load_config_file(project_config_dir))

        # Apply environment variables
        config_data = cls._apply_env_vars(config_data)

        # Build config object
        return cls._from_dict(config_data)

    @classmethod
    def _load_config_file(cls, config_dir: Path) -> dict[str, Any]:
        """Load config from a directory (TOML or YAML)."""
        toml_path = config_dir / "config.toml"
        yaml_path = config_dir / "config.yaml"
        yml_path = config_dir / "config.yml"

        if toml_path.exists():
            with open(toml_path, "rb") as f:
                return tomllib.load(f)
        elif yaml_path.exists():
            with open(yaml_path) as f:
                return yaml.safe_load(f) or {}
        elif yml_path.exists():
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
        # Parse providers
        providers: dict[str, ProviderConfig] = {}
        if "providers" in data:
            for name, prov_data in data["providers"].items():
                providers[name] = ProviderConfig(
                    base_url=prov_data.get("base_url", "https://api.openai.com"),
                    model=prov_data.get("model", "gpt-4o"),
                    api_key=prov_data.get("api_key"),
                )

        # Parse skills directories
        skills_dirs: list[Path] = []
        if "skills_dirs" in data:
            skills_dirs = [Path(p) for p in data["skills_dirs"]]

        # Parse extensions
        extensions: list[Path] = []
        if "extensions" in data:
            extensions = [Path(p) for p in data["extensions"]]

        # Parse session directory
        session_dir = Path(data.get("session_dir", Path.home() / ".agent" / "sessions"))

        # Parse thinking level
        thinking_str = data.get("thinking_level", "off")
        try:
            thinking_level = ThinkingLevel(thinking_str)
        except ValueError:
            thinking_level = ThinkingLevel.OFF

        # Parse prompt template directories
        prompt_template_dirs: list[Path] = []
        if "prompt_template_dirs" in data:
            prompt_template_dirs = [Path(p) for p in data["prompt_template_dirs"]]

        # Parse context file paths
        context_file_paths: list[Path] = []
        if "context_file_paths" in data:
            context_file_paths = [Path(p) for p in data["context_file_paths"]]

        legacy_max_tokens = data.get("max_tokens")

        context_max_tokens = data.get("context_max_tokens")
        if context_max_tokens is None:
            context_max_tokens = legacy_max_tokens if legacy_max_tokens is not None else 128000

        max_output_tokens = data.get("max_output_tokens")
        if max_output_tokens is None:
            max_output_tokens = min(8192, context_max_tokens)

        return cls(
            provider=data.get("provider", "openai"),
            model=data.get("model", "gpt-4o"),
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

    def get_provider_config(self) -> ProviderConfig:
        """Get the configuration for the current provider."""

        def _env_provider_key(provider: str) -> str | None:
            mapping = {
                "openai": "OPENAI_API_KEY",
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

        def _resolve_api_key(default: str | None, provider: str) -> str | None:
            # Provider-specific environment variable first
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

            # Fallback for other providers
            if agent_key:
                return agent_key
            return default

        if self.provider in self.providers:
            prov = self.providers[self.provider]
            # Override with top-level settings if specified
            return ProviderConfig(
                base_url=self.base_url or prov.base_url,
                model=self.model or prov.model,
                api_key=_resolve_api_key(self.api_key or prov.api_key, self.provider),
            )

        # Default provider configs
        default_configs = {
            "openai": ProviderConfig(
                base_url="https://api.openai.com",
                model=self.model,
                api_key=self.api_key,
            ),
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.com",
                model=self.model if self.model != "gpt-4o" else "claude-sonnet-4-20250514",
                api_key=self.api_key,
            ),
            "ollama": ProviderConfig(
                base_url="http://localhost:11434/v1",
                model=self.model,
                api_key="ollama",
            ),
            "openrouter": ProviderConfig(
                base_url="https://openrouter.ai/api/v1",
                model=self.model,
                api_key=self.api_key,
            ),
            "groq": ProviderConfig(
                base_url="https://api.groq.com/openai/v1",
                model=self.model,
                api_key=self.api_key,
            ),
        }

        if self.provider in default_configs:
            config = default_configs[self.provider]
            # Override with top-level settings
            return ProviderConfig(
                base_url=self.base_url or config.base_url,
                model=self.model,
                api_key=_resolve_api_key(self.api_key or config.api_key, self.provider),
            )

        # Fallback to using top-level settings directly
        return ProviderConfig(
            base_url=self.base_url or "https://api.openai.com",
            model=self.model,
            api_key=_resolve_api_key(self.api_key, self.provider),
        )
