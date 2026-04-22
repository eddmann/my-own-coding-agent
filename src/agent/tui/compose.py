"""TUI runtime composition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.extensions.host import ExtensionHost
from agent.prompts.loader import PromptTemplateLoader
from agent.runtime.agent import Agent
from agent.skills.loader import SkillLoader

if TYPE_CHECKING:
    from agent.config import Config
    from agent.llm.provider import LLMProvider
    from agent.runtime.session import Session
    from agent.tui.extension_bridge import TUIExtensionBridge


@dataclass(slots=True)
class TUILoaders:
    """Shared TUI loaders used by the runtime and prompt input."""

    skill_loader: SkillLoader
    template_loader: PromptTemplateLoader


@dataclass(slots=True)
class TUIRuntime:
    """Interactive runtime stack for the TUI."""

    agent: Agent
    extension_host: ExtensionHost


def build_tui_loaders(config: Config) -> TUILoaders:
    """Build loaders shared between autocomplete and the agent."""
    return TUILoaders(
        skill_loader=SkillLoader.with_defaults(extra_dirs=config.skills_dirs),
        template_loader=PromptTemplateLoader.with_defaults(extra_dirs=config.prompt_template_dirs),
    )


def build_tui_runtime(
    config: Config,
    provider: LLMProvider,
    session: Session | None,
    *,
    loaders: TUILoaders,
    extension_bridge: TUIExtensionBridge,
) -> TUIRuntime:
    """Build the runtime stack used by the interactive TUI."""
    agent = Agent(
        config.to_agent_settings(),
        provider,
        session,
        skill_loader=loaders.skill_loader,
        template_loader=loaders.template_loader,
    )
    extension_host = ExtensionHost(agent, paths=config.extensions)
    extension_host.bind_ui(extension_bridge.bindings())
    agent.set_hooks(extension_host)
    return TUIRuntime(agent=agent, extension_host=extension_host)
