"""Web runtime composition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.extensions.host import ExtensionHost
from agent.runtime.agent import Agent

if TYPE_CHECKING:
    from agent.config import Config
    from agent.llm.provider import LLMProvider
    from agent.runtime.session import Session
    from agent.web.bridge import WebExtensionBridge


@dataclass(slots=True)
class WebRuntime:
    """Runtime stack for one web delivery connection."""

    agent: Agent
    extension_host: ExtensionHost


def build_web_runtime(
    config: Config,
    provider: LLMProvider,
    session: Session,
    *,
    extension_bridge: WebExtensionBridge,
) -> WebRuntime:
    """Build the runtime stack used by the web delivery shell."""
    agent = Agent(
        config.to_agent_settings(),
        provider,
        session,
    )
    extension_host = ExtensionHost(agent, paths=config.extensions)
    extension_host.bind_ui(extension_bridge.bindings())
    agent.set_hooks(extension_host)
    return WebRuntime(agent=agent, extension_host=extension_host)
