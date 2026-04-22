"""Internal runtime state for the extension host."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable
    from pathlib import Path

    from agent.extensions.api import ExtensionUIBindings
    from agent.runtime.agent import Agent
    from agent.runtime.hooks import RunControl
    from agent.tools.base import BaseTool


@dataclass(slots=True)
class ExtensionRunState:
    """Per-run extension control state."""

    queued_user_messages: list[str] = field(default_factory=list)
    abort_requested: bool = False

    def request_abort(self) -> None:
        self.abort_requested = True

    def queue_user_message(self, text: str) -> None:
        if text.strip():
            self.queued_user_messages.append(text)

    def drain_user_messages(self) -> list[str]:
        queued = list(self.queued_user_messages)
        self.queued_user_messages.clear()
        return queued


class _ExtensionRunControl:
    """Run-control adapter exposed to the core runtime."""

    __slots__ = ("_state",)

    def __init__(self, state: ExtensionRunState) -> None:
        self._state = state

    def reset(self) -> None:
        self._state.abort_requested = False

    @property
    def aborted(self) -> bool:
        return self._state.abort_requested

    def drain_user_messages(self) -> list[str]:
        return self._state.drain_user_messages()


class ExtensionRuntime:
    """Mutable runtime state owned by the extension host."""

    __slots__ = ("agent", "_paths", "_ui_bindings", "_active_run_state")

    def __init__(self, agent: Agent, paths: list[Path] | None = None) -> None:
        self.agent = agent
        self._paths = list(paths or [])
        self._ui_bindings: ExtensionUIBindings | None = None
        self._active_run_state: ExtensionRunState | None = None

    @property
    def ui_bindings(self) -> ExtensionUIBindings | None:
        """Optional UI bindings exposed to extension contexts."""
        return self._ui_bindings

    @property
    def current_run_state(self) -> ExtensionRunState | None:
        """Return the currently active extension run state, if any."""
        return self._active_run_state

    def bind_ui(self, bindings: ExtensionUIBindings | None) -> None:
        """Bind optional UI callbacks for extensions."""
        self._ui_bindings = bindings

    def config_snapshot(self) -> dict[str, object]:
        """Build the config snapshot exposed to extension contexts."""
        return {
            "provider": self.agent.provider_name,
            "model": self.agent.model_name,
            "api_key": getattr(self.agent.provider, "api_key", None),
            "base_url": getattr(self.agent.provider, "base_url", None),
            "context_max_tokens": self.agent.context_max_tokens,
            "max_output_tokens": self.agent.max_output_tokens,
            "temperature": self.agent.temperature,
            "thinking_level": self.agent.thinking_level.value,
            "session_dir": self.agent.session_dir,
            "skills_dirs": list(self.agent.config.skills_dirs),
            "extensions": list(self._paths),
            "provider_overrides": {
                name: {
                    "base_url": override.base_url,
                    "model": override.model,
                    "api_key": override.api_key,
                }
                for name, override in getattr(self.agent.config, "providers", {}).items()
            },
            "prompt_template_dirs": list(self.agent.config.prompt_template_dirs),
            "context_file_paths": list(self.agent.config.context_file_paths),
            "custom_system_prompt": self.agent.config.custom_system_prompt,
            "append_system_prompt": self.agent.config.append_system_prompt,
        }

    def extension_paths(self, paths: list[Path] | None = None) -> list[Path]:
        """Resolve the extension paths to load."""
        return list(paths) if paths is not None else list(self._paths)

    def register_tools(self, tools: Iterable[BaseTool[Any]]) -> None:
        """Register extension-provided tools with the runtime."""
        for tool in tools:
            self.agent.register_tool(tool)

    @asynccontextmanager
    async def run_scope(self) -> AsyncIterator[RunControl]:
        """Expose run-scoped control state to the core runtime."""
        previous_state = self._active_run_state
        state = ExtensionRunState()
        self._active_run_state = state
        try:
            yield _ExtensionRunControl(state)
        finally:
            self._active_run_state = previous_state
