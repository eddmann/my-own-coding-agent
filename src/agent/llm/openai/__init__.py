"""Native OpenAI API provider with event-based streaming.

This module keeps the public OpenAIProvider API stable while routing
requests to either Chat Completions or Responses API implementations
based on the model name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .chat import OpenAIChatProvider
from .common import OpenAIError, uses_responses_api
from .responses import OpenAIResponsesProvider

if TYPE_CHECKING:
    import httpx

    from agent.core.message import Message
    from agent.llm.events import StreamOptions
    from agent.llm.stream import AssistantMessageEventStream


@dataclass(slots=True)
class OpenAIProvider:
    """OpenAI provider that routes between chat and responses implementations."""

    api_key: str
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    http_client: httpx.AsyncClient | None = field(default=None, repr=False)
    _chat: OpenAIChatProvider | None = field(default=None, repr=False)
    _responses: OpenAIResponsesProvider | None = field(default=None, repr=False)

    def set_model(self, model: str) -> None:
        """Update model and clear model-scoped caches in active adapters."""
        if not model or model == self.model:
            return
        self.model = model
        if self._chat is not None:
            self._chat.set_model(model)
        if self._responses is not None:
            self._responses.set_model(model)

    def _sync_provider(self, provider: OpenAIChatProvider | OpenAIResponsesProvider) -> None:
        provider.api_key = self.api_key
        provider.temperature = self.temperature
        provider.max_tokens = self.max_tokens
        provider.set_model(self.model)

    def _provider(self) -> OpenAIChatProvider | OpenAIResponsesProvider:
        if uses_responses_api(self.model):
            if self._responses is None:
                self._responses = OpenAIResponsesProvider(
                    api_key=self.api_key,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    http_client=self.http_client,
                )
            self._sync_provider(self._responses)
            return self._responses

        if self._chat is None:
            self._chat = OpenAIChatProvider(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                http_client=self.http_client,
            )
        self._sync_provider(self._chat)
        return self._chat

    @property
    def client(self) -> httpx.AsyncClient:
        """Expose the active client's HTTP session for compatibility."""
        return self._provider().client

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return self._provider().stream(messages, tools, options)

    def count_tokens(self, text: str) -> int:
        return self._provider().count_tokens(text)

    def count_messages_tokens(self, messages: list[Message]) -> int:
        return self._provider().count_messages_tokens(messages)

    def supports_thinking(self) -> bool:
        return self._provider().supports_thinking()

    async def list_models(self) -> list[str]:
        return await self._provider().list_models()

    async def close(self) -> None:
        if self._chat is not None:
            await self._chat.close()
        if self._responses is not None:
            await self._responses.close()


__all__ = ["OpenAIProvider", "OpenAIError"]
