"""Native OpenAI API provider with event-based streaming.

This module keeps the public OpenAIProvider API stable while routing
requests to either Chat Completions or Responses API implementations
based on the model name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from .chat import OpenAIChatProvider
from .common import OpenAIError, uses_responses_api
from .responses import OpenAIResponsesProvider

if TYPE_CHECKING:
    import httpx
    import tiktoken

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
    _chat: OpenAIChatProvider | None = field(default=None, repr=False)
    _responses: OpenAIResponsesProvider | None = field(default=None, repr=False)
    _shared_encoder: tiktoken.Encoding | None = field(default=None, repr=False)
    _shared_client: httpx.AsyncClient | None = field(default=None, repr=False)

    def _sync_provider(self, provider: OpenAIChatProvider | OpenAIResponsesProvider) -> None:
        provider.api_key = self.api_key
        provider.model = self.model
        provider.temperature = self.temperature
        provider.max_tokens = self.max_tokens
        if self._shared_client is not None:
            cast("Any", provider)._client = self._shared_client
        if self._shared_encoder is not None:
            cast("Any", provider)._encoder = self._shared_encoder

    def _provider(self) -> OpenAIChatProvider | OpenAIResponsesProvider:
        if uses_responses_api(self.model):
            if self._responses is None:
                self._responses = OpenAIResponsesProvider(
                    api_key=self.api_key,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            self._sync_provider(self._responses)
            return self._responses

        if self._chat is None:
            self._chat = OpenAIChatProvider(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        self._sync_provider(self._chat)
        return self._chat

    @property
    def _encoder(self) -> tiktoken.Encoding | None:  # noqa: SLF001
        if self._shared_encoder is None:
            for prov in (self._chat, self._responses):
                if prov is not None and getattr(prov, "_encoder", None) is not None:
                    self._shared_encoder = cast("Any", prov)._encoder
                    break
        return self._shared_encoder

    @_encoder.setter
    def _encoder(self, value: tiktoken.Encoding | None) -> None:  # noqa: SLF001
        self._shared_encoder = value
        for prov in (self._chat, self._responses):
            if prov is not None:
                cast("Any", prov)._encoder = value

    @property
    def client(self) -> httpx.AsyncClient:
        """Expose the active client's HTTP session for compatibility."""
        provider = self._provider()
        client = provider.client
        self._shared_client = client
        return client

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return self._provider().stream(messages, tools, options)

    def count_tokens(self, text: str) -> int:
        provider = self._provider()
        count = provider.count_tokens(text)
        self._shared_encoder = cast("Any", provider)._encoder
        return count

    def count_messages_tokens(self, messages: list[Message]) -> int:
        provider = self._provider()
        count = provider.count_messages_tokens(messages)
        self._shared_encoder = cast("Any", provider)._encoder
        return count

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
