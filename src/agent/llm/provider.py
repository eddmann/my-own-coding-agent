"""Abstract LLM provider protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent.core.message import Message
    from agent.llm.events import StreamOptions
    from agent.llm.stream import AssistantMessageEventStream


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers.

    All providers must implement stream() for event-based streaming,
    count_tokens() for context management, and supports_thinking()
    for capability detection.
    """

    model: str

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """Stream completion as events.

        Args:
            messages: The conversation history
            tools: Optional list of tool definitions in OpenAI format
            options: Optional streaming options (temperature, max_tokens, etc.)

        Returns:
            AssistantMessageEventStream that yields StreamEvent instances
            and provides final PartialMessage via .result()
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Estimate token count for context management.

        Args:
            text: The text to count tokens for

        Returns:
            Estimated token count
        """
        ...

    def count_messages_tokens(self, messages: list[Message]) -> int:
        """Estimate token count for a list of messages.

        Args:
            messages: The messages to count tokens for

        Returns:
            Estimated token count
        """
        ...

    def supports_thinking(self) -> bool:
        """Check if the current model supports thinking/reasoning.

        Returns:
            True if model supports thinking, False otherwise
        """
        ...

    async def list_models(self) -> list[str]:
        """List available models from the provider.

        Returns:
            List of model IDs, or empty list if not supported
        """
        ...

    async def close(self) -> None:
        """Close any open connections."""
        ...
