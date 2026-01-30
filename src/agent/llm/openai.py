"""Native OpenAI API provider with event-based streaming.

Supports:
- Modern OpenAI models (gpt-4o, gpt-4.5, gpt-5, o1, o3, etc.)
- Reasoning models with reasoning_effort
- Tool use with tool_choice modes
- Event-based streaming with a consistent event schema
- Request cancellation and retry with exponential backoff
- Cost calculation
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import tiktoken

from agent.llm.events import (
    Cost,
    DoneEvent,
    ErrorEvent,
    PartialMessage,
    StartEvent,
    StopReason,
    StreamOptions,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingBlock,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallBlock,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)
from agent.llm.pricing import get_pricing
from agent.llm.retry import RetryConfig, with_retry
from agent.llm.stream import AssistantMessageEventStream

if TYPE_CHECKING:
    from agent.core.message import Message

OPENAI_API_URL = "https://api.openai.com"


class OpenAIError(Exception):
    """Error from the OpenAI API with parsed message."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _parse_api_error(response: httpx.Response) -> str:
    """Extract error message from API response."""
    try:
        data = response.json()
        if "error" in data:
            error = data["error"]
            if isinstance(error, dict):
                return str(error.get("message", error))
            return str(error)
        return response.text or f"HTTP {response.status_code}"
    except Exception:
        return response.text or f"HTTP {response.status_code}"


def _map_stop_reason(openai_reason: str | None) -> StopReason:
    """Map OpenAI finish reason to our StopReason type."""
    mapping: dict[str | None, StopReason] = {
        "stop": "stop",
        "tool_calls": "tool_use",
        "length": "length",
        "content_filter": "stop",
        None: "stop",
    }
    return mapping.get(openai_reason, "stop")


@dataclass(slots=True)
class ModelCapabilities:
    """Model-specific capabilities and parameter support."""

    supports_temperature: bool = True
    token_param: str = "max_tokens"  # or "max_completion_tokens"
    is_reasoning: bool = False
    fixed_reasoning_effort: str | None = None
    max_output_tokens: int = 16384  # Model's max completion tokens


def _get_model_capabilities(model: str) -> ModelCapabilities:
    """Get model-specific capabilities based on model name."""
    model_lower = model.lower()

    # o1/o3 reasoning models - no temperature, use max_completion_tokens
    if model_lower.startswith(("o1", "o3")):
        return ModelCapabilities(
            supports_temperature=False,
            token_param="max_completion_tokens",
            is_reasoning=True,
        )

    # gpt-5-pro - only high reasoning, no temperature
    if model_lower.startswith("gpt-5-pro"):
        return ModelCapabilities(
            supports_temperature=False,
            token_param="max_completion_tokens",
            is_reasoning=True,
            fixed_reasoning_effort="high",
        )

    # gpt-5, gpt-5-mini, gpt-5-nano (NOT gpt-5.x) - no temperature
    if model_lower in ("gpt-5", "gpt-5-mini", "gpt-5-nano") or (
        model_lower.startswith(("gpt-5-mini-", "gpt-5-nano-"))
        and not model_lower.startswith("gpt-5.")
    ):
        return ModelCapabilities(
            supports_temperature=False,
            token_param="max_completion_tokens",
            is_reasoning=True,
        )

    # gpt-5.1, gpt-5.2+ - supports temperature and reasoning
    if model_lower.startswith("gpt-5."):
        return ModelCapabilities(
            supports_temperature=True,
            token_param="max_completion_tokens",
            is_reasoning=True,
        )

    # gpt-4.5, chatgpt-4o - newer format
    if model_lower.startswith(("gpt-4.5", "chatgpt-4o")):
        return ModelCapabilities(
            supports_temperature=True,
            token_param="max_completion_tokens",
            is_reasoning=False,
        )

    # Default: gpt-4o, gpt-4, gpt-3.5, etc.
    return ModelCapabilities(
        supports_temperature=True,
        token_param="max_tokens",
        is_reasoning=False,
    )


@dataclass(slots=True)
class OpenAIProvider:
    """Native OpenAI API provider with full feature support."""

    api_key: str
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    _encoder: tiktoken.Encoding | None = field(default=None, repr=False)

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=OPENAI_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    @property
    def encoder(self) -> tiktoken.Encoding:
        """Get or create the token encoder."""
        if self._encoder is None:
            try:
                self._encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fall back to cl100k_base for unknown models
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    def _get_capabilities(self) -> ModelCapabilities:
        """Get capabilities for the current model."""
        return _get_model_capabilities(self.model)

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        options: StreamOptions | None,
    ) -> dict[str, Any]:
        """Build the API request payload."""
        caps = self._get_capabilities()

        max_tokens = options.max_tokens if options and options.max_tokens else self.max_tokens
        # Clamp to model's max output tokens
        max_tokens = min(max_tokens, caps.max_output_tokens)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api_dict() for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Set token limit with appropriate parameter name
        payload[caps.token_param] = max_tokens

        # Set reasoning effort from model defaults or thinking_level
        reasoning_enabled = False
        if caps.fixed_reasoning_effort:
            payload["reasoning_effort"] = caps.fixed_reasoning_effort
            reasoning_enabled = True
        elif caps.is_reasoning and options and options.thinking_level:
            # Map thinking levels to OpenAI reasoning_effort
            effort_map = {
                "minimal": "low",
                "low": "low",
                "medium": "medium",
                "high": "high",
            }
            if options.thinking_level in effort_map:
                payload["reasoning_effort"] = effort_map[options.thinking_level]
                reasoning_enabled = True

        # Only include temperature if model supports it AND reasoning is not enabled
        # (reasoning models don't support temperature when reasoning_effort is set)
        if caps.supports_temperature and not reasoning_enabled:
            temp = (
                options.temperature
                if options and options.temperature is not None
                else self.temperature
            )
            payload["temperature"] = temp

        if tools:
            payload["tools"] = tools
            # Tool choice handling (OpenAI format)
            if options and options.tool_choice:
                tc = options.tool_choice
                if tc == "any":
                    # OpenAI uses "required" for "must use a tool"
                    payload["tool_choice"] = "required"
                elif tc == "none":
                    # Remove tools entirely when none
                    payload.pop("tools", None)
                elif isinstance(tc, dict) and "name" in tc:
                    # Specific tool
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tc["name"]},
                    }
                else:
                    # "auto", "required", etc. pass through
                    payload["tool_choice"] = tc
            else:
                payload["tool_choice"] = "auto"

        return payload

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """Stream completion as events.

        Returns an AssistantMessageEventStream that yields events
        and provides final PartialMessage via .result().
        """
        stream = AssistantMessageEventStream()
        asyncio.create_task(self._stream_impl(messages, tools, options, stream))
        return stream

    async def _stream_impl(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        options: StreamOptions | None,
        stream: AssistantMessageEventStream,
    ) -> None:
        """Internal implementation of streaming with retry and cancellation."""
        output = PartialMessage()
        retry_config = RetryConfig()

        # Track state
        text_started = False
        text_content = ""
        thinking_started = False
        thinking_content = ""
        pending_tool_calls: dict[int, ToolCallBlock] = {}
        finish_reason: str | None = None

        def _check_cancelled() -> bool:
            """Check if request was cancelled."""
            return bool(options and options.cancel_event and options.cancel_event.is_set())

        async def _do_stream() -> None:
            """Execute the streaming request (for retry)."""
            nonlocal text_started, text_content, thinking_started, thinking_content
            nonlocal pending_tool_calls, finish_reason

            # Check cancellation before starting
            if _check_cancelled():
                raise asyncio.CancelledError("Request cancelled")

            payload = self._build_payload(messages, tools, options)

            # Emit start event
            stream.push(StartEvent(partial=output))

            async with self.client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    error_msg = _parse_api_error(response)
                    raise OpenAIError(error_msg, response.status_code)

                async for line in response.aiter_lines():
                    # Check cancellation during streaming
                    if _check_cancelled():
                        stream.abort("Request cancelled")
                        return

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Handle usage in final chunk
                    if chunk.get("usage"):
                        usage_data = chunk["usage"]
                        output.usage = Usage(
                            input=usage_data.get("prompt_tokens", 0),
                            output=usage_data.get("completion_tokens", 0),
                        )
                        output.usage.update_total()

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})

                    # Check finish reason
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    # Handle reasoning/thinking content (multiple possible field names)
                    reasoning = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or delta.get("reasoning_text")
                    )
                    if reasoning:
                        if not thinking_started:
                            thinking_started = True
                            output.content.append(ThinkingBlock(thinking=""))
                            stream.push(
                                ThinkingStartEvent(
                                    content_index=len(output.content) - 1,
                                    partial=output,
                                )
                            )
                        thinking_content += reasoning
                        if output.content and isinstance(output.content[-1], ThinkingBlock):
                            output.content[-1].thinking = thinking_content
                        stream.push(
                            ThinkingDeltaEvent(
                                content_index=len(output.content) - 1,
                                delta=reasoning,
                                partial=output,
                            )
                        )

                    # Handle content tokens
                    if content := delta.get("content"):
                        # End thinking if we were in it
                        if thinking_started and not text_started:
                            stream.push(
                                ThinkingEndEvent(
                                    content_index=len(output.content) - 1,
                                    thinking=thinking_content,
                                    partial=output,
                                )
                            )

                        if not text_started:
                            text_started = True
                            output.content.append(TextBlock(text=""))
                            stream.push(
                                TextStartEvent(
                                    content_index=len(output.content) - 1,
                                    partial=output,
                                )
                            )
                        text_content += content
                        if output.content and isinstance(output.content[-1], TextBlock):
                            output.content[-1].text = text_content
                        stream.push(
                            TextDeltaEvent(
                                content_index=len(output.content) - 1,
                                delta=content,
                                partial=output,
                            )
                        )

                    # Handle tool calls
                    if tool_calls := delta.get("tool_calls"):
                        for tc in tool_calls:
                            idx = tc.get("index", 0)

                            if idx not in pending_tool_calls:
                                # New tool call
                                tool_block = ToolCallBlock(
                                    id=tc.get("id", ""),
                                    name=tc.get("function", {}).get("name", ""),
                                )
                                pending_tool_calls[idx] = tool_block
                                output.content.append(tool_block)
                                stream.push(
                                    ToolCallStartEvent(
                                        content_index=len(output.content) - 1,
                                        tool_id=tool_block.id,
                                        tool_name=tool_block.name,
                                        partial=output,
                                    )
                                )

                            tool_block = pending_tool_calls[idx]

                            # Update ID if provided
                            if tc.get("id"):
                                tool_block.id = tc["id"]

                            # Update name if provided
                            if func := tc.get("function"):
                                if func.get("name"):
                                    tool_block.name = func["name"]
                                if func.get("arguments"):
                                    tool_block._partial_json += func["arguments"]
                                    stream.push(
                                        ToolCallDeltaEvent(
                                            content_index=len(output.content) - 1,
                                            delta=func["arguments"],
                                            partial=output,
                                        )
                                    )

        try:
            await with_retry(_do_stream, retry_config)

            # Skip done event if already aborted
            if stream.is_aborted:
                return

            # End any open blocks
            if thinking_started and not text_started:
                stream.push(
                    ThinkingEndEvent(
                        content_index=0,
                        thinking=thinking_content,
                        partial=output,
                    )
                )

            if text_started:
                stream.push(
                    TextEndEvent(
                        content_index=len(output.content) - 1,
                        text=text_content,
                        partial=output,
                    )
                )

            # Emit completed tool calls
            for idx in sorted(pending_tool_calls.keys()):
                tool_block = pending_tool_calls[idx]
                try:
                    args = json.loads(tool_block._partial_json) if tool_block._partial_json else {}
                except json.JSONDecodeError:
                    args = {}
                tool_block.arguments = args
                stream.push(
                    ToolCallEndEvent(
                        content_index=len(output.content) - 1,
                        tool_call=tool_block,
                        partial=output,
                    )
                )

            # Set stop reason
            output.stop_reason = _map_stop_reason(finish_reason)

            # Calculate cost
            pricing = get_pricing(self.model, "openai")
            if pricing:
                output.usage.cost = Cost(
                    input=(output.usage.input / 1_000_000) * pricing.input,
                    output=(output.usage.output / 1_000_000) * pricing.output,
                    cache_read=(output.usage.cache_read / 1_000_000) * pricing.cache_read,
                    cache_write=(output.usage.cache_write / 1_000_000) * pricing.cache_write,
                )
                output.usage.cost.total = (
                    output.usage.cost.input
                    + output.usage.cost.output
                    + output.usage.cost.cache_read
                    + output.usage.cost.cache_write
                )

            # Emit done event
            stream.push(DoneEvent(stop_reason=output.stop_reason, message=output))
            stream.end()

        except asyncio.CancelledError:
            output.error_message = "Request cancelled"
            output.stop_reason = "aborted"
            stream.push(ErrorEvent(stop_reason="aborted", message=output))
            stream.end(output)

        except Exception as e:
            output.error_message = str(e)
            output.stop_reason = "error"
            stream.push(ErrorEvent(stop_reason="error", message=output))
            stream.end(output)

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.encoder.encode(text))

    def count_messages_tokens(self, messages: list[Message]) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for msg in messages:
            total += 4  # Overhead per message
            total += self.count_tokens(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total += self.count_tokens(tc.name)
                    total += self.count_tokens(json.dumps(tc.arguments))
        return total

    def supports_thinking(self) -> bool:
        """Check if the current model supports thinking/reasoning.

        Uses model registry for known models, falls back to
        capability detection for unknown models.
        """
        from agent.llm.models import get_model_info, supports_reasoning

        # Check registry first
        info = get_model_info(self.model)
        if info is not None:
            return info.reasoning

        # Fallback to name-based detection
        return supports_reasoning(self.model, provider="openai")

    async def list_models(self) -> list[str]:
        """Fetch available models from the API."""
        try:
            response = await self.client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            return sorted(models)
        except Exception:
            # Return empty list if API doesn't support /v1/models
            return []

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
