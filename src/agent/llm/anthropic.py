"""Native Anthropic API provider with event-based streaming.

Supports:
- Claude models (claude-3-opus, claude-3-sonnet, claude-sonnet-4, etc.)
- Extended thinking/reasoning with signatures
- Tool use with tool_choice modes
- Event-based streaming with a consistent event schema
- Request cancellation and retry with exponential backoff
- Cost calculation
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent.core.config import THINKING_BUDGETS, ThinkingLevel
from agent.core.message import Message, Role
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

ANTHROPIC_API_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicError(Exception):
    """Error from the Anthropic API with parsed message."""

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


def _map_stop_reason(anthropic_reason: str | None) -> StopReason:
    """Map Anthropic stop reason to our StopReason type."""
    mapping: dict[str | None, StopReason] = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "tool_use": "tool_use",
        "max_tokens": "length",
        None: "stop",
    }
    return mapping.get(anthropic_reason, "stop")


@dataclass(slots=True)
class AnthropicProvider:
    """Native Anthropic API provider with full feature support."""

    api_key: str
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=ANTHROPIC_API_URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Convert messages to Anthropic format, extracting system prompt.

        Anthropic format differences from OpenAI:
        - System prompt is a separate parameter, not in messages
        - Tool results are wrapped in user messages with type: "tool_result"
        - Thinking content uses type: "thinking" with optional signature
        """
        system_prompts: list[str] = []
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                if msg.content:
                    system_prompts.append(msg.content)
            elif msg.role == Role.USER:
                api_messages.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                content = self._build_assistant_content(msg)
                api_messages.append({"role": "assistant", "content": content})
            elif msg.role == Role.TOOL:
                # Tool results are wrapped in user messages
                api_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )

        system_prompt = "\n\n".join(system_prompts).strip()
        return system_prompt, api_messages

    def _build_assistant_content(self, msg: Message) -> list[dict[str, Any]] | str:
        """Build content array for assistant message."""
        content_blocks: list[dict[str, Any]] = []

        # Add thinking block if present
        if msg.thinking:
            block: dict[str, Any] = {"type": "thinking", "thinking": msg.thinking.text}
            if msg.thinking.signature:
                block["signature"] = msg.thinking.signature
            content_blocks.append(block)

        # Add text content
        if msg.content:
            content_blocks.append({"type": "text", "text": msg.content})

        # Add tool use blocks
        if msg.tool_calls:
            for tc in msg.tool_calls:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )

        # Return string if only text, otherwise return array
        if len(content_blocks) == 1 and content_blocks[0]["type"] == "text":
            return msg.content
        return content_blocks

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        return [
            {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "input_schema": tool["function"].get("parameters", {"type": "object"}),
            }
            for tool in tools
        ]

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        options: StreamOptions | None,
    ) -> dict[str, Any]:
        """Build the API request payload."""
        system_prompt, api_messages = self._convert_messages(messages)

        max_tokens = options.max_tokens if options and options.max_tokens else self.max_tokens

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": api_messages,
            "stream": True,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            payload["tools"] = self._convert_tools(tools)

        # Configure extended thinking if enabled
        thinking_level = None
        if options and options.thinking_level and options.thinking_level != "off":
            with contextlib.suppress(ValueError):
                thinking_level = ThinkingLevel(options.thinking_level)

        if thinking_level and thinking_level != ThinkingLevel.OFF:
            budget = THINKING_BUDGETS.get(thinking_level, 8192)
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
            # Ensure max_tokens can accommodate thinking + response
            payload["max_tokens"] = min(
                max_tokens + budget,
                128000,  # Claude's max output tokens
            )

        # Tool choice handling (Anthropic format)
        if options and options.tool_choice and tools:
            tc = options.tool_choice
            if tc == "required":
                # Anthropic uses "any" for "must use a tool"
                payload["tool_choice"] = {"type": "any"}
            elif tc == "any":
                payload["tool_choice"] = {"type": "any"}
            elif tc == "none":
                # Remove tools entirely when none
                payload.pop("tools", None)
            elif isinstance(tc, dict) and "name" in tc:
                # Specific tool
                payload["tool_choice"] = {"type": "tool", "name": tc["name"]}
            elif tc == "auto":
                payload["tool_choice"] = {"type": "auto"}

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

        # Track current content blocks
        current_block_type: str | None = None
        current_block_index: int = 0
        current_text = ""
        current_thinking = ""
        current_thinking_signature: str | None = None
        current_tool: ToolCallBlock | None = None

        def _check_cancelled() -> bool:
            """Check if request was cancelled."""
            return bool(options and options.cancel_event and options.cancel_event.is_set())

        async def _do_stream() -> None:
            """Execute the streaming request (for retry)."""
            nonlocal current_block_type, current_block_index, current_text
            nonlocal current_thinking, current_thinking_signature, current_tool

            # Check cancellation before starting
            if _check_cancelled():
                raise asyncio.CancelledError("Request cancelled")

            payload = self._build_payload(messages, tools, options)

            # Emit start event
            stream.push(StartEvent(partial=output))

            async with self.client.stream(
                "POST",
                "/v1/messages",
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    error_msg = _parse_api_error(response)
                    raise AnthropicError(error_msg, response.status_code)

                async for line in response.aiter_lines():
                    # Check cancellation during streaming
                    if _check_cancelled():
                        stream.abort("Request cancelled")
                        return

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if not data_str or data_str == "[DONE]":
                        continue

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("type")

                    if event_type == "message_start":
                        # Extract initial usage
                        msg = data.get("message", {})
                        usage_data = msg.get("usage", {})
                        output.usage = Usage(
                            input=usage_data.get("input_tokens", 0),
                            output=usage_data.get("output_tokens", 0),
                            cache_read=usage_data.get("cache_read_input_tokens", 0),
                            cache_write=usage_data.get("cache_creation_input_tokens", 0),
                        )
                        output.usage.update_total()

                    elif event_type == "content_block_start":
                        block = data.get("content_block", {})
                        block_type = block.get("type")
                        current_block_index = data.get("index", current_block_index)
                        current_block_type = block_type

                        if block_type == "text":
                            current_text = block.get("text", "")
                            output.content.append(TextBlock(text=current_text))
                            stream.push(
                                TextStartEvent(content_index=current_block_index, partial=output)
                            )

                        elif block_type == "thinking":
                            current_thinking = block.get("thinking", "")
                            current_thinking_signature = None
                            output.content.append(ThinkingBlock(thinking=current_thinking))
                            stream.push(
                                ThinkingStartEvent(
                                    content_index=current_block_index, partial=output
                                )
                            )

                        elif block_type == "tool_use":
                            current_tool = ToolCallBlock(
                                id=block.get("id", ""),
                                name=block.get("name", ""),
                            )
                            output.content.append(current_tool)
                            stream.push(
                                ToolCallStartEvent(
                                    content_index=current_block_index,
                                    tool_id=current_tool.id,
                                    tool_name=current_tool.name,
                                    partial=output,
                                )
                            )

                    elif event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        delta_type = delta.get("type")

                        if delta_type == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                current_text += text
                                # Update the block in output.content
                                if output.content and isinstance(output.content[-1], TextBlock):
                                    output.content[-1].text = current_text
                                stream.push(
                                    TextDeltaEvent(
                                        content_index=current_block_index,
                                        delta=text,
                                        partial=output,
                                    )
                                )

                        elif delta_type == "thinking_delta":
                            thinking_text = delta.get("thinking", "")
                            if thinking_text:
                                current_thinking += thinking_text
                                # Update the block in output.content
                                if output.content and isinstance(output.content[-1], ThinkingBlock):
                                    output.content[-1].thinking = current_thinking
                                stream.push(
                                    ThinkingDeltaEvent(
                                        content_index=current_block_index,
                                        delta=thinking_text,
                                        partial=output,
                                    )
                                )

                        elif delta_type == "signature_delta":
                            signature = delta.get("signature", "")
                            if signature:
                                if current_thinking_signature is None:
                                    current_thinking_signature = ""
                                current_thinking_signature += signature
                                # Update the block in output.content
                                if output.content and isinstance(output.content[-1], ThinkingBlock):
                                    output.content[-1].signature = current_thinking_signature

                        elif delta_type == "input_json_delta":
                            partial_json = delta.get("partial_json", "")
                            if partial_json and current_tool:
                                current_tool._partial_json += partial_json
                                stream.push(
                                    ToolCallDeltaEvent(
                                        content_index=current_block_index,
                                        delta=partial_json,
                                        partial=output,
                                    )
                                )

                    elif event_type == "content_block_stop":
                        if current_block_type == "text":
                            stream.push(
                                TextEndEvent(
                                    content_index=current_block_index,
                                    text=current_text,
                                    partial=output,
                                )
                            )
                            current_text = ""

                        elif current_block_type == "thinking":
                            stream.push(
                                ThinkingEndEvent(
                                    content_index=current_block_index,
                                    thinking=current_thinking,
                                    signature=current_thinking_signature,
                                    partial=output,
                                )
                            )
                            current_thinking = ""
                            current_thinking_signature = None

                        elif current_block_type == "tool_use" and current_tool:
                            # Parse accumulated JSON
                            try:
                                args = (
                                    json.loads(current_tool._partial_json)
                                    if current_tool._partial_json
                                    else {}
                                )
                            except json.JSONDecodeError:
                                args = {}

                            current_tool.arguments = args
                            stream.push(
                                ToolCallEndEvent(
                                    content_index=current_block_index,
                                    tool_call=current_tool,
                                    partial=output,
                                )
                            )
                            current_tool = None

                        current_block_type = None

                    elif event_type == "message_delta":
                        # Update usage and stop reason
                        delta = data.get("delta", {})
                        usage_delta = data.get("usage", {})

                        if usage_delta:
                            output.usage.output = usage_delta.get(
                                "output_tokens", output.usage.output
                            )
                            output.usage.update_total()

                        if "stop_reason" in delta:
                            output.stop_reason = _map_stop_reason(delta.get("stop_reason"))

        try:
            await with_retry(_do_stream, retry_config)

            # Skip done event if already aborted
            if stream.is_aborted:
                return

            # Calculate cost
            pricing = get_pricing(self.model, "anthropic")
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
        """Estimate token count for context management.

        Anthropic doesn't provide a public tokenizer, so we use a simple heuristic.
        ~4 characters per token is a reasonable approximation for English text.
        """
        return len(text) // 4 + 1

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
            if msg.thinking:
                total += self.count_tokens(msg.thinking.text)
        return total

    def supports_thinking(self) -> bool:
        """Check if the current model supports thinking/reasoning.

        Uses model registry to determine support. Claude 4+ models
        support extended thinking, older models don't.
        """
        from agent.llm.models import supports_reasoning

        return supports_reasoning(self.model, provider="anthropic")

    async def list_models(self) -> list[str]:
        """Return empty list - Anthropic doesn't expose model listing API."""
        # Anthropic doesn't have a public /v1/models endpoint
        return []

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
