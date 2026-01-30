"""OpenAI-compatible LLM provider with event-based streaming.

Works with: Ollama, LM Studio, OpenRouter, Groq, Together, Mistral, etc.

This is a simplified provider that focuses on compatibility rather than
advanced features like reasoning_effort. Use openai.py for native OpenAI.

Supports:
- Basic tool use with tool_choice modes
- Request cancellation and retry with exponential backoff
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import httpx
import tiktoken

from agent.llm.events import (
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
    ToolCallBlock,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)
from agent.llm.retry import RetryConfig, with_retry
from agent.llm.stream import AssistantMessageEventStream

if TYPE_CHECKING:
    from agent.core.message import Message


class LLMError(Exception):
    """Error from the LLM API with parsed message."""

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
class CompatSettings:
    """Provider-specific compatibility settings."""

    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    supports_developer_role: bool = False
    requires_tool_result_name: bool = False
    requires_thinking_as_text: bool = False
    tool_id_max_length: int | None = None  # Some providers limit tool ID length


def _detect_compat_settings(base_url: str) -> CompatSettings:
    """Auto-detect compatibility settings from base URL."""
    url_lower = base_url.lower()

    # Mistral quirks
    if "mistral.ai" in url_lower:
        return CompatSettings(
            tool_id_max_length=9,  # Mistral limits tool IDs to 9 chars
            requires_thinking_as_text=True,
        )

    # Groq defaults
    if "groq.com" in url_lower:
        return CompatSettings()

    # OpenRouter defaults
    if "openrouter.ai" in url_lower:
        return CompatSettings()

    # Local providers (Ollama, LM Studio)
    if "localhost" in url_lower or "127.0.0.1" in url_lower:
        return CompatSettings()

    # Default settings
    return CompatSettings()


@dataclass(slots=True)
class OpenAICompatibleProvider:
    """Provider for any OpenAI-compatible API."""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    _encoder: tiktoken.Encoding | None = field(default=None, repr=False)
    _compat: CompatSettings | None = field(default=None, repr=False)

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
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

    @property
    def compat(self) -> CompatSettings:
        """Get compatibility settings."""
        if self._compat is None:
            self._compat = _detect_compat_settings(self.base_url)
        return self._compat

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        options: StreamOptions | None,
    ) -> dict[str, Any]:
        """Build the API request payload."""
        max_tokens = options.max_tokens if options and options.max_tokens else self.max_tokens
        temp = (
            options.temperature if options and options.temperature is not None else self.temperature
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api_dict() for m in messages],
            "stream": True,
            self.compat.max_tokens_field: max_tokens,
            "temperature": temp,
        }

        if tools:
            # Apply tool ID length limit if needed
            if self.compat.tool_id_max_length:
                tools = self._truncate_tool_ids(tools)
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

    def _truncate_tool_ids(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Truncate tool IDs if provider requires it."""
        # This is mainly for Mistral which has a 9-char limit
        # The actual truncation happens when processing tool calls
        return tools

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
        # Store task reference to prevent garbage collection
        stream._task = asyncio.create_task(self._stream_impl(messages, tools, options, stream))
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
        pending_tool_calls: dict[int, ToolCallBlock] = {}
        finish_reason: str | None = None

        def _check_cancelled() -> bool:
            """Check if request was cancelled."""
            return bool(options and options.cancel_event and options.cancel_event.is_set())

        async def _do_stream() -> None:
            """Execute the streaming request (for retry)."""
            nonlocal text_started, text_content, pending_tool_calls, finish_reason

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
                    raise LLMError(error_msg, response.status_code)

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

                    # Handle usage if present
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

                    # Handle content tokens
                    if content := delta.get("content"):
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
                                tool_id = tc.get("id", "")
                                # Truncate ID if needed
                                if (
                                    self.compat.tool_id_max_length
                                    and len(tool_id) > self.compat.tool_id_max_length
                                ):
                                    tool_id = tool_id[: self.compat.tool_id_max_length]

                                tool_block = ToolCallBlock(
                                    id=tool_id,
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
                                tool_id = tc["id"]
                                if (
                                    self.compat.tool_id_max_length
                                    and len(tool_id) > self.compat.tool_id_max_length
                                ):
                                    tool_id = tool_id[: self.compat.tool_id_max_length]
                                tool_block.id = tool_id

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

            # End text block if started
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

    def supports_thinking(self) -> bool:
        """Check if the current model supports thinking/reasoning.

        Checks model registry first. For OpenAI-compatible providers,
        this may return True if proxying a reasoning-capable model
        (e.g., via OpenRouter).
        """
        from agent.llm.models import get_model_info, supports_reasoning

        # Check registry first (handles known models via OpenRouter etc.)
        info = get_model_info(self.model)
        if info is not None:
            return info.reasoning

        # Fallback: check if model name suggests reasoning support
        return supports_reasoning(self.model, provider="openai-compat")

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
