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
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import httpx

from agent.core.message import Message, Role
from agent.core.settings import THINKING_BUDGETS, ThinkingLevel
from agent.llm.events import (
    AssistantMetadataEvent,
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

from .oauth import load_oauth_credentials, refresh_anthropic_token, save_oauth_credentials

ANTHROPIC_API_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"

# Claude Code emulation
CLAUDE_CODE_VERSION = "2.1.2"
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
CLAUDE_CODE_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
]
_CC_TOOL_LOOKUP = {name.lower(): name for name in CLAUDE_CODE_TOOLS}


def _to_cc_tool_name(name: str) -> str:
    return _CC_TOOL_LOOKUP.get(name.lower(), name)


def _build_cc_tool_map(tools: list[dict[str, Any]] | None) -> dict[str, str]:
    if not tools:
        return {}
    mapping: dict[str, str] = {}
    for tool in tools:
        func = tool.get("function", {})
        tool_name = func.get("name")
        if isinstance(tool_name, str):
            mapping[_to_cc_tool_name(tool_name).lower()] = tool_name
    return mapping


def _is_oauth_token(api_key: str) -> bool:
    return api_key.startswith("sk-ant-oat")


def _is_expired(expires_ms: int | None) -> bool:
    if expires_ms is None:
        return False
    return expires_ms <= int(time.time() * 1000)


def _build_headers(api_key: str, use_oauth: bool, thinking_enabled: bool) -> dict[str, str]:
    beta_features = ["fine-grained-tool-streaming-2025-05-14"]
    if thinking_enabled:
        beta_features.append("interleaved-thinking-2025-05-14")

    if use_oauth:
        beta = f"claude-code-20250219,oauth-2025-04-20,{','.join(beta_features)}"
    else:
        beta = ",".join(beta_features)

    headers: dict[str, str] = {
        "accept": "application/json",
        "anthropic-dangerous-direct-browser-access": "true",
        "anthropic-beta": beta,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    if use_oauth:
        headers["authorization"] = f"Bearer {api_key}"
        headers["user-agent"] = f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)"
        headers["x-app"] = "cli"
    else:
        headers["x-api-key"] = api_key

    return headers


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


def _supports_adaptive_thinking(model_id: str) -> bool:
    """Return whether model uses Anthropic adaptive thinking (Opus 4.6+)."""
    model_lower = model_id.lower()
    return "opus-4-6" in model_lower or "opus-4.6" in model_lower


def _map_thinking_level_to_effort(level: ThinkingLevel) -> str:
    """Map local thinking levels to Anthropic adaptive effort values."""
    mapping = {
        ThinkingLevel.MINIMAL: "low",
        ThinkingLevel.LOW: "low",
        ThinkingLevel.MEDIUM: "medium",
        ThinkingLevel.HIGH: "high",
        ThinkingLevel.XHIGH: "max",
    }
    return mapping.get(level, "high")


@dataclass(slots=True)
class AnthropicProvider:
    """Native Anthropic API provider with full feature support."""

    api_key: str
    name: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192
    http_client: httpx.AsyncClient | None = field(default=None, repr=False)
    oauth_path: Path | None = None
    retry_config: RetryConfig = field(default_factory=RetryConfig, repr=False)
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.http_client is not None:
            self._client = self.http_client

    def set_model(self, model: str) -> None:
        """Update the active model."""
        from agent.llm.models import is_model_valid_for_provider

        if not model or model == self.model:
            return
        if not is_model_valid_for_provider(model, self.name):
            raise ValueError(f"Model '{model}' is not valid for provider '{self.name}'")
        self.model = model

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=ANTHROPIC_API_URL,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    def _convert_messages(
        self, messages: list[Message], use_oauth: bool = False
    ) -> tuple[str, list[dict[str, Any]]]:
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
                content = self._build_assistant_content(msg, use_oauth)
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

    def _build_assistant_content(
        self, msg: Message, use_oauth: bool = False
    ) -> list[dict[str, Any]] | str:
        """Build content array for assistant message."""
        content_blocks: list[dict[str, Any]] = []

        # Add thinking block if present
        if msg.thinking:
            block: dict[str, Any] = {"type": "thinking", "thinking": msg.thinking.text}
            signature = None
            if isinstance(msg.provider_metadata, dict):
                meta = msg.provider_metadata.get("anthropic")
                if isinstance(meta, dict):
                    signature = meta.get("thinking_signature")
            if signature:
                block["signature"] = signature
            content_blocks.append(block)

        # Add text content
        if msg.content:
            content_blocks.append({"type": "text", "text": msg.content})

        # Add tool use blocks
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = _to_cc_tool_name(tc.name) if use_oauth else tc.name
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tool_name,
                        "input": tc.arguments,
                    }
                )

        # Return string if only text, otherwise return array
        if len(content_blocks) == 1 and content_blocks[0]["type"] == "text":
            return msg.content
        return content_blocks

    def _convert_tools(self, tools: list[dict[str, Any]], use_oauth: bool) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        return [
            {
                "name": (
                    _to_cc_tool_name(tool["function"]["name"])
                    if use_oauth
                    else tool["function"]["name"]
                ),
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
        use_oauth: bool,
    ) -> dict[str, Any]:
        """Build the API request payload."""
        system_prompt, api_messages = self._convert_messages(messages, use_oauth)

        max_tokens = options.max_tokens if options and options.max_tokens else self.max_tokens

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": api_messages,
            "stream": True,
        }

        if use_oauth:
            system_blocks: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": CLAUDE_CODE_IDENTITY,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            if system_prompt:
                system_blocks.append(
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                )
            payload["system"] = system_blocks
        elif system_prompt:
            payload["system"] = system_prompt

        if tools:
            payload["tools"] = self._convert_tools(tools, use_oauth)

        # Configure extended thinking if enabled
        thinking_level = None
        if options and options.thinking_level and options.thinking_level != "off":
            with contextlib.suppress(ValueError):
                thinking_level = ThinkingLevel(options.thinking_level)

        if thinking_level and thinking_level != ThinkingLevel.OFF:
            if _supports_adaptive_thinking(self.model):
                # Opus 4.6+ uses adaptive thinking with explicit effort levels.
                payload["thinking"] = {"type": "adaptive"}
                payload["output_config"] = {"effort": _map_thinking_level_to_effort(thinking_level)}
            else:
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
                tool_name = tc["name"]
                if use_oauth:
                    tool_name = _to_cc_tool_name(tool_name)
                payload["tool_choice"] = {"type": "tool", "name": tool_name}
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
        api_key = options.api_key if options and options.api_key else self.api_key
        stream.attach_task(
            asyncio.create_task(self._stream_impl(messages, tools, options, stream, api_key))
        )
        return stream

    async def _stream_impl(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        options: StreamOptions | None,
        stream: AssistantMessageEventStream,
        api_key: str,
    ) -> None:
        """Internal implementation of streaming with retry and cancellation."""
        output = PartialMessage()

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

            resolved_key = api_key
            use_oauth = _is_oauth_token(resolved_key) if resolved_key else False
            if not resolved_key:
                creds = load_oauth_credentials(self.oauth_path)
                if creds:
                    if _is_expired(creds.expires):
                        try:
                            refreshed = await refresh_anthropic_token(creds.refresh)
                            save_oauth_credentials(refreshed, self.oauth_path)
                            creds = refreshed
                        except Exception:
                            creds = None
                    if creds:
                        resolved_key = creds.access
                        use_oauth = True

            if not resolved_key:
                raise AnthropicError("No Anthropic API key or OAuth credentials configured")
            if not use_oauth and not resolved_key.startswith("sk-ant-"):
                raise AnthropicError(
                    "Invalid Anthropic API key (expected sk-ant-*). "
                    "Set ANTHROPIC_API_KEY or use Anthropic OAuth."
                )

            tool_name_map = _build_cc_tool_map(tools) if use_oauth else {}

            thinking_enabled = bool(
                options and options.thinking_level and options.thinking_level != "off"
            )
            payload = self._build_payload(messages, tools, options, use_oauth)
            headers = _build_headers(resolved_key, use_oauth, thinking_enabled)

            # Emit start event
            stream.push(StartEvent(partial=output))

            async with self.client.stream(
                "POST",
                "/v1/messages",
                json=payload,
                headers=headers,
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
                            stream.push(TextStartEvent(content_index=current_block_index))

                        elif block_type == "thinking":
                            current_thinking = block.get("thinking", "")
                            current_thinking_signature = None
                            output.content.append(ThinkingBlock(thinking=current_thinking))
                            stream.push(ThinkingStartEvent(content_index=current_block_index))

                        elif block_type == "tool_use":
                            tool_name = block.get("name", "")
                            if use_oauth and tool_name_map:
                                tool_name = tool_name_map.get(tool_name.lower(), tool_name)
                            current_tool = ToolCallBlock(
                                id=block.get("id", ""),
                                name=tool_name,
                            )
                            output.content.append(current_tool)
                            stream.push(
                                ToolCallStartEvent(
                                    content_index=current_block_index,
                                    tool_id=current_tool.id,
                                    tool_name=current_tool.name,
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
                                    TextDeltaEvent(content_index=current_block_index, delta=text)
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
                                        content_index=current_block_index, delta=thinking_text
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
                                current_tool.append_arguments_delta(partial_json)
                                stream.push(
                                    ToolCallDeltaEvent(
                                        content_index=current_block_index, delta=partial_json
                                    )
                                )

                    elif event_type == "content_block_stop":
                        if current_block_type == "text":
                            stream.push(
                                TextEndEvent(content_index=current_block_index, text=current_text)
                            )
                            current_text = ""

                        elif current_block_type == "thinking":
                            stream.push(
                                ThinkingEndEvent(
                                    content_index=current_block_index,
                                    thinking=current_thinking,
                                    signature=current_thinking_signature,
                                )
                            )
                            if current_thinking_signature:
                                stream.push(
                                    AssistantMetadataEvent(
                                        metadata={
                                            "anthropic": {
                                                "thinking_signature": current_thinking_signature
                                            }
                                        }
                                    )
                                )
                            current_thinking = ""
                            current_thinking_signature = None

                        elif current_block_type == "tool_use" and current_tool:
                            # Parse accumulated JSON
                            try:
                                raw_args = current_tool.arguments_raw_json
                                args = json.loads(raw_args) if raw_args else {}
                            except json.JSONDecodeError:
                                args = {}

                            current_tool.arguments = args
                            stream.push(
                                ToolCallEndEvent(
                                    content_index=current_block_index, tool_call=current_tool
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
            await with_retry(_do_stream, self.retry_config)

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
        """Return known Anthropic models from the local registry."""
        from agent.llm.models import MODELS

        models = [model_id for model_id, info in MODELS.items() if info.provider == "anthropic"]
        return sorted(models)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
