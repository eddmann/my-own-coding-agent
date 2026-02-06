"""OpenAI Chat Completions provider with event-based streaming."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent.llm.events import (
    DoneEvent,
    ErrorEvent,
    PartialMessage,
    StartEvent,
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
from agent.llm.retry import RetryConfig, with_retry
from agent.llm.stream import AssistantMessageEventStream

from .common import (
    OpenAIBase,
    OpenAIError,
    apply_pricing,
    map_reasoning_effort,
    map_stop_reason,
    parse_api_error,
    parse_streaming_json,
)

if TYPE_CHECKING:
    from agent.core.message import Message


@dataclass(slots=True)
class OpenAIChatProvider(OpenAIBase):
    """OpenAI Chat Completions provider."""

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        options: StreamOptions | None,
    ) -> dict[str, Any]:
        """Build the Chat Completions API request payload."""
        caps = self._get_capabilities()

        max_tokens = options.max_tokens if options and options.max_tokens else self.max_tokens
        max_tokens = min(max_tokens, caps.max_output_tokens)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api_dict() for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        payload[caps.token_param] = max_tokens

        if caps.fixed_reasoning_effort:
            payload["reasoning_effort"] = caps.fixed_reasoning_effort
        elif caps.is_reasoning and options and options.thinking_level:
            effort = map_reasoning_effort(self.model, options.thinking_level)
            if effort is not None:
                payload["reasoning_effort"] = effort

        if caps.supports_temperature and not caps.is_reasoning:
            temp = (
                options.temperature
                if options and options.temperature is not None
                else self.temperature
            )
            payload["temperature"] = temp

        if tools:
            payload["tools"] = tools
            if options and options.tool_choice:
                tc = options.tool_choice
                if tc == "any":
                    payload["tool_choice"] = "required"
                elif tc == "none":
                    payload.pop("tools", None)
                elif isinstance(tc, dict) and "name" in tc:
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tc["name"]},
                    }
                else:
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
        """Stream completion as events."""
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

        text_started = False
        text_content = ""
        thinking_started = False
        thinking_content = ""
        pending_tool_calls: dict[int, ToolCallBlock] = {}
        finish_reason: str | None = None

        def _check_cancelled() -> bool:
            return bool(options and options.cancel_event and options.cancel_event.is_set())

        async def _do_stream() -> None:
            nonlocal text_started, text_content, thinking_started, thinking_content
            nonlocal pending_tool_calls, finish_reason

            if _check_cancelled():
                raise asyncio.CancelledError("Request cancelled")

            payload = self._build_payload(messages, tools, options)
            stream.push(StartEvent(partial=output))

            async with self.client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    error_msg = parse_api_error(response)
                    raise OpenAIError(error_msg, response.status_code)

                async for line in response.aiter_lines():
                    if _check_cancelled():
                        stream.abort("Request cancelled")
                        return

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("usage"):
                        usage_data = chunk["usage"]
                        cached_tokens = (
                            usage_data.get("prompt_tokens_details", {}).get("cached_tokens", 0) or 0
                        )
                        reasoning_tokens = (
                            usage_data.get("completion_tokens_details", {}).get(
                                "reasoning_tokens", 0
                            )
                            or 0
                        )
                        input_tokens = (usage_data.get("prompt_tokens", 0) or 0) - cached_tokens
                        output_tokens = (
                            usage_data.get("completion_tokens", 0) or 0
                        ) + reasoning_tokens
                        output.usage = Usage(
                            input=max(input_tokens, 0),
                            output=max(output_tokens, 0),
                            cache_read=max(cached_tokens, 0),
                            cache_write=0,
                        )
                        output.usage.total_tokens = (
                            output.usage.input + output.usage.output + output.usage.cache_read
                        )

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    reasoning_fields = ("reasoning_content", "reasoning", "reasoning_text")
                    reasoning = next(
                        (
                            delta.get(field)
                            for field in reasoning_fields
                            if delta.get(field) is not None and delta.get(field) != ""
                        ),
                        None,
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

                    if content := delta.get("content"):
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

                    if tool_calls := delta.get("tool_calls"):
                        for tc in tool_calls:
                            idx = tc.get("index", 0)

                            if idx not in pending_tool_calls:
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

                            if tc.get("id"):
                                tool_block.id = tc["id"]

                            if func := tc.get("function"):
                                if func.get("name"):
                                    tool_block.name = func["name"]
                                if func.get("arguments"):
                                    tool_block._partial_json += func["arguments"]
                                    tool_block.arguments = parse_streaming_json(
                                        tool_block._partial_json
                                    )
                                    stream.push(
                                        ToolCallDeltaEvent(
                                            content_index=len(output.content) - 1,
                                            delta=func["arguments"],
                                            partial=output,
                                        )
                                    )

        try:
            await with_retry(_do_stream, retry_config)

            if stream.is_aborted:
                return

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

            output.stop_reason = map_stop_reason(finish_reason)
            apply_pricing(output.usage, self.model)

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
