"""OpenAI Responses API provider with event-based streaming."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent.llm.events import (
    AssistantMetadataEvent,
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
)
from agent.llm.retry import RetryConfig, with_retry
from agent.llm.stream import AssistantMessageEventStream

from .common import (
    OpenAIBase,
    OpenAIError,
    apply_pricing,
    map_reasoning_effort,
    normalize_fc_id,
    parse_api_error,
    parse_streaming_json,
)

if TYPE_CHECKING:
    from agent.core.message import Message


@dataclass(slots=True)
class OpenAIResponsesProvider(OpenAIBase):
    """OpenAI Responses API provider."""

    def _split_tool_call_id(self, tool_call_id: str) -> tuple[str, str | None]:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, normalize_fc_id(item_id)
        return tool_call_id, None

    def _convert_responses_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"] or {}
                converted.append(
                    {
                        "type": "function",
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                        "strict": False,
                    }
                )
        return converted

    def _convert_responses_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        caps = self._get_capabilities()
        output: list[dict[str, Any]] = []
        msg_index = 0

        for msg in messages:
            role = msg.role.value

            if role == "system":
                role = "developer" if caps.is_reasoning else "system"
                output.append({"role": role, "content": msg.content})
                msg_index += 1
                continue

            if role == "user":
                if msg.content:
                    output.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": msg.content,
                                }
                            ],
                        }
                    )
                msg_index += 1
                continue

            if role == "assistant":
                pending_reasoning_item: dict[str, Any] | None = None

                # Reasoning item replay (if present) - only emit if followed by a message/tool
                if isinstance(msg.provider_metadata, dict):
                    responses_meta = msg.provider_metadata.get("openai_responses")
                    if isinstance(responses_meta, dict):
                        reasoning_item = responses_meta.get("reasoning_item")
                        if isinstance(reasoning_item, str):
                            try:
                                parsed = json.loads(reasoning_item)
                                if isinstance(parsed, dict):
                                    pending_reasoning_item = parsed
                            except Exception:
                                pending_reasoning_item = None

                def _emit_pending_reasoning() -> None:
                    nonlocal pending_reasoning_item
                    if pending_reasoning_item is not None:
                        output.append(pending_reasoning_item)
                        pending_reasoning_item = None

                if msg.content:
                    _emit_pending_reasoning()
                    msg_id = None
                    if isinstance(msg.provider_metadata, dict):
                        responses_meta = msg.provider_metadata.get("openai_responses")
                        if isinstance(responses_meta, dict):
                            msg_id = responses_meta.get("output_item_id")
                    if not msg_id:
                        msg_id = f"msg_{msg_index}"
                    output.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": msg.content,
                                    "annotations": [],
                                }
                            ],
                            "status": "completed",
                            "id": msg_id,
                        }
                    )

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if not tc.name:
                            continue
                        call_id, item_id = self._split_tool_call_id(tc.id)
                        if (
                            msg.model
                            and msg.model != self.model
                            and item_id
                            and item_id.startswith("fc_")
                        ):
                            item_id = None
                        if item_id:
                            _emit_pending_reasoning()
                        else:
                            pending_reasoning_item = None
                        item: dict[str, Any] = {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        }
                        if item_id:
                            item["id"] = item_id
                        output.append(item)

                msg_index += 1
                continue

            if role == "tool":
                if msg.tool_call_id:
                    call_id, _ = self._split_tool_call_id(msg.tool_call_id)
                    output.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": msg.content,
                        }
                    )
                msg_index += 1
                continue

            msg_index += 1

        return output

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        options: StreamOptions | None,
    ) -> dict[str, Any]:
        caps = self._get_capabilities()

        max_tokens = options.max_tokens if options and options.max_tokens else self.max_tokens
        max_tokens = min(max_tokens, caps.max_output_tokens)

        input_items = self._convert_responses_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "stream": True,
            "max_output_tokens": max_tokens,
        }

        reasoning_enabled = False
        if caps.fixed_reasoning_effort:
            payload["reasoning"] = {"effort": caps.fixed_reasoning_effort}
            payload["include"] = ["reasoning.encrypted_content"]
            reasoning_enabled = True
        elif caps.is_reasoning and options and options.thinking_level:
            effort = map_reasoning_effort(self.model, options.thinking_level)
            if effort is not None:
                payload["reasoning"] = {"effort": effort}
                payload["include"] = ["reasoning.encrypted_content"]
                reasoning_enabled = True

        if caps.supports_temperature and not caps.is_reasoning:
            temp = (
                options.temperature
                if options and options.temperature is not None
                else self.temperature
            )
            payload["temperature"] = temp

        if tools:
            payload["tools"] = self._convert_responses_tools(tools)
            if options and options.tool_choice:
                tc = options.tool_choice
                if tc == "any":
                    payload["tool_choice"] = "required"
                elif tc == "none":
                    payload["tool_choice"] = "none"
                elif isinstance(tc, dict) and "name" in tc:
                    payload["tool_choice"] = {"type": "function", "name": tc["name"]}
                else:
                    payload["tool_choice"] = tc

        # gpt-5: disable reasoning unless explicitly requested
        if caps.is_reasoning and not reasoning_enabled and self.model.startswith("gpt-5"):
            input_items.append(
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "# Juice: 0 !important",
                        }
                    ],
                }
            )

        return payload

    def _map_response_status(self, status: str | None) -> StopReason:
        if not status:
            return "stop"
        if status == "completed":
            return "stop"
        if status == "incomplete":
            return "length"
        if status in ("failed", "cancelled"):
            return "error"
        if status in ("in_progress", "queued"):
            return "stop"
        return "stop"

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()
        stream.attach_task(asyncio.create_task(self._stream_impl(messages, tools, options, stream)))
        return stream

    async def _stream_impl(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        options: StreamOptions | None,
        stream: AssistantMessageEventStream,
    ) -> None:
        output = PartialMessage()
        retry_config = RetryConfig()

        current_item: dict[str, Any] | None = None
        current_block: TextBlock | ThinkingBlock | ToolCallBlock | None = None

        def _check_cancelled() -> bool:
            return bool(options and options.cancel_event and options.cancel_event.is_set())

        def _content_index() -> int:
            return len(output.content) - 1

        async def _do_stream() -> None:
            nonlocal current_item, current_block

            if _check_cancelled():
                raise asyncio.CancelledError("Request cancelled")

            payload = self._build_payload(messages, tools, options)
            stream.push(StartEvent(partial=output))

            async with self.client.stream(
                "POST",
                "/v1/responses",
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
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")
                    if not event_type:
                        continue

                    if event_type == "response.output_item.added":
                        item = event.get("item") or {}
                        if item.get("type") == "reasoning":
                            current_item = item
                            current_block = ThinkingBlock(thinking="")
                            output.content.append(current_block)
                            stream.push(
                                ThinkingStartEvent(
                                    content_index=_content_index(),
                                    partial=output,
                                )
                            )
                        elif item.get("type") == "message":
                            current_item = item
                            current_block = TextBlock(text="")
                            output.content.append(current_block)
                            stream.push(
                                TextStartEvent(
                                    content_index=_content_index(),
                                    partial=output,
                                )
                            )
                        elif item.get("type") == "function_call":
                            current_item = item
                            call_id = item.get("call_id", "")
                            item_id = item.get("id")
                            if item_id:
                                item_id = normalize_fc_id(str(item_id))
                            tool_id = f"{call_id}|{item_id}" if item_id else call_id
                            tool_block = ToolCallBlock(
                                id=tool_id,
                                name=item.get("name", ""),
                                arguments={},
                            )
                            tool_block.set_arguments_raw_json(item.get("arguments", "") or "")
                            current_block = tool_block
                            output.content.append(tool_block)
                            stream.push(
                                ToolCallStartEvent(
                                    content_index=_content_index(),
                                    tool_id=current_block.id,
                                    tool_name=current_block.name,
                                    partial=output,
                                )
                            )

                    elif event_type == "response.reasoning_summary_part.added":
                        if current_item and current_item.get("type") == "reasoning":
                            current_item.setdefault("summary", []).append(event.get("part"))

                    elif event_type in (
                        "response.reasoning_summary_text.delta",
                        "response.reasoning_text.delta",
                    ):
                        if (
                            current_item
                            and current_item.get("type") == "reasoning"
                            and isinstance(current_block, ThinkingBlock)
                        ):
                            delta = event.get("delta", "")
                            if delta:
                                summary = current_item.get("summary")
                                if isinstance(summary, list) and summary:
                                    last_part = summary[-1]
                                    if isinstance(last_part, dict):
                                        last_part["text"] = f"{last_part.get('text', '')}{delta}"
                                current_block.thinking += delta
                                stream.push(
                                    ThinkingDeltaEvent(
                                        content_index=_content_index(),
                                        delta=delta,
                                        partial=output,
                                    )
                                )

                    elif event_type == "response.reasoning_summary_part.done":
                        if (
                            current_item
                            and current_item.get("type") == "reasoning"
                            and isinstance(current_block, ThinkingBlock)
                        ):
                            summary = current_item.get("summary")
                            if isinstance(summary, list) and summary:
                                last_part = summary[-1]
                                if isinstance(last_part, dict):
                                    last_part["text"] = f"{last_part.get('text', '')}\n\n"
                            current_block.thinking += "\n\n"
                            stream.push(
                                ThinkingDeltaEvent(
                                    content_index=_content_index(),
                                    delta="\n\n",
                                    partial=output,
                                )
                            )

                    elif event_type in ("response.output_text.delta", "response.refusal.delta"):
                        if (
                            current_item
                            and current_item.get("type") == "message"
                            and isinstance(current_block, TextBlock)
                        ):
                            delta = event.get("delta", "")
                            if delta:
                                current_block.text += delta
                                stream.push(
                                    TextDeltaEvent(
                                        content_index=_content_index(),
                                        delta=delta,
                                        partial=output,
                                    )
                                )

                    elif event_type == "response.function_call_arguments.delta":
                        if (
                            current_item
                            and current_item.get("type") == "function_call"
                            and isinstance(current_block, ToolCallBlock)
                        ):
                            delta = event.get("delta", "")
                            current_block.arguments = parse_streaming_json(
                                current_block.append_arguments_delta(delta)
                            )
                            stream.push(
                                ToolCallDeltaEvent(
                                    content_index=_content_index(),
                                    delta=delta,
                                    partial=output,
                                )
                            )

                    elif event_type == "response.function_call_arguments.done":
                        if (
                            current_item
                            and current_item.get("type") == "function_call"
                            and isinstance(current_block, ToolCallBlock)
                        ):
                            current_block.arguments = parse_streaming_json(
                                current_block.set_arguments_raw_json(event.get("arguments", ""))
                            )

                    elif event_type == "response.output_item.done":
                        item = event.get("item") or {}
                        if item.get("type") == "reasoning" and isinstance(
                            current_block, ThinkingBlock
                        ):
                            summary = item.get("summary") or []
                            text = (
                                "\n\n".join(part.get("text", "") for part in summary)
                                if summary
                                else current_block.thinking
                            )
                            current_block.thinking = text or ""
                            signature = json.dumps(item)
                            stream.push(
                                ThinkingEndEvent(
                                    content_index=_content_index(),
                                    thinking=current_block.thinking,
                                    signature=signature,
                                    partial=output,
                                )
                            )
                            stream.push(
                                AssistantMetadataEvent(
                                    metadata={"openai_responses": {"reasoning_item": signature}},
                                    partial=output,
                                )
                            )
                            current_block = None
                            current_item = None
                        elif item.get("type") == "message" and isinstance(current_block, TextBlock):
                            content = item.get("content") or []
                            text = "".join(
                                str(part.get("text") or part.get("refusal") or "")
                                for part in content
                                if isinstance(part, dict)
                            )
                            text_signature = (
                                item.get("id") or current_item.get("id") if current_item else None
                            )
                            current_block.text = text or current_block.text
                            stream.push(
                                TextEndEvent(
                                    content_index=_content_index(),
                                    text=current_block.text,
                                    partial=output,
                                )
                            )
                            if text_signature:
                                stream.push(
                                    AssistantMetadataEvent(
                                        metadata={
                                            "openai_responses": {"output_item_id": text_signature}
                                        },
                                        partial=output,
                                    )
                                )
                            current_block = None
                            current_item = None
                        elif item.get("type") == "function_call":
                            if isinstance(current_block, ToolCallBlock):
                                args_str = current_block.arguments_raw_json or item.get(
                                    "arguments", ""
                                )
                                try:
                                    args = json.loads(args_str) if args_str else {}
                                except json.JSONDecodeError:
                                    args = {}
                                current_block.arguments = args
                                stream.push(
                                    ToolCallEndEvent(
                                        content_index=_content_index(),
                                        tool_call=current_block,
                                        partial=output,
                                    )
                                )
                            current_block = None
                            current_item = None

                    elif event_type == "response.completed":
                        resp = event.get("response") or {}
                        usage_data = resp.get("usage") or event.get("usage") or {}
                        cached_tokens = 0
                        input_tokens = 0
                        output_tokens = 0
                        if usage_data:
                            cached_tokens = (
                                usage_data.get("input_tokens_details", {}).get("cached_tokens", 0)
                                or 0
                            )
                            input_tokens = (usage_data.get("input_tokens", 0) or 0) - cached_tokens
                            output_tokens = usage_data.get("output_tokens", 0) or 0
                            output.usage.input = max(input_tokens, 0)
                            output.usage.output = max(output_tokens, 0)
                            output.usage.cache_read = max(cached_tokens, 0)
                            output.usage.total_tokens = (
                                output.usage.input + output.usage.output + output.usage.cache_read
                            )

                        output.stop_reason = self._map_response_status(resp.get("status"))
                        if (
                            any(isinstance(b, ToolCallBlock) for b in output.content)
                            and output.stop_reason == "stop"
                        ):
                            output.stop_reason = "tool_use"

                    elif event_type in ("response.failed", "error"):
                        err = event.get("error") or {}
                        message = err.get("message") or str(err) or "Request failed"
                        raise OpenAIError(message)

        try:
            await with_retry(_do_stream, retry_config)

            if stream.is_aborted:
                return

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
