"""OpenAI Codex provider (ChatGPT OAuth backend)."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import tiktoken

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

from .oauth import (
    extract_account_id,
    load_oauth_credentials,
    refresh_openai_codex_token,
    save_oauth_credentials,
)

if TYPE_CHECKING:
    from pathlib import Path

    from agent.core.message import Message

OPENAI_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
OPENAI_CODEX_RESPONSES_PATH = "/codex/responses"
DEFAULT_CODEX_INSTRUCTIONS = "You are a helpful coding assistant."


class OpenAICodexError(Exception):
    """Error from the OpenAI Codex backend."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


@dataclass(slots=True)
class CodexModelCapabilities:
    """Model-specific capabilities and parameter support."""

    supports_temperature: bool = True
    is_reasoning: bool = False
    fixed_reasoning_effort: str | None = None
    max_output_tokens: int = 16384


def _get_model_capabilities(model: str) -> CodexModelCapabilities:
    model_lower = model.lower()

    if model_lower.startswith("gpt-5-pro"):
        return CodexModelCapabilities(
            supports_temperature=False,
            is_reasoning=True,
            fixed_reasoning_effort="high",
        )

    if model_lower.startswith(("o1", "o3", "gpt-5")):
        return CodexModelCapabilities(
            supports_temperature=False,
            is_reasoning=True,
        )

    return CodexModelCapabilities(
        supports_temperature=True,
        is_reasoning=False,
    )


def _map_reasoning_effort(model: str, thinking_level: str | None) -> str | None:
    if not thinking_level:
        return None

    if thinking_level in {"minimal", "low", "medium", "high"}:
        effort_map = {
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
        }
        return effort_map[thinking_level]

    if thinking_level != "xhigh":
        return None

    from agent.llm.models import supports_xhigh

    return "xhigh" if supports_xhigh(model, provider="openai-codex") else "high"


def _normalize_fc_id(value: str) -> str:
    if not value:
        return value
    if value.startswith("fc"):
        return value
    return f"fc_{value}"


def _parse_api_error(response: httpx.Response) -> str:
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


def _parse_streaming_json(text: str) -> dict[str, object]:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except Exception:
        for i in range(len(text) - 1, 1, -1):
            try:
                parsed = json.loads(text[:i])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return {}


def _apply_pricing(usage: Usage, model: str) -> None:
    pricing = get_pricing(model, "openai")
    if not pricing:
        return

    usage.cost = Cost(
        input=(usage.input / 1_000_000) * pricing.input,
        output=(usage.output / 1_000_000) * pricing.output,
        cache_read=(usage.cache_read / 1_000_000) * pricing.cache_read,
        cache_write=(usage.cache_write / 1_000_000) * pricing.cache_write,
    )
    usage.cost.total = (
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )


def _is_expired(expires_ms: int) -> bool:
    return expires_ms <= int(time.time() * 1000)


@dataclass(slots=True)
class OpenAICodexProvider:
    """OpenAI Codex provider with OAuth token refresh support."""

    api_key: str
    model: str = "gpt-5-codex"
    temperature: float = 0.7
    max_tokens: int = 8192
    base_url: str = OPENAI_CODEX_BASE_URL
    http_client: httpx.AsyncClient | None = field(default=None, repr=False)
    oauth_path: Path | None = None
    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    _encoder: tiktoken.Encoding | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.http_client is not None:
            self._client = self.http_client

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    @property
    def encoder(self) -> tiktoken.Encoding:
        if self._encoder is None:
            try:
                self._encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    def _get_capabilities(self) -> CodexModelCapabilities:
        return _get_model_capabilities(self.model)

    def _request_headers(self, token: str, account_id: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "agent",
        }

    def _split_tool_call_id(self, tool_call_id: str) -> tuple[str, str | None]:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, _normalize_fc_id(item_id)
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
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)

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

        instructions, api_messages = self._extract_instructions(messages)
        input_items = self._convert_responses_messages(api_messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": input_items,
            "store": False,
            "stream": True,
        }

        reasoning_enabled = False
        if caps.fixed_reasoning_effort:
            payload["reasoning"] = {"effort": caps.fixed_reasoning_effort}
            payload["include"] = ["reasoning.encrypted_content"]
            reasoning_enabled = True
        elif caps.is_reasoning and options and options.thinking_level:
            effort = _map_reasoning_effort(self.model, options.thinking_level)
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

    def _extract_instructions(self, messages: list[Message]) -> tuple[str, list[Message]]:
        """Move system messages into top-level instructions for Codex backend."""
        instructions_parts: list[str] = []
        api_messages: list[Message] = []

        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            if role == "system":
                if msg.content:
                    instructions_parts.append(msg.content)
                continue
            api_messages.append(msg)

        instructions = "\n\n".join(part for part in instructions_parts if part).strip()
        if not instructions:
            instructions = DEFAULT_CODEX_INSTRUCTIONS
        return instructions, api_messages

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

    async def _resolve_credentials(self) -> tuple[str, str]:
        resolved_key = self.api_key or ""
        if resolved_key:
            try:
                return resolved_key, extract_account_id(resolved_key)
            except Exception as exc:  # pragma: no cover - defensive parse guard
                raise OpenAICodexError(
                    "Invalid OpenAI Codex token. Use `agent auth login openai-codex`."
                ) from exc

        creds = load_oauth_credentials(self.oauth_path)
        if creds:
            if _is_expired(creds.expires):
                try:
                    creds = await refresh_openai_codex_token(creds.refresh)
                    save_oauth_credentials(creds, self.oauth_path)
                except Exception as exc:
                    raise OpenAICodexError(
                        "OpenAI Codex OAuth token refresh failed. "
                        "Run `agent auth login openai-codex`."
                    ) from exc
            return creds.access, creds.account_id

        raise OpenAICodexError("No OpenAI Codex OAuth credentials configured")

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()
        task = asyncio.create_task(self._stream_impl(messages, tools, options, stream))
        stream._task = task  # noqa: SLF001 - needed for compatibility with cancellation flow
        return stream

    async def _stream_impl(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None,
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

        resolved_key = ""
        account_id = ""

        async def _do_stream() -> None:
            nonlocal current_item, current_block

            if _check_cancelled():
                raise asyncio.CancelledError("Request cancelled")

            payload = self._build_payload(messages, tools, options)
            stream.push(StartEvent(partial=output))

            async with self.client.stream(
                "POST",
                OPENAI_CODEX_RESPONSES_PATH,
                json=payload,
                headers=self._request_headers(resolved_key, account_id),
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    error_msg = _parse_api_error(response)
                    raise OpenAICodexError(error_msg, response.status_code)

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
                                item_id = _normalize_fc_id(str(item_id))
                            tool_id = f"{call_id}|{item_id}" if item_id else call_id
                            current_block = ToolCallBlock(
                                id=tool_id,
                                name=item.get("name", ""),
                                arguments={},
                                _partial_json=item.get("arguments", "") or "",
                            )
                            output.content.append(current_block)
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
                            current_block._partial_json += delta
                            current_block.arguments = _parse_streaming_json(
                                current_block._partial_json
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
                            current_block._partial_json = event.get("arguments", "")
                            current_block.arguments = _parse_streaming_json(
                                current_block._partial_json
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
                                args_str = current_block._partial_json or item.get("arguments", "")
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
                        raise OpenAICodexError(message)

        try:
            resolved_key, account_id = await self._resolve_credentials()
            await with_retry(_do_stream, retry_config)

            if stream.is_aborted:
                return

            _apply_pricing(output.usage, self.model)
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
        return len(self.encoder.encode(text))

    def count_messages_tokens(self, messages: list[Message]) -> int:
        total = 0
        for msg in messages:
            total += 4
            total += self.count_tokens(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total += self.count_tokens(tc.name)
                    total += self.count_tokens(json.dumps(tc.arguments))
        return total

    def supports_thinking(self) -> bool:
        from agent.llm.models import get_model_info, supports_reasoning

        info = get_model_info(self.model)
        if info is not None:
            return info.reasoning
        return supports_reasoning(self.model, provider="openai-codex")

    async def list_models(self) -> list[str]:
        from agent.llm.models import MODELS

        models = [
            model_id
            for model_id, info in MODELS.items()
            if info.provider == "openai" and ("codex" in model_id or model_id.startswith("gpt-5"))
        ]
        return sorted(models)

    async def close(self) -> None:
        if self._client is not None and self.http_client is None:
            await self._client.aclose()
            self._client = None
