"""Anthropic provider tests using MockTransport (behavioral)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from agent.core.message import Message, Role, ThinkingContent
from agent.llm.anthropic import oauth as oauth_mod
from agent.llm.anthropic.oauth import OAuthCredentials
from agent.llm.anthropic.provider import ANTHROPIC_API_URL, AnthropicProvider
from agent.llm.events import (
    AssistantMetadataEvent,
    StreamOptions,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)


def _sse_payload(events: list[dict[str, object]]) -> bytes:
    chunks = []
    for event in events:
        chunks.append(f"data: {json.dumps(event)}\n\n".encode())
    chunks.append(b"data: [DONE]\n\n")
    return b"".join(chunks)


@pytest.mark.asyncio
async def test_anthropic_streaming_parses_thinking_text_and_tool_blocks() -> None:
    input_tokens = 3
    initial_output_tokens = 0
    output_tokens = 7
    thinking_index = 0
    text_index = 1
    tool_index = 2
    tool_id = "tool_1"
    tool_name = "read"
    tool_args_json = '{"path": "/tmp/x"}'
    sig_prefix = "sig_"
    sig_suffix = "123"

    events = [
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": initial_output_tokens,
                }
            },
        },
        {
            "type": "content_block_start",
            "index": thinking_index,
            "content_block": {"type": "thinking", "thinking": "a"},
        },
        {
            "type": "content_block_delta",
            "index": thinking_index,
            "delta": {"type": "thinking_delta", "thinking": "b"},
        },
        {
            "type": "content_block_delta",
            "index": thinking_index,
            "delta": {"type": "signature_delta", "signature": sig_prefix},
        },
        {
            "type": "content_block_delta",
            "index": thinking_index,
            "delta": {"type": "signature_delta", "signature": sig_suffix},
        },
        {"type": "content_block_stop", "index": thinking_index},
        {
            "type": "content_block_start",
            "index": text_index,
            "content_block": {"type": "text", "text": "Hel"},
        },
        {
            "type": "content_block_delta",
            "index": text_index,
            "delta": {"type": "text_delta", "text": "lo"},
        },
        {"type": "content_block_stop", "index": text_index},
        {
            "type": "content_block_start",
            "index": tool_index,
            "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name},
        },
        {
            "type": "content_block_delta",
            "index": tool_index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": tool_args_json,
            },
        },
        {"type": "content_block_stop", "index": tool_index},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": output_tokens},
        },
    ]

    payload = _sse_payload(events)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        return httpx.Response(200, content=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=ANTHROPIC_API_URL, transport=transport)
    provider = AnthropicProvider(
        api_key="sk-ant-test",
        model="claude-sonnet-4-20250514",
        http_client=client,
    )

    stream = provider.stream([Message.user("hi")])
    streamed_events = [event async for event in stream]
    result = await stream.result()

    await provider.close()

    assert result.usage.input == input_tokens
    assert result.usage.output == output_tokens

    thinking = next(block for block in result.content if isinstance(block, ThinkingBlock))
    text = next(block for block in result.content if isinstance(block, TextBlock))
    tool = next(block for block in result.content if isinstance(block, ToolCallBlock))

    assert thinking.thinking == "ab"
    assert thinking.signature == f"{sig_prefix}{sig_suffix}"
    assert text.text == "Hello"
    assert tool.id == tool_id
    assert tool.name == tool_name
    assert tool.arguments == {"path": "/tmp/x"}

    assert any(isinstance(event, AssistantMetadataEvent) for event in streamed_events)


@pytest.mark.asyncio
async def test_anthropic_oauth_headers_and_tool_mapping(tmp_path) -> None:
    captured = SimpleNamespace(headers=None, json=None)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.headers = dict(request.headers)
        captured.json = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse_payload([]))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=ANTHROPIC_API_URL, transport=transport)
    oauth_path = tmp_path / "anthropic-oauth.json"
    provider = AnthropicProvider(
        api_key="",
        model="claude-sonnet-4-20250514",
        http_client=client,
        oauth_path=oauth_path,
    )

    access_token = "access_token"
    expires_ms = 9_999_999_999_999
    creds = OAuthCredentials(refresh="r", access=access_token, expires=expires_ms)
    oauth_mod.save_oauth_credentials(creds, oauth_path)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    ]
    options = StreamOptions(thinking_level="minimal")

    stream = provider.stream([Message.user("hi")], tools=tools, options=options)
    await stream.result()

    await provider.close()

    assert captured.headers is not None
    assert captured.json is not None
    assert captured.headers.get("authorization") == f"Bearer {access_token}"
    assert "x-api-key" not in captured.headers
    assert "claude-code-20250219" in captured.headers.get("anthropic-beta", "")
    assert "oauth-2025-04-20" in captured.headers.get("anthropic-beta", "")
    assert "interleaved-thinking-2025-05-14" in captured.headers.get("anthropic-beta", "")

    tool_names = [tool["name"] for tool in captured.json.get("tools", [])]
    assert tool_names == ["Read"]


@pytest.mark.asyncio
async def test_anthropic_api_key_headers(tmp_path) -> None:
    captured = SimpleNamespace(headers=None)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.headers = dict(request.headers)
        return httpx.Response(200, content=_sse_payload([]))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=ANTHROPIC_API_URL, transport=transport)
    api_key = "sk-ant-test"
    oauth_path = tmp_path / "missing-oauth.json"
    provider = AnthropicProvider(
        api_key=api_key,
        model="claude-sonnet-4-20250514",
        http_client=client,
        oauth_path=oauth_path,
    )

    stream = provider.stream([Message.user("hi")])
    await stream.result()

    await provider.close()

    assert captured.headers is not None
    assert captured.headers.get("x-api-key") == api_key
    assert "authorization" not in captured.headers
    assert "claude-code-20250219" not in captured.headers.get("anthropic-beta", "")


@pytest.mark.asyncio
async def test_anthropic_payload_preserves_system_and_thinking_signature() -> None:
    captured = SimpleNamespace(json=None)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.json = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse_payload([]))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=ANTHROPIC_API_URL, transport=transport)
    api_key = "sk-ant-test"
    provider = AnthropicProvider(
        api_key=api_key,
        model="claude-sonnet-4-20250514",
        http_client=client,
    )

    assistant_msg = Message(
        role=Role.ASSISTANT,
        content="Hello",
        thinking=ThinkingContent(text="thoughts"),
        provider_metadata={"anthropic": {"thinking_signature": "sig_123"}},
    )
    messages = [
        Message.system("System A"),
        Message.user("Hi"),
        Message.system("System B"),
        assistant_msg,
    ]

    stream = provider.stream(messages)
    await stream.result()

    await provider.close()

    assert captured.json is not None
    assert captured.json.get("system") == "System A\n\nSystem B"
    assistant_payload = next(
        msg for msg in captured.json.get("messages", []) if msg.get("role") == "assistant"
    )
    content_blocks = assistant_payload.get("content")
    thinking_block = next(block for block in content_blocks if block.get("type") == "thinking")
    assert thinking_block.get("signature") == "sig_123"


@pytest.mark.asyncio
async def test_anthropic_payload_includes_thinking_budget() -> None:
    captured = SimpleNamespace(json=None)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.json = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse_payload([]))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=ANTHROPIC_API_URL, transport=transport)
    base_max_tokens = 2048
    provider = AnthropicProvider(
        api_key="sk-ant-test",
        model="claude-sonnet-4-20250514",
        max_tokens=base_max_tokens,
        http_client=client,
    )

    options = StreamOptions(thinking_level="high")

    stream = provider.stream([Message.user("hi")], options=options)
    await stream.result()

    await provider.close()

    assert captured.json is not None
    thinking = captured.json.get("thinking")
    assert thinking is not None
    assert thinking.get("type") == "enabled"
    assert thinking.get("budget_tokens") > 0
    assert captured.json.get("max_tokens") == base_max_tokens + thinking["budget_tokens"]


@pytest.mark.asyncio
async def test_anthropic_streams_error_on_non_2xx_response() -> None:
    error_message = "bad request"
    http_bad_request = 400

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(http_bad_request, json={"error": {"message": error_message}})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=ANTHROPIC_API_URL, transport=transport)
    provider = AnthropicProvider(
        api_key="sk-ant-test",
        model="claude-sonnet-4-20250514",
        http_client=client,
    )

    stream = provider.stream([Message.user("hi")])
    result = await stream.result()

    await provider.close()

    assert result.stop_reason == "error"
    assert result.error_message is not None
    assert error_message in result.error_message


@pytest.mark.asyncio
async def test_anthropic_retries_on_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = {"count": 0}
    input_tokens = 1
    output_tokens = 1
    http_server_error = 500

    events = [
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": input_tokens, "output_tokens": 0}},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "Hi"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": output_tokens},
        },
    ]

    payload = _sse_payload(events)

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["count"] += 1
        if call_count["count"] == 1:
            return httpx.Response(http_server_error, json={"error": {"message": "transient"}})
        return httpx.Response(200, content=payload)

    async def _no_sleep(delay: float) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr("agent.llm.retry.asyncio.sleep", _no_sleep)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=ANTHROPIC_API_URL, transport=transport)
    provider = AnthropicProvider(
        api_key="sk-ant-test",
        model="claude-sonnet-4-20250514",
        http_client=client,
    )

    stream = provider.stream([Message.user("hi")])
    result = await stream.result()

    await provider.close()

    assert call_count["count"] == 2
    text = next(block for block in result.content if isinstance(block, TextBlock))
    assert text.text == "Hi"
