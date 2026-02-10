"""OpenAI-compatible provider tests using MockTransport (behavioral)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.core.message import Message
from agent.llm.events import StreamOptions, TextBlock, ToolCallBlock
from agent.llm.openai_compat import OpenAICompatibleProvider
from agent.llm.retry import RetryConfig


def _sse_payload(events: list[dict[str, object]]) -> bytes:
    chunks = []
    for event in events:
        chunks.append(f"data: {json.dumps(event)}\n\n".encode())
    chunks.append(b"data: [DONE]\n\n")
    return b"".join(chunks)


@pytest.mark.asyncio
async def test_openai_compatible_payload_and_streams_text_and_tool_call() -> None:
    captured = SimpleNamespace(json=None)
    prompt_tokens = 1
    completion_tokens = 2
    tool_id = "tool_call_123456"
    tool_id_max_length = 9
    truncated_id = tool_id[:tool_id_max_length]
    tool_args = '{"path":"/tmp/x"}'

    events = [
        {"choices": [{"delta": {"content": "Hi"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": tool_id,
                                "function": {"name": "read", "arguments": tool_args},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.json = json.loads(request.content.decode("utf-8"))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url="https://mistral.ai", transport=transport)
    provider = OpenAICompatibleProvider(
        base_url="https://mistral.ai",
        api_key="sk-test",
        model="mistral-model",
        http_client=client,
    )

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

    stream = provider.stream([Message.user("hi")], tools=tools)
    result = await stream.result()

    await provider.client.aclose()

    assert captured.json is not None
    assert captured.json.get("model") == provider.model
    assert "max_tokens" in captured.json
    assert "max_completion_tokens" not in captured.json
    assert captured.json.get("tool_choice") == "auto"

    text = next(block for block in result.content if isinstance(block, TextBlock))
    tool = next(block for block in result.content if isinstance(block, ToolCallBlock))

    assert text.text == "Hi"
    assert tool.id == truncated_id
    assert tool.arguments == {"path": "/tmp/x"}
    assert result.usage.input == prompt_tokens
    assert result.usage.output == completion_tokens


@pytest.mark.asyncio
async def test_openai_compatible_streams_error_on_non_2xx_response() -> None:
    error_message = "bad request"
    http_bad_request = 400

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(http_bad_request, json={"error": {"message": error_message}})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url="https://example.com", transport=transport)
    provider = OpenAICompatibleProvider(
        base_url="https://example.com",
        api_key="sk-test",
        model="gpt-4o",
        http_client=client,
    )

    stream = provider.stream([Message.user("hi")])
    result = await stream.result()

    await provider.client.aclose()

    assert result.stop_reason == "error"
    assert result.error_message is not None
    assert error_message in result.error_message


@pytest.mark.asyncio
async def test_openai_compatible_retries_on_transient_error() -> None:
    call_count = {"count": 0}
    prompt_tokens = 1
    completion_tokens = 1
    http_server_error = 500

    events = [
        {"choices": [{"delta": {"content": "Hi"}}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["count"] += 1
        if call_count["count"] == 1:
            return httpx.Response(http_server_error, json={"error": {"message": "transient"}})
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url="https://example.com", transport=transport)
    provider = OpenAICompatibleProvider(
        base_url="https://example.com",
        api_key="sk-test",
        model="gpt-4o",
        http_client=client,
        retry_config=RetryConfig(max_retries=1, initial_delay=0.0, max_delay=0.0, jitter=False),
    )

    stream = provider.stream([Message.user("hi")])
    result = await stream.result()

    await provider.client.aclose()

    assert call_count["count"] == 2
    text = next(block for block in result.content if isinstance(block, TextBlock))
    assert text.text == "Hi"


@pytest.mark.asyncio
async def test_openai_compatible_cancels_before_request() -> None:
    cancel_event = asyncio.Event()
    cancel_event.set()

    provider = OpenAICompatibleProvider(
        base_url="https://example.com",
        api_key="sk-test",
        model="gpt-4o",
    )

    options = StreamOptions(cancel_event=cancel_event)

    stream = provider.stream([Message.user("hi")], options=options)
    result = await stream.result()

    assert result.stop_reason == "aborted"


@pytest.mark.asyncio
async def test_openai_compatible_list_models_fetches_models_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={"data": [{"id": "model-b"}, {"id": "model-a"}]},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url="https://example.com", transport=transport)
    provider = OpenAICompatibleProvider(
        base_url="https://example.com",
        api_key="sk-test",
        model="gpt-4o",
        http_client=client,
    )

    models = await provider.list_models()
    await provider.client.aclose()

    assert models == ["model-a", "model-b"]


def test_openai_compatible_set_model_allows_cross_family_model_ids() -> None:
    provider = OpenAICompatibleProvider(
        base_url="https://example.com",
        api_key="sk-test",
        model="gpt-4o",
    )

    provider.set_model("claude-sonnet-4-5")

    assert provider.model == "claude-sonnet-4-5"
