"""OpenAI provider tests using MockTransport (behavioral)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.core.message import Message, Role, ToolCall
from agent.llm.events import StreamOptions, TextBlock, ToolCallBlock
from agent.llm.openai import OpenAIProvider

OPENAI_API_BASE_URL = "https://api.openai.com"


def _sse_payload(events: list[dict[str, object]]) -> bytes:
    chunks = []
    for event in events:
        chunks.append(f"data: {json.dumps(event)}\n\n".encode())
    chunks.append(b"data: [DONE]\n\n")
    return b"".join(chunks)


@pytest.mark.asyncio
async def test_openai_chat_payload_includes_temperature_and_streams_text() -> None:
    captured = SimpleNamespace(json=None)
    prompt_tokens = 1
    completion_tokens = 2
    cached_tokens = 0
    reasoning_tokens = 0
    events = [
        {"choices": [{"delta": {"content": "Hi"}}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.json = json.loads(request.content.decode("utf-8"))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_API_BASE_URL, transport=transport)
    provider = OpenAIProvider(
        api_key="sk-test",
        model="gpt-4o",
        http_client=client,
    )

    stream = provider.stream([Message.user("hi")])
    result = await stream.result()

    await provider.client.aclose()

    assert captured.json is not None
    assert "temperature" in captured.json
    assert "max_tokens" in captured.json
    assert "max_completion_tokens" not in captured.json

    text = next(block for block in result.content if isinstance(block, TextBlock))
    assert text.text == "Hi"
    assert result.usage.input == prompt_tokens
    assert result.usage.output == completion_tokens


@pytest.mark.asyncio
async def test_openai_chat_payload_omits_temperature_for_gpt5() -> None:
    captured = SimpleNamespace(json=None)
    prompt_tokens = 1
    completion_tokens = 1
    cached_tokens = 0
    reasoning_tokens = 0
    events = [
        {"choices": [{"delta": {"content": "Yo"}}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.json = json.loads(request.content.decode("utf-8"))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_API_BASE_URL, transport=transport)
    provider = OpenAIProvider(
        api_key="sk-test",
        model="gpt-5",
        http_client=client,
    )

    stream = provider.stream([Message.user("hi")])
    await stream.result()

    await provider.client.aclose()

    assert captured.json is not None
    assert "temperature" not in captured.json
    assert "max_completion_tokens" in captured.json


@pytest.mark.asyncio
async def test_openai_responses_payload_and_stream() -> None:
    captured = SimpleNamespace(json=None)
    input_tokens = 1
    output_tokens = 2
    cached_tokens = 0
    message_id = "msg_1"
    response_status = "completed"
    events = [
        {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": message_id},
        },
        {"type": "response.output_text.delta", "delta": "Hi"},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": message_id,
                "content": [{"type": "output_text", "text": "Hi"}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "status": response_status,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_tokens_details": {"cached_tokens": cached_tokens},
                },
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.json = json.loads(request.content.decode("utf-8"))
        assert request.url.path == "/v1/responses"
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_API_BASE_URL, transport=transport)
    provider = OpenAIProvider(
        api_key="sk-test",
        model="gpt-5-codex",
        http_client=client,
    )

    stream = provider.stream([Message.user("hi")])
    result = await stream.result()

    await provider.client.aclose()

    assert captured.json is not None
    assert "temperature" not in captured.json
    assert "max_output_tokens" in captured.json
    assert "input" in captured.json

    text = next(block for block in result.content if isinstance(block, TextBlock))
    assert text.text == "Hi"
    assert result.usage.input == input_tokens
    assert result.usage.output == output_tokens


@pytest.mark.asyncio
async def test_openai_responses_payload_uses_xhigh_for_gpt53_codex() -> None:
    captured = SimpleNamespace(json=None)
    zero_tokens = 0
    events = [
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": zero_tokens,
                    "output_tokens": zero_tokens,
                    "input_tokens_details": {"cached_tokens": zero_tokens},
                },
            },
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.json = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_API_BASE_URL, transport=transport)
    provider = OpenAIProvider(
        api_key="sk-test",
        model="gpt-5.3-codex",
        http_client=client,
    )

    stream = provider.stream([Message.user("hi")], options=StreamOptions(thinking_level="xhigh"))
    await stream.result()
    await provider.client.aclose()

    assert captured.json is not None
    assert captured.json.get("reasoning", {}).get("effort") == "xhigh"


@pytest.mark.asyncio
async def test_openai_responses_payload_clamps_xhigh_for_non_xhigh_model() -> None:
    captured = SimpleNamespace(json=None)
    zero_tokens = 0
    events = [
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": zero_tokens,
                    "output_tokens": zero_tokens,
                    "input_tokens_details": {"cached_tokens": zero_tokens},
                },
            },
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.json = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_API_BASE_URL, transport=transport)
    provider = OpenAIProvider(
        api_key="sk-test",
        model="gpt-5-codex",
        http_client=client,
    )

    stream = provider.stream([Message.user("hi")], options=StreamOptions(thinking_level="xhigh"))
    await stream.result()
    await provider.client.aclose()

    assert captured.json is not None
    assert captured.json.get("reasoning", {}).get("effort") == "high"


@pytest.mark.asyncio
async def test_openai_responses_payload_replays_reasoning_and_output_ids() -> None:
    captured = SimpleNamespace(json=None)
    response_status = "completed"
    cached_tokens = 0
    zero_tokens = 0
    events = [
        {
            "type": "response.completed",
            "response": {
                "status": response_status,
                "usage": {
                    "input_tokens": zero_tokens,
                    "output_tokens": zero_tokens,
                    "input_tokens_details": {"cached_tokens": cached_tokens},
                },
            },
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.json = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_API_BASE_URL, transport=transport)
    provider = OpenAIProvider(
        api_key="sk-test",
        model="gpt-5-codex",
        http_client=client,
    )

    reasoning_text = "r"
    reasoning_item = {"type": "reasoning", "summary": [{"text": reasoning_text}]}
    assistant_msg = Message(
        role=Role.ASSISTANT,
        content="Hello",
        provider_metadata={
            "openai_responses": {
                "output_item_id": "msg_123",
                "reasoning_item": json.dumps(reasoning_item),
            }
        },
    )
    messages = [Message.system("System"), Message.user("Hi"), assistant_msg]

    stream = provider.stream(messages)
    await stream.result()

    await provider.client.aclose()

    assert captured.json is not None
    items = captured.json.get("input", [])
    reasoning = next(item for item in items if item.get("type") == "reasoning")
    message = next(item for item in items if item.get("type") == "message")
    assert reasoning.get("summary") == [{"text": reasoning_text}]
    assert message.get("id") == "msg_123"


@pytest.mark.asyncio
async def test_openai_responses_payload_replays_tool_call_id() -> None:
    captured = SimpleNamespace(json=None)
    response_status = "completed"
    cached_tokens = 0
    zero_tokens = 0
    events = [
        {
            "type": "response.completed",
            "response": {
                "status": response_status,
                "usage": {
                    "input_tokens": zero_tokens,
                    "output_tokens": zero_tokens,
                    "input_tokens_details": {"cached_tokens": cached_tokens},
                },
            },
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.json = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_API_BASE_URL, transport=transport)
    provider = OpenAIProvider(
        api_key="sk-test",
        model="gpt-5-codex",
        http_client=client,
    )

    call_id = "call_123"
    item_id = "fc_abc"
    tool_call = ToolCall(
        id=f"{call_id}|{item_id}",
        name="read",
        arguments={"path": "/tmp/x"},
    )
    assistant_msg = Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[tool_call],
        model="gpt-5-codex",
    )
    messages = [Message.system("System"), Message.user("Hi"), assistant_msg]

    stream = provider.stream(messages)
    await stream.result()

    await provider.client.aclose()

    assert captured.json is not None
    items = captured.json.get("input", [])
    function_call = next(item for item in items if item.get("type") == "function_call")
    assert function_call.get("call_id") == call_id
    assert function_call.get("id") == item_id


@pytest.mark.asyncio
async def test_openai_responses_streams_reasoning_and_tool_call_deltas() -> None:
    captured = SimpleNamespace(json=None)
    reasoning_text = "step"
    call_id = "call_1"
    item_id = "fc_1"
    tool_args = '{"path":"/tmp/x"}'
    cached_tokens = 0
    zero_tokens = 0

    events = [
        {
            "type": "response.output_item.added",
            "item": {"type": "reasoning", "id": "reasoning_1", "summary": []},
        },
        {"type": "response.reasoning_summary_part.added", "part": {"text": ""}},
        {"type": "response.reasoning_summary_text.delta", "delta": reasoning_text},
        {
            "type": "response.output_item.done",
            "item": {"type": "reasoning", "summary": [{"text": reasoning_text}]},
        },
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "call_id": call_id,
                "id": item_id,
                "name": "read",
                "arguments": "",
            },
        },
        {"type": "response.function_call_arguments.delta", "delta": tool_args},
        {"type": "response.function_call_arguments.done", "arguments": tool_args},
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": call_id, "arguments": tool_args},
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": zero_tokens,
                    "output_tokens": zero_tokens,
                    "input_tokens_details": {"cached_tokens": cached_tokens},
                },
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.json = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_API_BASE_URL, transport=transport)
    provider = OpenAIProvider(
        api_key="sk-test",
        model="gpt-5-codex",
        http_client=client,
    )

    stream = provider.stream([Message.user("hi")])
    streamed_events = [event async for event in stream]
    result = await stream.result()

    await provider.client.aclose()

    assert any(event.type == "thinking_delta" for event in streamed_events)
    tool_block = next(block for block in result.content if isinstance(block, ToolCallBlock))
    assert tool_block.arguments == {"path": "/tmp/x"}
    assert tool_block.id == f"{call_id}|{item_id}"


@pytest.mark.asyncio
async def test_openai_responses_streams_error_event() -> None:
    error_message = "Request failed"
    events = [{"type": "error", "error": {"message": error_message}}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_API_BASE_URL, transport=transport)
    provider = OpenAIProvider(
        api_key="sk-test",
        model="gpt-5-codex",
        http_client=client,
    )

    stream = provider.stream([Message.user("hi")])
    result = await stream.result()

    await provider.client.aclose()

    assert result.stop_reason == "error"
    assert result.error_message is not None
    assert error_message in result.error_message


@pytest.mark.asyncio
async def test_openai_cancels_before_request() -> None:
    cancel_event = asyncio.Event()
    cancel_event.set()

    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")

    options = StreamOptions(cancel_event=cancel_event)

    stream = provider.stream([Message.user("hi")], options=options)
    result = await stream.result()

    assert result.stop_reason == "aborted"


@pytest.mark.asyncio
async def test_openai_provider_list_models_returns_registry_models() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")

    models = await provider.list_models()

    assert "gpt-4o" in models
    assert models == sorted(models)


def test_openai_provider_set_model_updates_and_rejects_cross_provider_models() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")

    provider.set_model("gpt-5")

    assert provider.model == "gpt-5"

    with pytest.raises(ValueError, match="not valid for provider"):
        provider.set_model("claude-sonnet-4-5")
