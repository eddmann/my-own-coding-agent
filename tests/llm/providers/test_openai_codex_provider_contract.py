"""OpenAI Codex provider tests."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from agent.core.message import Message, Role, ToolCall
from agent.llm.events import StreamOptions, ToolCallBlock
from agent.llm.openai_codex import OPENAI_CODEX_BASE_URL, OpenAICodexProvider
from agent.llm.openai_codex.oauth import (
    OAuthCredentials,
    load_oauth_credentials,
    save_oauth_credentials,
)


def _sse_payload(events: list[dict[str, object]]) -> bytes:
    chunks = []
    for event in events:
        chunks.append(f"data: {json.dumps(event)}\n\n".encode())
    chunks.append(b"data: [DONE]\n\n")
    return b"".join(chunks)


def _jwt_for_account(account_id: str) -> str:
    header = {"alg": "none", "typ": "JWT"}
    payload = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
        }
    }

    def _encode(data: dict[str, object]) -> str:
        raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{_encode(header)}.{_encode(payload)}.sig"


@pytest.mark.asyncio
async def test_openai_codex_sends_expected_headers_and_path() -> None:
    captured: dict[str, object] = {}
    token = _jwt_for_account("acct_123")
    events = [
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            },
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_CODEX_BASE_URL, transport=transport)
    provider = OpenAICodexProvider(
        api_key=token,
        model="gpt-5-codex",
        http_client=client,
    )

    stream = provider.stream([Message.user("hi")])
    await stream.result()
    await provider.close()

    headers = captured["headers"]
    assert captured["path"] == "/backend-api/codex/responses"
    assert isinstance(headers, dict)
    assert headers.get("authorization") == f"Bearer {token}"
    assert headers.get("chatgpt-account-id") == "acct_123"
    assert headers.get("openai-beta") == "responses=experimental"
    assert headers.get("originator") == "agent"
    assert isinstance(captured["json"], dict)
    assert captured["json"]["model"] == "gpt-5-codex"
    assert captured["json"]["instructions"] == "You are a helpful coding assistant."


@pytest.mark.asyncio
async def test_openai_codex_uses_oauth_file_and_refreshes_token(tmp_path) -> None:
    expired_token = _jwt_for_account("acct_old")
    fresh_token = _jwt_for_account("acct_new")

    oauth_path = tmp_path / "openai-codex-oauth.json"
    save_oauth_credentials(
        OAuthCredentials(
            refresh="refresh_old",
            access=expired_token,
            expires=0,
            account_id="acct_old",
        ),
        oauth_path,
    )

    async def fake_refresh(refresh_token: str) -> OAuthCredentials:
        assert refresh_token == "refresh_old"
        return OAuthCredentials(
            refresh="refresh_new",
            access=fresh_token,
            expires=9_999_999_999_999,
            account_id="acct_new",
        )

    captured: dict[str, object] = {}
    events = [
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            },
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_CODEX_BASE_URL, transport=transport)
    provider = OpenAICodexProvider(
        api_key="",
        model="gpt-5-codex",
        http_client=client,
        oauth_path=oauth_path,
        refresh_token_fn=fake_refresh,
    )

    stream = provider.stream([Message.user("hi")])
    await stream.result()
    await provider.close()

    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers.get("authorization") == f"Bearer {fresh_token}"
    assert headers.get("chatgpt-account-id") == "acct_new"

    refreshed = load_oauth_credentials(oauth_path)
    assert refreshed is not None
    assert refreshed.refresh == "refresh_new"
    assert refreshed.account_id == "acct_new"


@pytest.mark.asyncio
async def test_openai_codex_errors_when_no_credentials(tmp_path) -> None:
    provider = OpenAICodexProvider(
        api_key="",
        model="gpt-5-codex",
        oauth_path=tmp_path / "missing-oauth.json",
    )

    stream = provider.stream([Message.user("hi")])
    result = await stream.result()

    assert result.stop_reason == "error"
    assert result.error_message is not None
    assert "No OpenAI Codex OAuth credentials configured" in result.error_message


@pytest.mark.asyncio
async def test_openai_codex_payload_replays_reasoning_and_output_ids() -> None:
    captured: dict[str, object] = {}
    token = _jwt_for_account("acct_123")
    events = [
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            },
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_CODEX_BASE_URL, transport=transport)
    provider = OpenAICodexProvider(api_key=token, model="gpt-5-codex", http_client=client)

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
    await provider.close()

    payload = captured["json"]
    assert isinstance(payload, dict)
    items = payload.get("input", [])
    reasoning = next(item for item in items if item.get("type") == "reasoning")
    message = next(item for item in items if item.get("type") == "message")
    assert reasoning.get("summary") == [{"text": reasoning_text}]
    assert message.get("id") == "msg_123"


@pytest.mark.asyncio
async def test_openai_codex_payload_replays_tool_call_id() -> None:
    captured: dict[str, object] = {}
    token = _jwt_for_account("acct_123")
    events = [
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            },
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_CODEX_BASE_URL, transport=transport)
    provider = OpenAICodexProvider(api_key=token, model="gpt-5-codex", http_client=client)

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
    await provider.close()

    payload = captured["json"]
    assert isinstance(payload, dict)
    items = payload.get("input", [])
    function_call = next(item for item in items if item.get("type") == "function_call")
    assert function_call.get("call_id") == call_id
    assert function_call.get("id") == item_id


@pytest.mark.asyncio
async def test_openai_codex_streams_reasoning_and_tool_call_deltas() -> None:
    token = _jwt_for_account("acct_123")
    reasoning_text = "step"
    call_id = "call_1"
    item_id = "fc_1"
    tool_args = '{"path":"/tmp/x"}'
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
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_CODEX_BASE_URL, transport=transport)
    provider = OpenAICodexProvider(api_key=token, model="gpt-5-codex", http_client=client)

    stream = provider.stream([Message.user("hi")])
    streamed_events = [event async for event in stream]
    result = await stream.result()
    await provider.close()

    assert any(event.type == "thinking_delta" for event in streamed_events)
    tool_block = next(block for block in result.content if isinstance(block, ToolCallBlock))
    assert tool_block.arguments == {"path": "/tmp/x"}
    assert tool_block.id == f"{call_id}|{item_id}"


@pytest.mark.asyncio
async def test_openai_codex_streams_error_event() -> None:
    token = _jwt_for_account("acct_123")
    error_message = "Request failed"
    events = [{"type": "error", "error": {"message": error_message}}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_CODEX_BASE_URL, transport=transport)
    provider = OpenAICodexProvider(api_key=token, model="gpt-5-codex", http_client=client)

    stream = provider.stream([Message.user("hi")])
    result = await stream.result()
    await provider.close()

    assert result.stop_reason == "error"
    assert result.error_message is not None
    assert error_message in result.error_message


@pytest.mark.asyncio
async def test_openai_codex_payload_uses_reasoning_effort_and_omits_temperature_for_gpt5() -> None:
    captured: dict[str, object] = {}
    token = _jwt_for_account("acct_123")
    events = [
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            },
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_CODEX_BASE_URL, transport=transport)
    provider = OpenAICodexProvider(api_key=token, model="gpt-5.3-codex", http_client=client)

    stream = provider.stream([Message.user("hi")], options=StreamOptions(thinking_level="xhigh"))
    await stream.result()
    await provider.close()

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload.get("reasoning", {}).get("effort") == "xhigh"
    assert "temperature" not in payload


@pytest.mark.asyncio
async def test_openai_codex_payload_uses_system_messages_as_instructions() -> None:
    captured: dict[str, object] = {}
    token = _jwt_for_account("acct_123")
    events = [
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            },
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse_payload(events))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OPENAI_CODEX_BASE_URL, transport=transport)
    provider = OpenAICodexProvider(api_key=token, model="gpt-5-codex", http_client=client)

    messages = [
        Message.system("System A"),
        Message.system("System B"),
        Message.user("Hi"),
    ]
    stream = provider.stream(messages, options=StreamOptions(thinking_level="high"))
    await stream.result()
    await provider.close()

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload.get("instructions") == "System A\n\nSystem B"

    input_items = payload.get("input", [])
    assert input_items == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Hi",
                }
            ],
        }
    ]


@pytest.mark.asyncio
async def test_openai_codex_provider_list_models_returns_codex_registry_models() -> None:
    provider = OpenAICodexProvider(api_key=_jwt_for_account("acct_123"), model="gpt-5-codex")

    models = await provider.list_models()

    assert "gpt-5-codex" in models
    assert models == sorted(models)


def test_openai_codex_provider_set_model_updates_and_rejects_cross_provider_models() -> None:
    provider = OpenAICodexProvider(api_key=_jwt_for_account("acct_123"), model="gpt-5-codex")

    provider.set_model("gpt-5")

    assert provider.model == "gpt-5"

    with pytest.raises(ValueError, match="not valid for provider"):
        provider.set_model("claude-sonnet-4-5")
