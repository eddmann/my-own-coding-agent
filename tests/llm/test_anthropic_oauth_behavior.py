"""Behavior tests for Anthropic OAuth helpers."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from agent.llm.anthropic import oauth as oauth


def test_build_anthropic_auth_url_contains_required_parameters():
    auth_url, verifier = oauth.build_anthropic_auth_url()
    parsed = urlparse(auth_url)
    params = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "claude.ai"
    assert params["redirect_uri"] == [oauth.REDIRECT_URI]
    assert params["scope"] == [oauth.SCOPES]
    assert params["code_challenge_method"] == ["S256"]
    assert params["state"] == [verifier]


def test_oauth_credentials_round_trip_and_clear(temp_dir):
    oauth_path = temp_dir / "anthropic-oauth.json"
    creds = oauth.OAuthCredentials(refresh="r1", access="a1", expires=12345)

    oauth.save_oauth_credentials(creds, oauth_path)
    loaded = oauth.load_oauth_credentials(oauth_path)

    assert loaded == creds

    oauth.clear_oauth_credentials(oauth_path)
    assert oauth.load_oauth_credentials(oauth_path) is None


def test_load_oauth_credentials_returns_none_for_invalid_payload(temp_dir):
    oauth_path = temp_dir / "anthropic-oauth.json"
    oauth_path.write_text(json.dumps({"refresh": "r"}))

    loaded = oauth.load_oauth_credentials(oauth_path)

    assert loaded is None


def test_status_flow_reports_missing_credentials():
    emitted: list[str] = []

    oauth.status_flow(emitted.append, credentials_loader=lambda: None)

    assert emitted == ["No OAuth credentials found"]


def test_login_flow_exchanges_code_and_saves_credentials():
    captured: dict[str, str] = {}
    emitted: list[str] = []
    creds = oauth.OAuthCredentials(refresh="r2", access="a2", expires=99999)

    async def fake_exchange(code_state: str, verifier: str) -> oauth.OAuthCredentials:
        captured["code_state"] = code_state
        captured["verifier"] = verifier
        return creds

    oauth.login_flow(
        lambda _prompt: "code-1#state-1",
        emitted.append,
        auth_url_builder=lambda: ("https://auth", "verifier-1"),
        code_exchanger=fake_exchange,
        credentials_saver=lambda saved: captured.setdefault("saved", saved.access),
    )

    assert emitted[0] == "Open this URL in your browser to authorize:"
    assert emitted[1] == "https://auth"
    assert emitted[2] == "Credentials saved to ~/.agent/anthropic-oauth.json"
    assert captured["code_state"] == "code-1#state-1"
    assert captured["verifier"] == "verifier-1"
    assert captured["saved"] == "a2"


@pytest.mark.asyncio
async def test_exchange_anthropic_code_posts_expected_payload():
    captured: dict[str, object] = {}
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"refresh_token": "refresh_1", "access_token": "access_1", "expires_in": 3600},
        )

    transport = httpx.MockTransport(handler)

    def fake_async_client(*, timeout):
        return real_async_client(transport=transport, timeout=timeout)

    creds = await oauth.exchange_anthropic_code(
        "auth-code#state-9",
        "verifier-9",
        client_factory=fake_async_client,
    )

    assert captured["url"] == oauth.TOKEN_URL
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["grant_type"] == "authorization_code"
    assert payload["code"] == "auth-code"
    assert payload["state"] == "state-9"
    assert payload["code_verifier"] == "verifier-9"
    assert creds.refresh == "refresh_1"
    assert creds.access == "access_1"


@pytest.mark.asyncio
async def test_exchange_anthropic_code_raises_on_http_error():
    real_async_client = httpx.AsyncClient

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="denied")

    transport = httpx.MockTransport(handler)

    def fake_async_client(*, timeout):
        return real_async_client(transport=transport, timeout=timeout)

    with pytest.raises(RuntimeError, match="Token exchange failed"):
        await oauth.exchange_anthropic_code("code", "verifier", client_factory=fake_async_client)


@pytest.mark.asyncio
async def test_refresh_anthropic_token_posts_expected_payload():
    captured: dict[str, object] = {}
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"refresh_token": "refresh_2", "access_token": "access_2", "expires_in": 3600},
        )

    transport = httpx.MockTransport(handler)

    def fake_async_client(*, timeout):
        return real_async_client(transport=transport, timeout=timeout)

    creds = await oauth.refresh_anthropic_token("refresh-old", client_factory=fake_async_client)

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["grant_type"] == "refresh_token"
    assert payload["refresh_token"] == "refresh-old"
    assert creds.refresh == "refresh_2"
    assert creds.access == "access_2"
