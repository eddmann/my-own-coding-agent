"""Behavior tests for OpenAI Codex OAuth helpers."""

from __future__ import annotations

import base64
import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from agent.llm.openai_codex import oauth


def _jwt_for_account(account_id: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({oauth.JWT_CLAIM_PATH: {"chatgpt_account_id": account_id}}).encode("utf-8")
    ).decode().rstrip("=")
    return f"{header}.{payload}.signature"


def test_build_openai_codex_auth_url_contains_required_parameters():
    auth_url, verifier, state = oauth.build_openai_codex_auth_url()
    parsed = urlparse(auth_url)
    params = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "auth.openai.com"
    assert params["client_id"] == [oauth.CLIENT_ID]
    assert params["redirect_uri"] == [oauth.REDIRECT_URI]
    assert params["scope"] == [oauth.SCOPES]
    assert params["state"] == [state]
    assert params["code_challenge_method"] == ["S256"]
    assert verifier


def test_extract_account_id_from_jwt():
    token = _jwt_for_account("acct_123")

    account_id = oauth.extract_account_id(token)

    assert account_id == "acct_123"


def test_extract_account_id_raises_for_invalid_token():
    with pytest.raises(RuntimeError, match="Failed to extract chatgpt_account_id"):
        oauth.extract_account_id("not-a-jwt")


def test_oauth_credentials_round_trip_and_clear(temp_dir):
    oauth_path = temp_dir / "openai-codex-oauth.json"
    creds = oauth.OAuthCredentials(
        refresh="refresh_1",
        access=_jwt_for_account("acct_1"),
        expires=12345,
        account_id="acct_1",
    )

    oauth.save_oauth_credentials(creds, oauth_path)
    loaded = oauth.load_oauth_credentials(oauth_path)

    assert loaded == creds

    oauth.clear_oauth_credentials(oauth_path)
    assert oauth.load_oauth_credentials(oauth_path) is None


def test_load_oauth_credentials_returns_none_for_invalid_payload(temp_dir):
    oauth_path = temp_dir / "openai-codex-oauth.json"
    oauth_path.write_text(json.dumps({"refresh": "x"}))

    loaded = oauth.load_oauth_credentials(oauth_path)

    assert loaded is None


def test_login_flow_parses_redirect_url_and_saves_credentials():
    captured: dict[str, str] = {}
    emitted: list[str] = []
    creds = oauth.OAuthCredentials(
        refresh="refresh_2",
        access=_jwt_for_account("acct_2"),
        expires=99999,
        account_id="acct_2",
    )

    async def fake_exchange(code: str, verifier: str) -> oauth.OAuthCredentials:
        captured["code"] = code
        captured["verifier"] = verifier
        return creds

    oauth.login_flow(
        lambda _prompt: "http://localhost:1455/auth/callback?code=code-2&state=state-2",
        emitted.append,
        auth_url_builder=lambda: ("https://auth", "verifier-2", "state-2"),
        code_exchanger=fake_exchange,
        credentials_saver=lambda saved: captured.setdefault("saved", saved.account_id),
    )

    assert emitted[0] == "Open this URL in your browser to authorize:"
    assert emitted[1] == "https://auth"
    assert emitted[2] == "Credentials saved to ~/.agent/openai-codex-oauth.json"
    assert captured["code"] == "code-2"
    assert captured["verifier"] == "verifier-2"
    assert captured["saved"] == "acct_2"


def test_login_flow_rejects_state_mismatch():
    with pytest.raises(RuntimeError, match="State mismatch"):
        oauth.login_flow(
            lambda _prompt: "code-3#wrong-state",
            lambda _msg: None,
            auth_url_builder=lambda: ("https://auth", "verifier-3", "expected-state"),
        )


def test_login_flow_rejects_missing_authorization_code():
    with pytest.raises(RuntimeError, match="Missing authorization code"):
        oauth.login_flow(
            lambda _prompt: "",
            lambda _msg: None,
            auth_url_builder=lambda: ("https://auth", "verifier-4", "state-4"),
        )


@pytest.mark.asyncio
async def test_exchange_openai_codex_code_posts_expected_form_data():
    captured: dict[str, object] = {}
    real_async_client = httpx.AsyncClient
    token = _jwt_for_account("acct_3")

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["form"] = parse_qs(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"refresh_token": "refresh_3", "access_token": token, "expires_in": 3600},
        )

    transport = httpx.MockTransport(handler)

    def fake_async_client(*, timeout):
        return real_async_client(transport=transport, timeout=timeout)

    creds = await oauth.exchange_openai_codex_code(
        "code-3",
        "verifier-3",
        client_factory=fake_async_client,
    )

    assert captured["url"] == oauth.TOKEN_URL
    form = captured["form"]
    assert isinstance(form, dict)
    assert form["grant_type"] == ["authorization_code"]
    assert form["code"] == ["code-3"]
    assert form["code_verifier"] == ["verifier-3"]
    assert creds.refresh == "refresh_3"
    assert creds.account_id == "acct_3"


@pytest.mark.asyncio
async def test_refresh_openai_codex_token_posts_expected_form_data():
    captured: dict[str, object] = {}
    real_async_client = httpx.AsyncClient
    token = _jwt_for_account("acct_4")

    def handler(request: httpx.Request) -> httpx.Response:
        captured["form"] = parse_qs(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"refresh_token": "refresh_4", "access_token": token, "expires_in": 3600},
        )

    transport = httpx.MockTransport(handler)

    def fake_async_client(*, timeout):
        return real_async_client(transport=transport, timeout=timeout)

    creds = await oauth.refresh_openai_codex_token(
        "refresh-old",
        client_factory=fake_async_client,
    )

    form = captured["form"]
    assert isinstance(form, dict)
    assert form["grant_type"] == ["refresh_token"]
    assert form["refresh_token"] == ["refresh-old"]
    assert creds.refresh == "refresh_4"
    assert creds.account_id == "acct_4"


@pytest.mark.asyncio
async def test_exchange_openai_codex_code_raises_on_http_error():
    real_async_client = httpx.AsyncClient

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    transport = httpx.MockTransport(handler)

    def fake_async_client(*, timeout):
        return real_async_client(transport=transport, timeout=timeout)

    with pytest.raises(RuntimeError, match="Token exchange failed"):
        await oauth.exchange_openai_codex_code("code", "verifier", client_factory=fake_async_client)
