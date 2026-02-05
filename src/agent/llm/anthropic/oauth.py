"""Anthropic OAuth (Claude Pro/Max) helpers."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
SCOPES = "org:create_api_key user:profile user:inference"

# Base64 encoded Anthropic OAuth client ID (mirrors pi)
_CLIENT_ID_B64 = "OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl"


def _client_id() -> str:
    return base64.b64decode(_CLIENT_ID_B64).decode("utf-8")


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE verifier + challenge."""
    verifier = _base64url(secrets.token_bytes(32))
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = _base64url(digest)
    return verifier, challenge


@dataclass(slots=True)
class OAuthCredentials:
    refresh: str
    access: str
    expires: int

    @classmethod
    def from_token_response(cls, data: dict[str, Any]) -> OAuthCredentials:
        expires_in = int(data.get("expires_in") or 0)
        # 5 min buffer
        expires_at = int(time.time() * 1000) + (expires_in * 1000) - (5 * 60 * 1000)
        return cls(
            refresh=str(data.get("refresh_token", "")),
            access=str(data.get("access_token", "")),
            expires=expires_at,
        )


def default_oauth_path() -> Path:
    return Path.home() / ".agent" / "anthropic-oauth.json"


def load_oauth_credentials(path: Path | None = None) -> OAuthCredentials | None:
    target = path or default_oauth_path()
    if not target.exists():
        return None
    try:
        raw = json.loads(target.read_text())
        if not isinstance(raw, dict):
            return None
        refresh = raw.get("refresh")
        access = raw.get("access")
        expires = raw.get("expires")
        if (
            not isinstance(refresh, str)
            or not isinstance(access, str)
            or not isinstance(expires, (int, float))
        ):
            return None
        return OAuthCredentials(refresh=refresh, access=access, expires=int(expires))
    except Exception:
        return None


def save_oauth_credentials(creds: OAuthCredentials, path: Path | None = None) -> None:
    target = path or default_oauth_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"refresh": creds.refresh, "access": creds.access, "expires": creds.expires}
    target.write_text(json.dumps(payload, indent=2))
    with contextlib.suppress(Exception):
        os.chmod(target, 0o600)


def clear_oauth_credentials(path: Path | None = None) -> None:
    target = path or default_oauth_path()
    if target.exists():
        target.unlink()


def build_anthropic_auth_url() -> tuple[str, str]:
    """Return (auth_url, verifier). Caller persists verifier for token exchange."""
    verifier, challenge = generate_pkce()
    params = {
        "code": "true",
        "client_id": _client_id(),
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}", verifier


def _parse_code_state(value: str) -> tuple[str, str]:
    parts = value.strip().split("#", 1)
    code = parts[0].strip()
    state = parts[1].strip() if len(parts) > 1 else ""
    return code, state


async def exchange_anthropic_code(code_state: str, verifier: str) -> OAuthCredentials:
    """Exchange authorization code for tokens."""
    code, state = _parse_code_state(code_state)
    payload = {
        "grant_type": "authorization_code",
        "client_id": _client_id(),
        "code": code,
        "state": state,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.post(TOKEN_URL, json=payload)
    if response.status_code >= 400:
        raise RuntimeError(f"Token exchange failed: {response.text}")
    data = json.loads(response.text)
    return OAuthCredentials.from_token_response(data)


async def refresh_anthropic_token(refresh_token: str) -> OAuthCredentials:
    """Refresh an Anthropic OAuth token."""
    payload = {
        "grant_type": "refresh_token",
        "client_id": _client_id(),
        "refresh_token": refresh_token,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.post(TOKEN_URL, json=payload)
    if response.status_code >= 400:
        raise RuntimeError(f"Anthropic token refresh failed: {response.text}")
    data = json.loads(response.text)
    return OAuthCredentials.from_token_response(data)


def login_flow(prompt: Callable[[str], str], emit: Callable[[str], None]) -> None:
    """CLI login flow helper."""
    auth_url, verifier = build_anthropic_auth_url()
    emit("Open this URL in your browser to authorize:")
    emit(auth_url)
    code_state = prompt("Paste the authorization code (code#state)")
    creds = asyncio.run(exchange_anthropic_code(code_state, verifier))
    save_oauth_credentials(creds)
    emit("Credentials saved to ~/.agent/anthropic-oauth.json")


def logout_flow(emit: Callable[[str], None]) -> None:
    """CLI logout flow helper."""
    clear_oauth_credentials()
    emit("Logged out of anthropic")


def status_flow(emit: Callable[[str], None]) -> None:
    """CLI status flow helper."""
    creds = load_oauth_credentials()
    if not creds:
        emit("No OAuth credentials found")
        return
    emit("anthropic: oauth")
