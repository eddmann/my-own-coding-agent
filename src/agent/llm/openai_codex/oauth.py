"""OpenAI Codex OAuth (ChatGPT Plus/Pro) helpers."""

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
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPES = "openid profile email offline_access"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
JWT_CLAIM_PATH = "https://api.openai.com/auth"


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _decode_base64url(value: str) -> bytes:
    padding = "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode(value + padding)


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE verifier + challenge."""
    verifier = _base64url(secrets.token_bytes(32))
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = _base64url(digest)
    return verifier, challenge


def _create_state() -> str:
    return secrets.token_hex(16)


def extract_account_id(access_token: str) -> str:
    """Extract ChatGPT account id from JWT access token."""
    try:
        parts = access_token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT shape")
        payload = json.loads(_decode_base64url(parts[1]))
        account_id = payload.get(JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
        if not isinstance(account_id, str) or not account_id:
            raise ValueError("Missing account id")
        return account_id
    except Exception as exc:
        raise RuntimeError("Failed to extract chatgpt_account_id from access token") from exc


@dataclass(slots=True)
class OAuthCredentials:
    refresh: str
    access: str
    expires: int
    account_id: str

    @classmethod
    def from_token_response(cls, data: dict[str, Any]) -> OAuthCredentials:
        access = str(data.get("access_token", ""))
        refresh = str(data.get("refresh_token", ""))
        expires_in = int(data.get("expires_in") or 0)
        expires_at = int(time.time() * 1000) + (expires_in * 1000) - (5 * 60 * 1000)
        return cls(
            refresh=refresh,
            access=access,
            expires=expires_at,
            account_id=extract_account_id(access),
        )


def default_oauth_path() -> Path:
    return Path.home() / ".agent" / "openai-codex-oauth.json"


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
        account_id = raw.get("account_id")
        if (
            not isinstance(refresh, str)
            or not isinstance(access, str)
            or not isinstance(expires, (int, float))
            or not isinstance(account_id, str)
        ):
            return None
        return OAuthCredentials(
            refresh=refresh,
            access=access,
            expires=int(expires),
            account_id=account_id,
        )
    except Exception:
        return None


def save_oauth_credentials(creds: OAuthCredentials, path: Path | None = None) -> None:
    target = path or default_oauth_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "refresh": creds.refresh,
        "access": creds.access,
        "expires": creds.expires,
        "account_id": creds.account_id,
    }
    target.write_text(json.dumps(payload, indent=2))
    with contextlib.suppress(Exception):
        os.chmod(target, 0o600)


def clear_oauth_credentials(path: Path | None = None) -> None:
    target = path or default_oauth_path()
    if target.exists():
        target.unlink()


def build_openai_codex_auth_url() -> tuple[str, str, str]:
    """Return (auth_url, verifier, state)."""
    verifier, challenge = generate_pkce()
    state = _create_state()
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "agent",
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}", verifier, state


def _parse_code_and_state(value: str) -> tuple[str, str | None]:
    raw = value.strip()
    if not raw:
        return "", None

    try:
        parsed = urlparse(raw)
        if parsed.scheme and parsed.netloc:
            query = parse_qs(parsed.query)
            code = (query.get("code") or [""])[0]
            state_values = query.get("state")
            state = state_values[0] if state_values else None
            return code, state
    except Exception:
        pass

    if "#" in raw:
        code, state = raw.split("#", 1)
        return code.strip(), state.strip()

    if raw.startswith("code="):
        pairs = parse_qs(raw)
        code = (pairs.get("code") or [""])[0].strip()
        state_values = pairs.get("state")
        state = state_values[0] if state_values else None
        return code, state

    return raw, None


async def exchange_openai_codex_code(code: str, verifier: str) -> OAuthCredentials:
    """Exchange authorization code for tokens."""
    payload = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "code_verifier": verifier,
        "redirect_uri": REDIRECT_URI,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.post(TOKEN_URL, data=payload)
    if response.status_code >= 400:
        raise RuntimeError(f"Token exchange failed: {response.text}")
    data = json.loads(response.text)
    return OAuthCredentials.from_token_response(data)


async def refresh_openai_codex_token(refresh_token: str) -> OAuthCredentials:
    """Refresh an OpenAI Codex OAuth token."""
    payload = {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.post(TOKEN_URL, data=payload)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI Codex token refresh failed: {response.text}")
    data = json.loads(response.text)
    return OAuthCredentials.from_token_response(data)


def login_flow(prompt: Callable[[str], str], emit: Callable[[str], None]) -> None:
    """CLI login flow helper."""
    auth_url, verifier, expected_state = build_openai_codex_auth_url()
    emit("Open this URL in your browser to authorize:")
    emit(auth_url)
    raw = prompt("Paste the redirect URL (or code#state)")
    code, state = _parse_code_and_state(raw)
    if not code:
        raise RuntimeError("Missing authorization code")
    if state is not None and state != expected_state:
        raise RuntimeError("State mismatch")
    creds = asyncio.run(exchange_openai_codex_code(code, verifier))
    save_oauth_credentials(creds)
    emit("Credentials saved to ~/.agent/openai-codex-oauth.json")


def logout_flow(emit: Callable[[str], None]) -> None:
    """CLI logout flow helper."""
    clear_oauth_credentials()
    emit("Logged out of openai-codex")
