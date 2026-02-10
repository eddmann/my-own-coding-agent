"""Behavior tests for auth CLI commands via CLI runner boundary."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from agent import cli


def test_auth_login_rejects_unsupported_provider():
    runner = CliRunner()
    result = runner.invoke(cli.app, ["auth", "login", "unsupported"])

    assert result.exit_code == 1
    assert "Unsupported provider" in result.stderr


def test_auth_logout_rejects_unsupported_provider():
    runner = CliRunner()
    result = runner.invoke(cli.app, ["auth", "logout", "unsupported"])

    assert result.exit_code == 1
    assert "Unsupported provider" in result.stderr


def test_auth_status_reports_when_no_credentials(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["auth", "status"])

    assert result.exit_code == 0
    assert "No OAuth credentials found" in result.stdout


def test_auth_status_reports_each_available_credential(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    oauth_dir = home / ".agent"
    oauth_dir.mkdir()

    (oauth_dir / "anthropic-oauth.json").write_text(
        json.dumps({"refresh": "r", "access": "a", "expires": 123456})
    )
    (oauth_dir / "openai-codex-oauth.json").write_text(
        json.dumps(
            {
                "refresh": "r2",
                "access": "a2",
                "expires": 654321,
                "account_id": "acct_123",
            }
        )
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["auth", "status"])

    assert result.exit_code == 0
    assert "anthropic: oauth" in result.stdout
    assert "openai-codex: oauth" in result.stdout


def test_auth_login_dispatches_supported_providers(monkeypatch):
    calls: list[str] = []

    def fake_anthropic_login(prompt_fn, echo_fn):
        calls.append("anthropic")

    def fake_openai_codex_login(prompt_fn, echo_fn):
        calls.append("openai-codex")

    monkeypatch.setattr(cli, "anthropic_login_flow", fake_anthropic_login)
    monkeypatch.setattr(cli, "openai_codex_login_flow", fake_openai_codex_login)

    runner = CliRunner()
    anthropic = runner.invoke(cli.app, ["auth", "login", "anthropic"])
    codex = runner.invoke(cli.app, ["auth", "login", "openai-codex"])

    assert anthropic.exit_code == 0
    assert codex.exit_code == 0
    assert calls == ["anthropic", "openai-codex"]


def test_auth_logout_dispatches_supported_providers(monkeypatch):
    calls: list[str] = []

    def fake_anthropic_logout(echo_fn):
        calls.append("anthropic")

    def fake_openai_codex_logout(echo_fn):
        calls.append("openai-codex")

    monkeypatch.setattr(cli, "anthropic_logout_flow", fake_anthropic_logout)
    monkeypatch.setattr(cli, "openai_codex_logout_flow", fake_openai_codex_logout)

    runner = CliRunner()
    anthropic = runner.invoke(cli.app, ["auth", "logout", "anthropic"])
    codex = runner.invoke(cli.app, ["auth", "logout", "openai-codex"])

    assert anthropic.exit_code == 0
    assert codex.exit_code == 0
    assert calls == ["anthropic", "openai-codex"]
