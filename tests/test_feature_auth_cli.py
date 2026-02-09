"""Behavior tests for auth CLI commands."""

from __future__ import annotations

from typer.testing import CliRunner

from agent import cli


def test_auth_login_openai_codex_dispatches(monkeypatch):
    monkeypatch.setattr(
        cli,
        "openai_codex_login_flow",
        lambda prompt, emit: emit("codex login invoked"),
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["auth", "login", "openai-codex"])

    assert result.exit_code == 0
    assert "codex login invoked" in result.stdout


def test_auth_logout_openai_codex_dispatches(monkeypatch):
    monkeypatch.setattr(
        cli,
        "openai_codex_logout_flow",
        lambda emit: emit("codex logout invoked"),
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["auth", "logout", "openai-codex"])

    assert result.exit_code == 0
    assert "codex logout invoked" in result.stdout
