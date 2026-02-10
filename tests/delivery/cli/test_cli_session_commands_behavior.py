"""Behavior tests for CLI session commands via CLI runner boundary."""

from __future__ import annotations

import textwrap

from typer.testing import CliRunner

from agent import cli
from agent.core.message import Message, Role
from agent.core.session import Session


def test_cli_fork_errors_when_no_sessions_exist(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["fork"])

    assert result.exit_code == 1
    assert "No previous session found" in result.stderr


def test_cli_fork_errors_when_explicit_session_path_missing(temp_dir):
    runner = CliRunner()
    result = runner.invoke(cli.app, ["fork", "--session", str(temp_dir / "missing.jsonl")])

    assert result.exit_code == 1
    assert "Session file not found" in result.stderr


def test_cli_tree_errors_when_no_sessions_exist(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["tree"])

    assert result.exit_code == 1
    assert "No previous session found" in result.stderr


def test_cli_tree_errors_when_explicit_session_path_missing(temp_dir):
    runner = CliRunner()
    result = runner.invoke(cli.app, ["tree", "--session", str(temp_dir / "missing.jsonl")])

    assert result.exit_code == 1
    assert "Session file not found" in result.stderr


def test_cli_fork_errors_when_from_message_cannot_be_resolved(temp_dir):
    source = Session.new(temp_dir)
    source.append(Message.user("first"))

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["fork", "--session", str(source.path), "--from", "does-not-exist"],
    )

    assert result.exit_code == 1
    assert "Could not resolve message" in result.stderr


def test_cli_tree_errors_when_target_entry_cannot_be_resolved(temp_dir):
    session = Session.new(temp_dir)
    session.append(Message.user("first"))

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["tree", "--session", str(session.path), "--to", "does-not-exist"],
    )

    assert result.exit_code == 1
    assert "Could not resolve message" in result.stderr


def test_cli_fork_starts_tui_with_forked_session(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    source = Session.new(temp_dir)
    source.append(Message.user("first"))
    source.append(Message.assistant("second"))

    seen: dict[str, object] = {}
    sentinel_provider = object()

    def fake_create_provider(config):
        return sentinel_provider

    def fake_run_tui(config, session, llm_provider):
        seen["session"] = session
        seen["provider"] = llm_provider

    monkeypatch.setattr(cli, "_create_llm_provider", fake_create_provider)
    monkeypatch.setattr(cli, "_run_tui", fake_run_tui)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["fork", "--session", str(source.path), "--from", "assistant"],
    )

    assert result.exit_code == 0
    assert "Forked session:" in result.stdout
    assert seen["provider"] is sentinel_provider

    forked = seen["session"]
    assert isinstance(forked, Session)
    assert forked.metadata.parent_session_id == source.metadata.id
    assert forked.messages
    assert forked.messages[-1].role == Role.ASSISTANT


def test_cli_tree_sets_leaf_and_starts_tui(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    session = Session.new(temp_dir)
    session.append(Message.user("first"))
    session.append(Message.assistant("second"))

    seen: dict[str, object] = {}
    sentinel_provider = object()

    def fake_create_provider(config):
        return sentinel_provider

    def fake_run_tui(config, session, llm_provider):
        seen["session"] = session
        seen["provider"] = llm_provider

    monkeypatch.setattr(cli, "_create_llm_provider", fake_create_provider)
    monkeypatch.setattr(cli, "_run_tui", fake_run_tui)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["tree", "--session", str(session.path), "--to", "0"],
    )

    assert result.exit_code == 0
    assert "Set session leaf:" in result.stdout
    assert seen["provider"] is sentinel_provider

    updated = seen["session"]
    assert isinstance(updated, Session)
    assert len(updated.messages) == 1
    assert updated.messages[0].content == "first"


def test_cli_sessions_shows_no_sessions_when_empty(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["sessions"])

    assert result.exit_code == 0
    assert "No sessions found" in result.stdout


def test_cli_sessions_reports_load_errors(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    session_dir = home / ".agent" / "sessions"
    session_dir.mkdir(parents=True)
    (session_dir / "broken-session.jsonl").write_text("not-json\n")

    runner = CliRunner()
    result = runner.invoke(cli.app, ["sessions"])

    assert result.exit_code == 0
    assert "broken-session.jsonl (error:" in result.stdout


def test_cli_config_show_prints_current_configuration(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["config-show"])

    assert result.exit_code == 0
    assert "Current configuration:" in result.stdout
    assert "Provider:" in result.stdout
    assert "Model:" in result.stdout


def test_cli_config_show_prints_provider_overrides_from_config_file(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    agent_dir = project / ".agent"
    agent_dir.mkdir()
    (agent_dir / "config.toml").write_text(
        textwrap.dedent(
            """
            [providers.openrouter]
            base_url = "https://openrouter.ai/api/v1"
            model = "openai/gpt-5"
            """
        )
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["config-show"])

    assert result.exit_code == 0
    assert "Configured providers:" in result.stdout
    assert "openrouter:" in result.stdout
    assert "openai/gpt-5" in result.stdout
