"""Behavior tests for CLI run command flows via CLI runner boundary."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from agent import cli
from agent.core.session import Session

if TYPE_CHECKING:
    from pathlib import Path


def write_extension(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            from agent.extensions.api import ExtensionAPI

            def setup(api: ExtensionAPI):
                def dump(args, ctx):
                    cfg = ctx.get_config()
                    return "|".join([
                        str(cfg["prompt_template_dirs"]),
                        str(cfg["context_file_paths"]),
                        str(cfg["custom_system_prompt"]),
                        str(cfg["append_system_prompt"]),
                    ])
                api.register_command("dump-config", dump)
            """
        )
    )


def test_cli_preserves_prompt_and_context_fields(temp_dir, monkeypatch):
    project = temp_dir / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    agent_dir = project / ".agent"
    agent_dir.mkdir()

    prompts_dir = project / "prompts"
    prompts_dir.mkdir()

    context_file = project / "CONTEXT.md"
    context_file.write_text("context")

    (agent_dir / "config.toml").write_text(
        textwrap.dedent(
            f"""
            custom_system_prompt = "custom"
            append_system_prompt = "append"
            prompt_template_dirs = ["{prompts_dir}"]
            context_file_paths = ["{context_file}"]
            """
        )
    )

    ext_path = project / "ext_dump.py"
    write_extension(ext_path)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "run",
            "--headless",
            "--extension",
            str(ext_path),
            "/dump-config",
            "-m",
            "gpt-4o",
        ],
    )

    assert result.exit_code == 0
    output = result.stdout
    assert "custom" in output
    assert "append" in output
    assert str(prompts_dir) in output
    assert str(context_file) in output


def test_cli_run_rejects_invalid_thinking_level():
    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", "--headless", "hi", "--thinking", "invalid"])

    assert result.exit_code == 1
    assert "Invalid thinking level" in result.stderr


def test_cli_run_headless_requires_prompt():
    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", "--headless"])

    assert result.exit_code == 1
    assert "Prompt required in headless mode" in result.stderr


def test_cli_run_fails_on_invalid_provider_model_pair():
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["run", "--headless", "hi", "--provider", "openai", "--model", "claude-sonnet-4-5"],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "not valid for provider" in str(result.exception)


def test_cli_run_headless_loads_explicit_session(temp_dir, monkeypatch):
    project = temp_dir / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    ext_path = project / "ext_dump.py"
    write_extension(ext_path)
    session = Session.new(temp_dir)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "run",
            "--headless",
            "--session",
            str(session.path),
            "--extension",
            str(ext_path),
            "/dump-config",
        ],
    )

    assert result.exit_code == 0
    assert "Loaded session:" in result.stdout


def test_cli_run_errors_when_explicit_session_path_missing(temp_dir, monkeypatch):
    project = temp_dir / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    missing = temp_dir / "missing.jsonl"

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["run", "--headless", "hello", "--session", str(missing)],
    )

    assert result.exit_code == 1
    assert "Session file not found" in result.stderr


def test_cli_run_resume_without_previous_session_still_runs_headless(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    ext_path = project / "ext_dump.py"
    write_extension(ext_path)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["run", "--resume", "--headless", "--extension", str(ext_path), "/dump-config"],
    )

    assert result.exit_code == 0
    assert "No previous session found" in result.stderr


def test_cli_run_dispatches_to_tui_when_not_headless(temp_dir, monkeypatch):
    project = temp_dir / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    calls: dict[str, int] = {"tui": 0, "headless": 0}
    sentinel_provider = object()

    def fake_create_provider(config):
        return sentinel_provider

    def fake_run_tui(config, session, llm_provider):
        assert session is None
        assert llm_provider is sentinel_provider
        calls["tui"] += 1

    async def fake_run_headless(config, prompt, session, llm_provider):
        calls["headless"] += 1

    monkeypatch.setattr(cli, "_create_llm_provider", fake_create_provider)
    monkeypatch.setattr(cli, "_run_tui", fake_run_tui)
    monkeypatch.setattr(cli, "_run_headless", fake_run_headless)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run"])

    assert result.exit_code == 0
    assert calls["tui"] == 1
    assert calls["headless"] == 0


def test_cli_run_dispatches_to_headless_when_prompt_or_headless_flag(temp_dir, monkeypatch):
    project = temp_dir / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    calls: dict[str, int] = {"tui": 0, "headless": 0}
    seen_prompt: list[str] = []
    sentinel_provider = object()

    def fake_create_provider(config):
        return sentinel_provider

    def fake_run_tui(config, session, llm_provider):
        calls["tui"] += 1

    async def fake_run_headless(config, prompt, session, llm_provider):
        assert session is None
        assert llm_provider is sentinel_provider
        seen_prompt.append(prompt)
        calls["headless"] += 1

    monkeypatch.setattr(cli, "_create_llm_provider", fake_create_provider)
    monkeypatch.setattr(cli, "_run_tui", fake_run_tui)
    monkeypatch.setattr(cli, "_run_headless", fake_run_headless)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", "--headless", "hello"])

    assert result.exit_code == 0
    assert calls["tui"] == 0
    assert calls["headless"] == 1
    assert seen_prompt == ["hello"]
