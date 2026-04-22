"""Behavior tests for the CLI web command."""

from __future__ import annotations

from typer.testing import CliRunner

from agent import cli


def test_cli_web_dispatches_to_local_server(temp_dir, monkeypatch):
    project = temp_dir / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    calls: list[tuple[str, int]] = []

    def fake_run_web(config, host: str, port: int) -> None:
        calls.append((host, port))

    monkeypatch.setattr(cli, "_run_web", fake_run_web)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["web", "--host", "0.0.0.0", "--port", "9000"])

    assert result.exit_code == 0
    assert calls == [("0.0.0.0", 9000)]
