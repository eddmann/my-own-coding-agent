"""Behavior tests for CLI config merging."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from agent import cli

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

    config_path = agent_dir / "config.toml"
    config_path.write_text(
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
            "override-model",
        ],
    )

    assert result.exit_code == 0
    output = result.stdout
    assert "custom" in output
    assert "append" in output
    assert str(prompts_dir) in output
    assert str(context_file) in output
