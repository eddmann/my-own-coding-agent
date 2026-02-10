"""Behavior tests for prompt template loading and parsing."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from agent.prompts.loader import PromptTemplateLoader
from agent.prompts.parser import expand_template, parse_command
from agent.prompts.template import TemplateSource

if TYPE_CHECKING:
    from pathlib import Path


def write_template(dir_path: Path, filename: str, content: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / filename).write_text(textwrap.dedent(content).lstrip())


def test_prompt_template_priority(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = temp_dir / "project"
    extra = temp_dir / "extra"

    monkeypatch.setenv("HOME", str(home))

    user_dir = home / ".agent" / "prompts"
    project_dir = project / ".agent" / "prompts"

    write_template(
        user_dir,
        "deploy.md",
        """
        ---
        name: deploy
        description: user version
        ---

        User template
        """,
    )
    write_template(
        extra,
        "deploy.md",
        """
        ---
        name: deploy
        description: path version
        ---

        Path template
        """,
    )
    write_template(
        project_dir,
        "deploy.md",
        """
        ---
        name: deploy
        description: project version
        ---

        Project template
        """,
    )

    loader = PromptTemplateLoader.with_defaults(extra_dirs=[extra], cwd=project)

    template = loader.get("deploy")

    assert template is not None
    assert template.description == "project version"
    assert "Project template" in template.content


def test_prompt_template_frontmatter_name_override(temp_dir):
    templates = temp_dir / "templates"
    write_template(
        templates,
        "build.md",
        """
        ---
        name: release
        description: build description
        ---

        run build
        """,
    )

    loader = PromptTemplateLoader([(templates, TemplateSource.PATH)])
    template = loader.get("release")

    assert template is not None
    assert template.name == "release"
    assert template.description == "build description"


def test_prompt_template_loader_ignores_non_markdown(temp_dir):
    templates = temp_dir / "templates"
    templates.mkdir()
    (templates / "note.txt").write_text("ignore")

    loader = PromptTemplateLoader([(templates, TemplateSource.PATH)])

    assert loader.list_templates() == []


def test_parse_command_handles_quotes():
    command = parse_command('/review "file name.py" next')

    assert command is not None
    assert command.template_name == "review"
    assert command.arguments == ["file name.py", "next"]


def test_expand_template_substitutes_arguments():
    command = parse_command("/tmpl one two three")
    assert command is not None

    content = "A=$1 B=$2 REST=$@ SLICE=${@:2}"
    expanded = expand_template(content, command)

    assert "A=one" in expanded
    assert "B=two" in expanded
    assert "REST=one two three" in expanded
    assert "SLICE=two three" in expanded
