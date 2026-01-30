"""Behavior tests for prompt autocomplete."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, OptionList

from agent.prompts.loader import PromptTemplateLoader
from agent.prompts.template import TemplateSource
from agent.skills.loader import SkillLoader
from agent.skills.skill import SkillSource
from agent.tui.input import PromptInput

if TYPE_CHECKING:
    from pathlib import Path


def write_skill(dir_path: Path, name: str) -> None:
    skill_dir = dir_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""
            ---
            name: {name}
            description: test skill
            ---

            # {name}
            """
        ).lstrip()
    )


def write_template(dir_path: Path, filename: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / filename).write_text(
        textwrap.dedent(
            """
            ---
            name: build
            description: build template
            ---

            run build
            """
        ).lstrip()
    )


class InputApp(App[None]):
    def __init__(self, prompt_input: PromptInput) -> None:
        super().__init__()
        self._prompt_input = prompt_input

    def compose(self) -> ComposeResult:
        yield self._prompt_input


@pytest.mark.asyncio
async def test_autocomplete_includes_skills_templates_and_extensions(temp_dir):
    skills_dir = temp_dir / "skills"
    templates_dir = temp_dir / "templates"

    write_skill(skills_dir, "deploy")
    write_template(templates_dir, "build.md")

    skill_loader = SkillLoader([(skills_dir, SkillSource.PATH)])
    template_loader = PromptTemplateLoader([(templates_dir, TemplateSource.PATH)])

    prompt = PromptInput(
        skill_loader=skill_loader,
        template_loader=template_loader,
        extension_commands=["ping"],
        id="prompt-input",
    )
    app = InputApp(prompt)

    async with app.run_test() as pilot:
        await pilot.pause()

        input_widget = prompt.query_one("#prompt-inner", Input)
        input_widget.focus()
        await pilot.press("$")
        await pilot.pause()

        option_list = prompt.query_one("#suggestions", OptionList)
        options = [
            option_list.get_option_at_index(i).prompt for i in range(option_list.option_count)
        ]

        assert "$deploy" in options

        input_widget.value = ""
        await pilot.press("/")
        await pilot.pause()

        options = [
            option_list.get_option_at_index(i).prompt for i in range(option_list.option_count)
        ]

        assert "/build" in options
        assert "/ping" in options
        assert "/help" in options
