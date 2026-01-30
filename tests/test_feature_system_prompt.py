"""Behavior tests for system prompt assembly."""

from __future__ import annotations

from agent.core.prompt_builder import ContextFile, SystemPromptOptions, build_system_prompt
from agent.skills.skill import Skill, SkillSource


def test_system_prompt_sections_in_order(temp_dir):
    cwd = temp_dir / "work"
    ctx = ContextFile(path=cwd / "AGENTS.md", content="ctx content", source="project")
    skill = Skill(
        name="deploy",
        description="deploy app",
        readme_path=cwd / "skills" / "deploy" / "SKILL.md",
        readme_content="Skill body",
        base_dir=cwd,
        source=SkillSource.PROJECT,
    )

    options = SystemPromptOptions(
        custom_prompt="custom",
        selected_tools=["read", "edit", "grep", "find", "ls"],
        append_system_prompt="append",
        cwd=cwd,
        context_files=[ctx],
        skills=[skill],
    )

    prompt = build_system_prompt(options)

    sections = [
        "custom",
        "Available tools:",
        "Guidelines:",
        "## Project Context (AGENTS.md)",
        "<available_skills>",
        "Environment:",
        "append",
    ]

    indices = [prompt.find(section) for section in sections]
    assert all(index >= 0 for index in indices)
    assert indices == sorted(indices)

    assert "- read: Read file contents with line numbers" in prompt
    assert "When exploring files" in prompt
    assert "Working Directory" in prompt
