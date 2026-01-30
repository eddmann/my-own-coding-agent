"""Behavior tests for skills loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent.core.agent import Agent
from agent.core.config import Config
from agent.skills.loader import SkillLoader
from agent.skills.skill import SkillSource
from tests.fakes import FakeLLMProvider, make_text_events

if TYPE_CHECKING:
    from pathlib import Path


def write_skill(dir_path: Path, name: str, description: str) -> None:
    skill_dir = dir_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n"
    )


@pytest.mark.asyncio
async def test_skill_loader_priority(temp_dir, monkeypatch):
    home = temp_dir / "home"
    cwd = temp_dir / "project"
    extra = temp_dir / "extra"

    monkeypatch.setenv("HOME", str(home))

    (home / ".agent" / "skills").mkdir(parents=True, exist_ok=True)
    (cwd / ".agent" / "skills").mkdir(parents=True, exist_ok=True)
    extra.mkdir(parents=True, exist_ok=True)

    write_skill(home / ".agent" / "skills", "deploy", "user")
    write_skill(extra, "deploy", "path")
    write_skill(cwd / ".agent" / "skills", "deploy", "project")

    loader = SkillLoader.with_defaults(extra_dirs=[extra], cwd=cwd)

    skill = loader.get("deploy")
    assert skill is not None
    assert skill.description == "project"


def test_skill_validation_rejects_missing_description(temp_dir):
    skill_dir = temp_dir / "badskill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text("---\nname: badskill\n---\n\n# bad")

    loader = SkillLoader([(temp_dir, SkillSource.PATH)])
    _ = loader.skills

    assert loader.diagnostics.error_count == 1


@pytest.mark.asyncio
async def test_skill_slash_command_expands_into_prompt(temp_dir):
    skills_dir = temp_dir / "skills"
    write_skill(skills_dir, "deploy", "deploy skill")

    config = Config(
        provider="openai",
        model="fake-model",
        api_key="test",
        session_dir=temp_dir,
        context_max_tokens=2048,
        max_output_tokens=1024,
        skills_dirs=[skills_dir],
    )

    agent = Agent(config)
    fake = FakeLLMProvider([make_text_events("ok")])
    agent.provider = fake
    agent.context.provider = fake

    async for _ in agent.run("$deploy staging"):
        pass

    user_messages = [m for m in agent.session.messages if m.role.value == "user"]
    assert user_messages
    content = user_messages[-1].content
    assert '<skill name="deploy"' in content
    assert "References are relative to" in content
    assert "staging" in content
