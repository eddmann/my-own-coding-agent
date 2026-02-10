"""Behavior tests for skill loading and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent.skills.loader import SkillLoader
from agent.skills.skill import SkillSource

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
