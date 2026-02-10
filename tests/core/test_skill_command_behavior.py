"""Behavior tests for skill command expansion in the agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent.config import Config
from agent.core.agent import Agent
from tests.test_doubles.llm_provider_fake import LLMProviderFake
from tests.test_doubles.llm_stream_builders import make_text_events

if TYPE_CHECKING:
    from pathlib import Path


def write_skill(dir_path: Path, name: str, description: str) -> None:
    skill_dir = dir_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n"
    )


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

    provider = LLMProviderFake([make_text_events("ok")])
    agent = Agent(config.to_agent_settings(), provider)

    async for _ in agent.run("$deploy staging"):
        pass

    user_messages = [message for message in agent.session.messages if message.role.value == "user"]
    assert user_messages
    content = user_messages[-1].content
    assert '<skill name="deploy"' in content
    assert "References are relative to" in content
    assert "staging" in content
