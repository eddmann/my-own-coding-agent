"""Behavior tests for the smaller example guard extensions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config import Config
from agent.extensions.host import ExtensionHost
from agent.runtime.agent import Agent
from tests.test_doubles.llm_provider_fake import LLMProviderFake
from tests.test_doubles.llm_stream_builders import make_tool_call_events

REPO_ROOT = Path(__file__).resolve().parents[3]


def build_agent(temp_dir, scripts):
    config = Config(
        provider="openai",
        model="gpt-5.4",
        api_key="test-key",
        session_dir=temp_dir,
        context_max_tokens=2048,
        max_output_tokens=2048,
    )
    provider = LLMProviderFake(
        scripts,
        name="openai",
        model="gpt-5.4",
        available_models=["gpt-5.4"],
    )
    agent = Agent(config.to_agent_settings(), provider, cwd=temp_dir)
    host = ExtensionHost(agent)
    agent.set_hooks(host)
    return agent, provider, host


def protected_paths_extension_path() -> Path:
    return REPO_ROOT / "examples" / "extensions" / "protected-paths.py"


def commit_guard_extension_path() -> Path:
    return REPO_ROOT / "examples" / "extensions" / "commit-guard.py"


async def run_command(agent: Agent, prompt: str) -> list[str]:
    outputs: list[str] = []
    async for chunk in agent.run(prompt):
        if chunk.type == "message" and chunk.payload.role.value == "system":
            outputs.append(chunk.payload.content)
    return outputs


@pytest.mark.asyncio
async def test_protected_paths_extension_blocks_sensitive_write(temp_dir):
    blocked_path = temp_dir / ".agent" / "secret.txt"
    agent, _, host = build_agent(
        temp_dir,
        [
            make_tool_call_events(
                "write_1", "write", {"path": str(blocked_path), "content": "nope\n"}
            )
        ],
    )

    try:
        await host.load_extensions([protected_paths_extension_path()])

        async for _ in agent.run("write the secret file"):
            pass

        assert not blocked_path.exists()
        tool_messages = [
            message for message in agent.session.messages if message.role.value == "tool"
        ]
        assert tool_messages
        assert "Tool blocked" in tool_messages[-1].content
        assert "protected path" in tool_messages[-1].content
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_commit_guard_extension_reports_dirty_status_and_blocks_git_commit(
    temp_dir, monkeypatch
):
    import subprocess

    status_lines = " M README.md\n?? draft.txt\n"

    def fake_run(args, check=False, capture_output=False, text=False):
        assert args == ["git", "status", "--porcelain"]
        return SimpleNamespace(stdout=status_lines)

    monkeypatch.setattr(subprocess, "run", fake_run)

    agent, _, host = build_agent(
        temp_dir,
        [
            make_tool_call_events(
                "bash_1", "bash", {"command": "git commit -m test", "cwd": str(temp_dir)}
            )
        ],
    )

    try:
        await host.load_extensions([commit_guard_extension_path()])

        dirty_messages = await run_command(agent, "/dirty")
        assert dirty_messages
        assert "dirty working tree" in dirty_messages[0]
        assert "README.md" in dirty_messages[0]

        async for _ in agent.run("commit the current work"):
            pass

        tool_messages = [
            message for message in agent.session.messages if message.role.value == "tool"
        ]
        assert tool_messages
        assert "Tool blocked" in tool_messages[-1].content
        assert "dirty working tree" in tool_messages[-1].content
        assert "README.md" in tool_messages[-1].content
    finally:
        await agent.close()
