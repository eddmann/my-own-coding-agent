"""Behavior tests for the example subagent extension."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from agent.config import Config
from agent.extensions.api import ExtensionUIBindings
from agent.extensions.host import ExtensionHost
from agent.llm import factory as llm_factory
from agent.runtime.agent import Agent
from tests.test_doubles.llm_provider_fake import LLMProviderFake
from tests.test_doubles.llm_stream_builders import make_text_events, make_tool_call_events

REPO_ROOT = Path(__file__).resolve().parents[3]


def build_agent(temp_dir, scripts):
    config = Config(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        session_dir=temp_dir,
        context_max_tokens=2048,
        max_output_tokens=2048,
    )
    provider = LLMProviderFake(scripts, name="openai", model="gpt-4o")
    agent = Agent(config.to_agent_settings(), provider, cwd=temp_dir)
    host = ExtensionHost(agent)
    agent.set_hooks(host)
    return agent, provider, host


def extension_path() -> Path:
    return REPO_ROOT / "examples" / "extensions" / "subagents.py"


async def wait_for_subagent_status(
    agent: Agent, run_id: str, status: str, attempts: int = 200
) -> str:
    for _ in range(attempts):
        messages = await run_command(agent, "/subagents")
        if messages and run_id in messages[0] and f"[{status}]" in messages[0]:
            return messages[0]
        await asyncio.sleep(0)
    raise AssertionError(f"subagent {run_id} did not reach status {status}")


def extract_run_id(system_messages: list[str]) -> str:
    for message in reversed(system_messages):
        match = re.search(r"Launched subagent ([0-9a-f]{8})", message)
        if match:
            return match.group(1)
    raise AssertionError(f"could not find subagent id in: {system_messages}")


async def run_command(agent: Agent, prompt: str) -> list[str]:
    outputs: list[str] = []
    async for chunk in agent.run(prompt):
        if chunk.type == "message" and chunk.payload.role.value == "system":
            outputs.append(chunk.payload.content)
    return outputs


@pytest.mark.asyncio
async def test_subagent_researcher_can_complete_and_apply_summary(temp_dir, monkeypatch):
    child_providers: list[LLMProviderFake] = []
    widgets: list[tuple[str, str | None]] = []

    def fake_create_provider(**kwargs):
        provider = LLMProviderFake(
            [
                make_text_events(
                    '{"summary":"Context mapped","details":"Relevant code paths identified.",'
                    '"findings":["Extension hooks traced","Tool activation checked"],'
                    '"recommended_next_step":"Open the runtime hooks next"}'
                )
            ],
            name="openai",
            model=kwargs["model"] or "gpt-4o",
        )
        child_providers.append(provider)
        return provider

    monkeypatch.setattr(llm_factory, "create_provider", fake_create_provider)

    agent, parent_provider, host = build_agent(temp_dir, [make_text_events("parent ok")])
    host.bind_ui(
        ExtensionUIBindings(
            set_widget=lambda slot, view: widgets.append(
                (slot, None if view is None else view.render())
            )
        )
    )
    try:
        await host.load_extensions([extension_path()])

        launch_messages = await run_command(agent, "/subagent researcher map the extension host")
        run_id = extract_run_id(launch_messages)

        listing = await wait_for_subagent_status(agent, run_id, "completed")
        assert "Context mapped" in listing
        assert any(
            content is not None and f"{run_id} [completed]" in content for _, content in widgets
        )

        prior_parent_calls = len(parent_provider.stream_calls)
        apply_messages = await run_command(agent, f"/subagent-apply {run_id} summary")
        assert any(
            f"Queued summary result from subagent {run_id}" in message for message in apply_messages
        )
        post_apply_listing = await run_command(agent, "/subagents")
        assert post_apply_listing
        assert f"{run_id} [applied]" in post_apply_listing[0]
        assert widgets
        assert widgets[-1][0] == "right_panel"
        assert widgets[-1][1] is None

        assert len(parent_provider.stream_calls) > prior_parent_calls
        final_user_messages = [
            message.content
            for message in parent_provider.stream_calls[prior_parent_calls]["messages"]
            if message.role.value == "user"
        ]
        assert "Context mapped" in final_user_messages[-1]

        assert child_providers
        tool_names = [
            schema["function"]["name"] for schema in child_providers[0].stream_calls[0]["tools"]
        ]
        assert tool_names == ["read", "grep", "find", "ls"]
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_subagent_show_uses_present_and_reports_files_and_commands(temp_dir, monkeypatch):
    child_providers: list[LLMProviderFake] = []
    changed_file = temp_dir / "notes.txt"

    def fake_create_provider(**kwargs):
        provider = LLMProviderFake(
            [
                make_tool_call_events(
                    "write-1",
                    "write",
                    {"path": str(changed_file), "content": "done\n"},
                ),
                make_tool_call_events(
                    "bash-1",
                    "bash",
                    {"command": "echo ok", "cwd": str(temp_dir)},
                ),
                make_text_events(
                    '{"summary":"Implemented the requested change",'
                    '"details":"Wrote notes.txt and verified the shell command.",'
                    '"findings":["Created notes.txt","Ran echo ok"],'
                    '"recommended_next_step":"Run the focused tests next"}'
                ),
            ],
            name="openai",
            model=kwargs["model"] or "gpt-4o",
        )
        child_providers.append(provider)
        return provider

    monkeypatch.setattr(llm_factory, "create_provider", fake_create_provider)

    agent, _, host = build_agent(temp_dir, [])
    presented: list[str] = []
    host.bind_ui(
        ExtensionUIBindings(
            confirm=lambda prompt: True,
            present=lambda view: presented.append(view.render()),
        )
    )
    try:
        await host.load_extensions([extension_path()])

        launch_messages = await run_command(agent, "/subagent implementer create a note file")
        run_id = extract_run_id(launch_messages)

        await wait_for_subagent_status(agent, run_id, "completed")

        show_messages = await run_command(agent, f"/subagent-show {run_id}")
        assert show_messages
        assert any("Implemented the requested change" in message for message in show_messages)
        assert presented
        assert "notes.txt" in presented[0]
        assert "echo ok" in presented[0]
        assert "Implemented the requested change" in presented[0]
        assert "Created notes.txt" in presented[0]
        assert "Run the focused tests next" in presented[0]
        assert changed_file.exists()

        assert child_providers
        tool_names = [
            schema["function"]["name"] for schema in child_providers[0].stream_calls[0]["tools"]
        ]
        assert tool_names == ["read", "write", "edit", "bash", "grep", "find", "ls"]
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_subagent_no_args_uses_select_input_and_confirm_flow(temp_dir, monkeypatch):
    child_providers: list[LLMProviderFake] = []

    def fake_create_provider(**kwargs):
        provider = LLMProviderFake(
            [
                make_text_events(
                    '{"summary":"Review finished","details":"Checked the extension surface.",'
                    '"findings":["No blocking issues"],'
                    '"recommended_next_step":"Apply the review summary"}'
                )
            ],
            name="openai",
            model=kwargs["model"] or "gpt-4o",
        )
        child_providers.append(provider)
        return provider

    monkeypatch.setattr(llm_factory, "create_provider", fake_create_provider)

    agent, _, host = build_agent(temp_dir, [])
    selects: list[tuple[str, list[str]]] = []
    prompts: list[tuple[str, str | None]] = []
    host.bind_ui(
        ExtensionUIBindings(
            select=lambda prompt, options: (
                selects.append((prompt, options))
                or "reviewer - Read-only bug, regression, and testing review."
            ),
            input=lambda prompt, default=None: (
                prompts.append((prompt, default)) or "inspect the new extension host"
            ),
        )
    )
    try:
        await host.load_extensions([extension_path()])

        launch_messages = await run_command(agent, "/subagent")
        run_id = extract_run_id(launch_messages)

        listing = await wait_for_subagent_status(agent, run_id, "completed")
        assert "Review finished" in listing
        assert selects
        assert selects[0][0] == "Choose subagent profile"
        assert any(option.startswith("reviewer - ") for option in selects[0][1])
        assert prompts
        assert prompts[0][0] == "Task for reviewer"

        assert child_providers
        tool_names = [
            schema["function"]["name"] for schema in child_providers[0].stream_calls[0]["tools"]
        ]
        assert tool_names == ["read", "grep", "find", "ls"]
    finally:
        await agent.close()
