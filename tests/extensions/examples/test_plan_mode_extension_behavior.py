"""Behavior tests for the example plan mode extension."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.config import Config
from agent.extensions.api import ExtensionUIBindings
from agent.extensions.host import ExtensionHost
from agent.runtime.agent import Agent
from agent.runtime.settings import ThinkingLevel
from tests.test_doubles.llm_provider_fake import LLMProviderFake
from tests.test_doubles.llm_stream_builders import make_text_events

REPO_ROOT = Path(__file__).resolve().parents[3]


def build_agent(temp_dir, scripts):
    config = Config(
        provider="openai",
        model="gpt-5.4",
        api_key="test-key",
        session_dir=temp_dir,
        context_max_tokens=4096,
        max_output_tokens=4096,
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


def extension_path() -> Path:
    return REPO_ROOT / "examples" / "extensions" / "plan-mode.py"


async def run_command(agent: Agent, prompt: str) -> list[str]:
    outputs: list[str] = []
    async for chunk in agent.run(prompt):
        if chunk.type == "message" and chunk.payload.role.value == "system":
            outputs.append(chunk.payload.content)
    return outputs


@pytest.mark.asyncio
async def test_plan_mode_can_build_persist_and_apply_plan(temp_dir):
    plan_json = json.dumps(
        {
            "summary": "Refactor the extension host in staged steps.",
            "steps": [
                "Document the extension context surface.",
                "Add a focused example extension.",
                "Verify the behavior with tests.",
            ],
            "risks": ["Breaking old examples while updating docs."],
            "validation": ["Run the focused test suite.", "Check the updated docs render cleanly."],
            "notes": ["Keep the example smaller than subagents."],
        }
    )
    agent, provider, host = build_agent(
        temp_dir,
        [
            make_text_events(plan_json),
            make_text_events("Implemented from the approved plan."),
        ],
    )
    original_tools = agent.tools.list_active_tools()

    try:
        await host.load_extensions([extension_path()])

        draft_messages = await run_command(
            agent, "/plan design a better mid-sized extension example"
        )
        assert any("Queued planning request" in message for message in draft_messages)
        assert agent.tools.list_active_tools() == ["read", "grep", "find", "ls"]
        assert agent.config.thinking_level == ThinkingLevel.HIGH

        assert len(provider.stream_calls) == 1
        tool_names = [schema["function"]["name"] for schema in provider.stream_calls[0]["tools"]]
        assert tool_names == ["read", "grep", "find", "ls"]
        first_user_messages = [
            message.content
            for message in provider.stream_calls[0]["messages"]
            if message.role.value == "user"
        ]
        assert "Planning mode is active" in first_user_messages[-1]

        show_messages = await run_command(agent, "/plan show")
        assert show_messages
        assert "Refactor the extension host in staged steps." in show_messages[0]
        assert "Document the extension context surface." in show_messages[0]

        plan_path = temp_dir / ".agent" / "plans" / f"{agent.session.metadata.id}.json"
        assert plan_path.exists()
        payload = json.loads(plan_path.read_text())
        assert payload["summary"] == "Refactor the extension host in staged steps."
        assert payload["steps"][1] == "Add a focused example extension."

        apply_messages = await run_command(agent, "/plan apply")
        assert any(
            "Queued current plan into the main thread" in message for message in apply_messages
        )
        assert len(provider.stream_calls) == 2
        assert agent.tools.list_active_tools() == original_tools
        assert agent.config.thinking_level == ThinkingLevel.OFF

        second_user_messages = [
            message.content
            for message in provider.stream_calls[1]["messages"]
            if message.role.value == "user"
        ]
        assert "Execute this approved plan." in second_user_messages[-1]
        assert "Verify the behavior with tests." in second_user_messages[-1]
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_plan_mode_uses_ui_status_widget_and_presented_show(temp_dir):
    plan_json = json.dumps(
        {
            "summary": "Review the session tree flow before changing it.",
            "steps": ["Map the current session behavior.", "List the risky edge cases."],
            "validation": ["Run the session behavior tests."],
        }
    )
    agent, _, host = build_agent(temp_dir, [make_text_events(plan_json)])
    notifications: list[tuple[str, str]] = []
    statuses: list[str | None] = []
    widgets: list[tuple[str, str | None]] = []
    presented: list[str] = []

    host.bind_ui(
        ExtensionUIBindings(
            notify=lambda message, level: notifications.append((message, level)),
            set_status=lambda text: statuses.append(text),
            set_widget=lambda slot, view: widgets.append(
                (slot, None if view is None else view.render())
            ),
            present=lambda view: presented.append(view.render()),
        )
    )

    try:
        await host.load_extensions([extension_path()])

        on_messages = await run_command(agent, "/plan on")
        assert any("Plan mode enabled" in message for message in on_messages)
        assert ("Plan mode enabled", "info") in notifications
        assert statuses == ["plan mode"]
        assert widgets
        assert widgets[0][0] == "footer"
        assert widgets[0][1] is not None
        assert "Plan mode [ready]" in widgets[0][1]

        await run_command(agent, "/plan review the session tree behavior")

        show_messages = await run_command(agent, "/plan show")
        assert show_messages
        assert "Review the session tree flow before changing it." in show_messages[0]
        assert presented
        assert "Map the current session behavior." in presented[0]
        assert "Run the session behavior tests." in presented[0]

        off_messages = await run_command(agent, "/plan off")
        assert any("Plan mode disabled" in message for message in off_messages)
        assert statuses[-1] is None
        assert widgets[-1] == ("footer", None)
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_plan_mode_clear_removes_saved_plan(temp_dir):
    plan_json = json.dumps(
        {
            "summary": "Clean up the plan state.",
            "steps": ["Draft the plan.", "Clear the saved plan."],
        }
    )
    agent, _, host = build_agent(temp_dir, [make_text_events(plan_json)])

    try:
        await host.load_extensions([extension_path()])

        await run_command(agent, "/plan draft a temporary plan")

        plan_path = temp_dir / ".agent" / "plans" / f"{agent.session.metadata.id}.json"
        assert plan_path.exists()

        clear_messages = await run_command(agent, "/plan clear")

        assert any("Cleared current plan" in message for message in clear_messages)
        assert not plan_path.exists()
    finally:
        await agent.close()
