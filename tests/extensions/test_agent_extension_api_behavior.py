"""Behavior tests for the v1 extension host API."""

from __future__ import annotations

import asyncio
import textwrap
from typing import TYPE_CHECKING

import pytest

from agent.config import Config
from agent.extensions.api import ExtensionUIBindings
from agent.extensions.host import ExtensionHost
from agent.runtime.agent import Agent
from agent.runtime.message import Message
from agent.runtime.settings import ThinkingLevel
from tests.test_doubles.llm_provider_fake import LLMProviderFake
from tests.test_doubles.llm_stream_builders import make_text_events

if TYPE_CHECKING:
    from pathlib import Path


def build_agent(temp_dir, scripts):
    config = Config(
        provider="openai",
        model="gpt-4o",
        api_key="test",
        session_dir=temp_dir,
        context_max_tokens=2048,
        max_output_tokens=2048,
    )
    provider = LLMProviderFake(scripts, name="openai", model="gpt-4o")
    agent = Agent(config.to_agent_settings(), provider, cwd=temp_dir)
    host = ExtensionHost(agent)
    agent.set_hooks(host)
    return agent, provider, host


def write_extension(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content))


@pytest.mark.asyncio
async def test_extension_command_can_queue_follow_up_user_message(temp_dir):
    ext_path = temp_dir / "ext_followup.py"
    write_extension(
        ext_path,
        """
        from agent.extensions.api import ExtensionAPI

        def setup(api: ExtensionAPI):
            async def followup(args, ctx):
                await ctx.runtime.send_user_message("queued follow-up")
                return None
            api.register_command("followup", followup)
        """,
    )

    agent, provider, host = build_agent(temp_dir, [make_text_events("done")])
    await host.load_extensions([ext_path])

    async for _ in agent.run("/followup"):
        pass

    assert len(provider.stream_calls) == 1
    user_messages = [
        m.content for m in provider.stream_calls[0]["messages"] if m.role.value == "user"
    ]
    assert user_messages[-1] == "queued follow-up"
    assert [m.content for m in agent.session.messages if m.role.value == "user"] == [
        "/followup",
        "queued follow-up",
    ]


@pytest.mark.asyncio
async def test_extension_command_can_use_model_tools_and_runtime_context(temp_dir):
    ext_path = temp_dir / "ext_mode.py"
    write_extension(
        ext_path,
        """
        from agent.extensions.api import ExtensionAPI

        def setup(api: ExtensionAPI):
            def mode(args, ctx):
                ctx.model.set("gpt-5.4")
                ctx.model.set_thinking_level("high")
                ctx.tools.set_active(["read", "find"])
                return "|".join(
                    [
                        str(ctx.cwd.name),
                        ctx.model.get(),
                        ctx.model.get_thinking_level(),
                        ",".join(ctx.tools.active()),
                        str(
                            "You are a helpful coding assistant"
                            in ctx.runtime.get_system_prompt()
                        ),
                    ]
                )
            api.register_command("mode", mode)
        """,
    )

    agent, provider, host = build_agent(temp_dir, [])
    provider._available_models.append("gpt-5.4")
    await host.load_extensions([ext_path])

    outputs: list[str] = []
    async for chunk in agent.run("/mode"):
        if chunk.type == "message" and chunk.payload.role.value == "system":
            outputs.append(chunk.payload.content)

    assert outputs
    assert temp_dir.name in outputs[0]
    assert "gpt-5.4" in outputs[0]
    assert "high" in outputs[0]
    assert "read,find" in outputs[0]
    assert "True" in outputs[0]
    assert agent.provider.model == "gpt-5.4"
    assert agent.config.thinking_level == ThinkingLevel.HIGH
    assert agent.tools.list_active_tools() == ["read", "find"]


@pytest.mark.asyncio
async def test_extension_command_active_tools_affect_next_llm_turn(temp_dir):
    ext_path = temp_dir / "ext_active_tools.py"
    write_extension(
        ext_path,
        """
        from agent.extensions.api import ExtensionAPI

        def setup(api: ExtensionAPI):
            async def focus(args, ctx):
                ctx.tools.set_active(["read", "find"])
                await ctx.runtime.send_user_message("go")
                return None
            api.register_command("focus", focus)
        """,
    )

    agent, provider, host = build_agent(temp_dir, [make_text_events("ok")])
    await host.load_extensions([ext_path])

    async for _ in agent.run("/focus"):
        pass

    assert len(provider.stream_calls) == 1
    tool_names = [schema["function"]["name"] for schema in provider.stream_calls[0]["tools"]]
    assert tool_names == ["read", "find"]


@pytest.mark.asyncio
async def test_extension_command_can_register_runtime_tool_via_ctx_tools(temp_dir):
    ext_path = temp_dir / "ext_runtime_tool.py"
    write_extension(
        ext_path,
        """
        from pydantic import BaseModel
        from agent.extensions.api import ExtensionAPI
        from agent.tools.base import BaseTool

        class DynamicParams(BaseModel):
            text: str

        class DynamicTool(BaseTool[DynamicParams]):
            name = "dynamic_echo"
            description = "Echo text from a runtime-registered tool."
            parameters = DynamicParams

            async def execute(self, params: DynamicParams) -> str:
                return f"dynamic:{params.text}"

        def setup(api: ExtensionAPI):
            def enable(args, ctx):
                ctx.tools.register(DynamicTool())
                return "enabled"

            api.register_command("enable-runtime-tool", enable)
        """,
    )

    agent, provider, host = build_agent(temp_dir, [make_text_events("ok")])
    await host.load_extensions([ext_path])

    outputs: list[str] = []
    async for chunk in agent.run("/enable-runtime-tool"):
        if chunk.type == "message" and chunk.payload.role.value == "system":
            outputs.append(chunk.payload.content)

    assert outputs
    assert "enabled" in outputs[0]
    assert "dynamic_echo" in agent.tools

    result = await agent.tools.execute("dynamic_echo", {"text": "hi"})
    assert result.is_error is False
    assert result.content == "dynamic:hi"

    async for _ in agent.run("hello"):
        pass

    tool_names = [schema["function"]["name"] for schema in provider.stream_calls[0]["tools"]]
    assert "dynamic_echo" in tool_names


@pytest.mark.asyncio
async def test_extension_command_can_switch_session_via_new_and_fork(temp_dir):
    ext_path = temp_dir / "ext_session.py"
    write_extension(
        ext_path,
        """
        from agent.extensions.api import ExtensionAPI

        def setup(api: ExtensionAPI):
            async def new_session(args, ctx):
                session_id = await ctx.session.new()
                return f"new:{session_id}"

            async def fork_here(args, ctx):
                current_message_id = ctx.session.messages()[-1].id
                session_id = await ctx.session.fork(current_message_id)
                return f"fork:{session_id}:{ctx.session.parent_id}"

            api.register_command("new-session", new_session)
            api.register_command("fork-here", fork_here)
        """,
    )

    agent, _, host = build_agent(temp_dir, [])
    await host.load_extensions([ext_path])
    original_id = agent.session.metadata.id

    async for _ in agent.run("/new-session"):
        pass

    assert agent.session.metadata.id != original_id
    assert agent.session.metadata.parent_session_id is None
    new_id = agent.session.metadata.id

    async for _ in agent.run("/fork-here"):
        pass

    assert agent.session.metadata.id != new_id
    assert agent.session.metadata.parent_session_id == new_id


@pytest.mark.asyncio
async def test_extension_command_can_use_ui_callbacks(temp_dir):
    ext_path = temp_dir / "ext_ui.py"
    write_extension(
        ext_path,
        """
        from agent.extensions.api import ExtensionAPI

        def setup(api: ExtensionAPI):
            def ui_demo(args, ctx):
                if ctx.ui is not None:
                    ctx.ui.notify("heads up", "warning")
                    ctx.ui.set_status("busy")
                return "ok"
            api.register_command("ui-demo", ui_demo)
        """,
    )

    agent, _, host = build_agent(temp_dir, [])
    notifications: list[tuple[str, str]] = []
    statuses: list[str | None] = []
    host.bind_ui(
        ExtensionUIBindings(
            notify=lambda message, level: notifications.append((message, level)),
            set_status=lambda text: statuses.append(text),
        )
    )
    await host.load_extensions([ext_path])

    async for _ in agent.run("/ui-demo"):
        pass

    assert notifications == [("heads up", "warning")]
    assert statuses == ["busy"]


@pytest.mark.asyncio
async def test_extension_command_can_use_interactive_ui_callbacks(temp_dir):
    ext_path = temp_dir / "ext_ui_interactive.py"
    write_extension(
        ext_path,
        """
        from agent.extensions.api import ExtensionAPI, ViewControl

        class DemoView:
            def __init__(self):
                self._value = None

            def render(self):
                return "demo"

            def controls(self):
                return [ViewControl(kind="button", name="done", label="Done")]

            def handle_action(self, action, value=None):
                if action == "done":
                    self._value = "done"

            def is_done(self):
                return self._value is not None

            def result(self):
                return self._value

        def setup(api: ExtensionAPI):
            async def ui_demo(args, ctx):
                if ctx.ui is None:
                    return "missing"
                name = await ctx.ui.input("name", default="anon")
                confirmed = await ctx.ui.confirm("confirm?")
                choice = await ctx.ui.select("pick", ["a", "b"])
                result = await ctx.ui.present(DemoView())
                ctx.ui.set_widget("footer", None)
                return f"{name}|{confirmed}|{choice}|{result}"
            api.register_command("ui-demo-extended", ui_demo)
        """,
    )

    agent, _, host = build_agent(temp_dir, [])
    host.bind_ui(
        ExtensionUIBindings(
            input=lambda prompt, default: "bob",
            confirm=lambda prompt: True,
            select=lambda prompt, options: options[-1],
            present=lambda view: "done",
            set_widget=lambda slot, view: None,
        )
    )
    await host.load_extensions([ext_path])

    outputs: list[str] = []
    async for chunk in agent.run("/ui-demo-extended"):
        if chunk.type == "message" and chunk.payload.role.value == "system":
            outputs.append(chunk.payload.content)

    assert outputs
    assert "bob|True|b|done" in outputs[0]


@pytest.mark.asyncio
async def test_session_lifecycle_events_use_matching_session_context(temp_dir):
    agent, _, host = build_agent(temp_dir, [])
    records: list[tuple[str, str, str]] = []
    original_session_id = agent.session.metadata.id

    def on_session_end(event, ctx):
        records.append((event.type, event.session_id, ctx.session.id))

    def on_session_start(event, ctx):
        records.append((event.type, event.session_id, ctx.session.id))

    host.api.on("session_end", on_session_end)
    host.api.on("session_start", on_session_start)

    await agent.new_session()

    for _ in range(3):
        await asyncio.sleep(0)

    assert records == [
        ("session_end", original_session_id, original_session_id),
        ("session_start", agent.session.metadata.id, agent.session.metadata.id),
    ]


@pytest.mark.asyncio
async def test_agent_close_waits_for_async_session_end_handlers(temp_dir):
    agent, _, host = build_agent(temp_dir, [])
    completed: list[str] = []

    async def on_session_end(event, ctx):
        await asyncio.sleep(0)
        completed.append(ctx.session.id)

    host.api.on("session_end", on_session_end)

    await agent.close()

    assert completed == [agent.session.metadata.id]


@pytest.mark.asyncio
async def test_session_lifecycle_waits_for_session_end_before_session_start(temp_dir):
    agent, _, host = build_agent(temp_dir, [])
    original_session_id = agent.session.metadata.id
    events: list[tuple[str, str]] = []

    async def on_session_end(event, ctx):
        events.append(("session_end:start", event.session_id))
        await asyncio.sleep(0)
        events.append(("session_end:finish", ctx.session.id))

    async def on_session_start(event, ctx):
        events.append(("session_start", ctx.session.id))

    host.api.on("session_end", on_session_end)
    host.api.on("session_start", on_session_start)

    await agent.new_session()

    assert events == [
        ("session_end:start", original_session_id),
        ("session_end:finish", original_session_id),
        ("session_start", agent.session.metadata.id),
    ]


@pytest.mark.asyncio
async def test_session_set_leaf_does_not_emit_session_lifecycle_events(temp_dir):
    agent, _, host = build_agent(temp_dir, [])
    agent.session.append(Message.user("first"))
    agent.session.append(Message.assistant("second"))
    target_entry_id = agent.session.entries[-1].id
    events: list[str] = []

    def on_session_end(event, ctx):
        events.append(f"end:{ctx.session.id}")

    def on_session_start(event, ctx):
        events.append(f"start:{ctx.session.id}")

    host.api.on("session_end", on_session_end)
    host.api.on("session_start", on_session_start)

    agent.set_leaf(target_entry_id)

    for _ in range(3):
        await asyncio.sleep(0)

    assert events == []


@pytest.mark.asyncio
async def test_runtime_follow_up_messages_do_not_leak_outside_run_scope(temp_dir):
    ext_path = temp_dir / "ext_session_leak.py"
    write_extension(
        ext_path,
        """
        from agent.extensions.api import ExtensionAPI

        def setup(api: ExtensionAPI):
            async def on_session_start(event, ctx):
                await ctx.runtime.send_user_message("should not leak")

            api.on("session_start", on_session_start)
        """,
    )

    agent, provider, host = build_agent(temp_dir, [make_text_events("done")])
    await host.load_extensions([ext_path])

    async for _ in agent.run("hello"):
        pass

    assert len(provider.stream_calls) == 1
    user_messages = [
        message.content
        for message in provider.stream_calls[0]["messages"]
        if message.role.value == "user"
    ]
    assert user_messages == ["hello"]
