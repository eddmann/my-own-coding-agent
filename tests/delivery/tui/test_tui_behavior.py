"""Behavior tests for TUI flows through the AgentApp runner boundary."""

from __future__ import annotations

import asyncio
import textwrap
from typing import TYPE_CHECKING, Any

import pytest
from textual.widgets import (
    Button,
    Input,
    ListView,
    OptionList,
    RadioButton,
    RadioSet,
    Select,
    Static,
    Tree,
)

from agent.config import Config
from agent.core.message import Message, ThinkingContent, ToolCall
from agent.core.session import Session
from agent.core.settings import ThinkingLevel
from agent.llm.events import DoneEvent, ErrorEvent, PartialMessage, StreamOptions, TextStartEvent
from agent.llm.stream import AssistantMessageEventStream
from agent.tui.app import AgentApp
from agent.tui.chat import MessageWidget, SkillInvocationWidget, ThinkingWidget, ToolWidget
from agent.tui.context_modal import ContextModal
from agent.tui.input import PromptInput
from agent.tui.model_modal import ModelModal
from agent.tui.session_modal import SessionForkModal, SessionLoadModal, SessionTreeModal
from tests.test_doubles.llm_provider_fake import LLMProviderFake

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


def write_extension(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            from agent.extensions.api import ExtensionAPI

            def setup(api: ExtensionAPI):
                def ping(args, ctx):
                    return "pong"
                api.register_command("ping", ping)
            """
        ).lstrip()
    )


class CancelAwareProviderFake(LLMProviderFake):
    """Provider double that keeps streaming until cancellation is signaled."""

    def __init__(self) -> None:
        super().__init__([], name="openai", model="gpt-4o")
        self.started = asyncio.Event()

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        self.stream_calls.append({"messages": list(messages), "tools": tools, "options": options})

        stream = AssistantMessageEventStream()
        cancel_event = options.cancel_event if options is not None else None

        async def _run() -> None:
            self.started.set()
            stream.push(TextStartEvent(content_index=0))

            while cancel_event is not None and not cancel_event.is_set():
                await asyncio.sleep(0.01)

            if cancel_event is not None and cancel_event.is_set():
                message = PartialMessage(stop_reason="aborted", error_message="cancelled")
                stream.push(ErrorEvent(stop_reason="aborted", message=message))
                stream.end(message)
                return

            stream.push(DoneEvent(message=PartialMessage()))
            stream.end()

        stream.attach_task(asyncio.create_task(_run()))
        return stream


def status_left_text(app: AgentApp) -> str:
    status = app.query_one("#status-line")
    left = status.query_one("#status-left", Static)
    renderable = left.render()
    return getattr(renderable, "plain", str(renderable))


def system_messages(app: AgentApp) -> list[str]:
    chat = app.query_one("#chat-view")
    messages: list[str] = []
    for widget in chat.query(Static):
        if "message-system" not in widget.classes:
            continue
        renderable = widget.render()
        messages.append(getattr(renderable, "plain", str(renderable)))
    return messages


def prompt_input(app: AgentApp) -> Input:
    prompt = app.query_one("#prompt-input", PromptInput)
    input_widget = prompt.query_one("#prompt-inner", Input)
    input_widget.focus()
    return input_widget


def render_text(widget: Static) -> str:
    renderable = widget.render()
    return getattr(renderable, "plain", str(renderable))


async def submit(app: AgentApp, pilot: Any, text: str) -> None:
    input_widget = prompt_input(app)
    input_widget.value = text
    await pilot.press("enter")
    await pilot.pause()


async def wait_for_idle(app: AgentApp, pilot: Any, ticks: int = 50) -> None:
    for _ in range(ticks):
        if not app.is_processing:
            return
        await pilot.pause()
    assert not app.is_processing


@pytest.mark.asyncio
async def test_tui_runner_autocomplete_includes_skills_templates_and_extensions(temp_dir):
    skills_dir = temp_dir / "skills"
    templates_dir = temp_dir / "templates"
    ext_path = temp_dir / "ext_ping.py"

    write_skill(skills_dir, "deploy")
    write_template(templates_dir, "build.md")
    write_extension(ext_path)

    config = Config(
        provider="openai",
        model="gpt-4o",
        api_key="test",
        session_dir=temp_dir,
        skills_dirs=[skills_dir],
        prompt_template_dirs=[templates_dir],
        extensions=[ext_path],
    )
    app = AgentApp(config, provider=LLMProviderFake([]))

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()

        input_widget = prompt_input(app)

        await pilot.press("$")
        await pilot.pause()

        prompt = app.query_one("#prompt-input", PromptInput)
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


@pytest.mark.asyncio
async def test_tui_runner_autocomplete_tab_and_history_navigation(temp_dir):
    config = Config(provider="openai", model="gpt-4o", api_key="test", session_dir=temp_dir)
    app = AgentApp(config, provider=LLMProviderFake([]))

    async with app.run_test() as pilot:
        await pilot.pause()

        input_widget = prompt_input(app)
        await pilot.press("/", "h")
        await pilot.pause()

        prompt = app.query_one("#prompt-input", PromptInput)
        option_list = prompt.query_one("#suggestions", OptionList)
        assert option_list.display

        await pilot.press("tab")
        await pilot.pause()
        assert input_widget.value == "/help"

        input_widget.value = "/help "
        await pilot.press("enter")
        await pilot.pause()
        assert any("ctrl+c quit" in msg for msg in system_messages(app))

        await submit(app, pilot, "first prompt")
        await wait_for_idle(app, pilot)

        await submit(app, pilot, "second prompt")
        await wait_for_idle(app, pilot)

        input_widget = prompt_input(app)
        input_widget.value = ""
        await pilot.press("up")
        await pilot.pause()
        assert input_widget.value == "second prompt"

        await pilot.press("up")
        await pilot.pause()
        assert input_widget.value == "first prompt"

        await pilot.press("down")
        await pilot.pause()
        assert input_widget.value == "second prompt"

        await pilot.press("down")
        await pilot.pause()
        assert input_widget.value == ""


@pytest.mark.asyncio
async def test_tui_runner_model_modal_updates_model_and_thinking(temp_dir):
    session = Session.new(temp_dir)
    config = Config(
        provider="openai",
        model="gpt-4o",
        api_key="test",
        session_dir=temp_dir,
        thinking_level=ThinkingLevel.OFF,
    )
    provider = LLMProviderFake(
        [],
        name="openai",
        model="gpt-4o",
        available_models=["gpt-4o", "gpt-5"],
    )
    app = AgentApp(config, provider=provider, session=session)

    async with app.run_test() as pilot:
        await pilot.pause()

        prompt = app.query_one("#prompt-input", PromptInput)
        input_widget = prompt.query_one("#prompt-inner", Input)
        input_widget.focus()
        # Trailing space avoids autocomplete interception on first Enter.
        input_widget.value = "/model "
        await pilot.press("enter")
        await pilot.pause()

        modal = app.screen
        assert isinstance(modal, ModelModal)

        select = modal.query_one("#model-select", Select)
        select.value = "gpt-5"
        await pilot.pause()

        radio = modal.query_one("#thinking-radio", RadioSet)
        target = None
        for button in radio.query(RadioButton):
            if button.name == "low" and not button.disabled:
                target = button
                break

        assert target is not None
        target.toggle()
        await pilot.pause()

        save = modal.query_one("#save-btn", Button)
        save.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert app.agent.provider.model == "gpt-5"
        assert app.agent.config.thinking_level == ThinkingLevel.LOW
        left = status_left_text(app)
        assert "gpt-5" in left
        assert "thinking:low" in left


@pytest.mark.asyncio
async def test_tui_runner_model_command_switches_and_rejects_invalid_model(temp_dir):
    config = Config(provider="openai", model="gpt-4o", api_key="test", session_dir=temp_dir)
    provider = LLMProviderFake(
        [],
        name="openai",
        model="gpt-4o",
        available_models=["gpt-4o", "gpt-5"],
    )
    app = AgentApp(config, provider=provider)

    async with app.run_test() as pilot:
        await pilot.pause()

        await submit(app, pilot, "/model gpt-5")
        assert app.agent.provider.model == "gpt-5"
        assert any("switched to gpt-5" in msg for msg in system_messages(app))
        assert "gpt-5" in status_left_text(app)

        await submit(app, pilot, "/model claude-sonnet-4-5")
        assert app.agent.provider.model == "gpt-5"
        assert any("not valid for provider 'openai'" in msg for msg in system_messages(app))
        assert "gpt-5" in status_left_text(app)


@pytest.mark.asyncio
async def test_tui_runner_rehydrates_tool_and_thinking_widgets(temp_dir):
    session = Session.new(temp_dir)

    tool_call = ToolCall(id="call_1", name="read", arguments={"path": "file.txt"})
    session.append(Message.user("Read"))
    session.append(
        Message.assistant(
            "Here you go",
            tool_calls=[tool_call],
            thinking=ThinkingContent(text="thinking..."),
        )
    )
    session.append(Message.tool_result("call_1", "result"))

    config = Config(
        provider="openai",
        model="gpt-4o",
        api_key="test",
        session_dir=temp_dir,
        context_max_tokens=2048,
        max_output_tokens=2048,
    )

    app = AgentApp(
        config,
        provider=LLMProviderFake([], name="openai", model="gpt-4o"),
        session=session,
    )
    async with app.run_test() as pilot:
        await pilot.pause()

        chat = app.query_one("#chat-view")
        tool_widgets = list(chat.query(ToolWidget))
        thinking_widgets = list(chat.query(ThinkingWidget))
        assistant_widgets = list(chat.query(MessageWidget))

        assert tool_widgets
        assert thinking_widgets
        assert any("Here you go" in w.text_content() for w in assistant_widgets)


@pytest.mark.asyncio
async def test_tui_runner_session_commands_report_errors_for_bad_targets(temp_dir):
    missing_session_file = temp_dir / "missing.jsonl"

    config = Config(provider="openai", model="gpt-4o", api_key="test", session_dir=temp_dir)
    app = AgentApp(config, provider=LLMProviderFake([]))

    async with app.run_test() as pilot:
        await pilot.pause()

        await submit(app, pilot, f"/load {missing_session_file}")
        assert any("session file not found:" in msg for msg in system_messages(app))

        await submit(app, pilot, "/fork not-a-message")
        assert any(
            "could not resolve message: not-a-message" in msg for msg in system_messages(app)
        )

        await submit(app, pilot, "/tree not-an-entry")
        assert any("could not resolve entry: not-an-entry" in msg for msg in system_messages(app))


@pytest.mark.asyncio
async def test_tui_runner_help_new_and_clear_commands_update_runtime_state(temp_dir):
    config = Config(provider="openai", model="gpt-4o", api_key="test", session_dir=temp_dir)
    app = AgentApp(config, provider=LLMProviderFake([]))

    async with app.run_test() as pilot:
        await pilot.pause()

        initial_session_id = app.agent.session.metadata.id

        await submit(app, pilot, "/help ")
        assert any("ctrl+c quit" in msg for msg in system_messages(app))

        await submit(app, pilot, "/new ")
        new_session_id = app.agent.session.metadata.id
        assert new_session_id != initial_session_id
        assert any("new session started" in msg for msg in system_messages(app))
        assert f"session:{new_session_id}" in status_left_text(app)

        await submit(app, pilot, "hello")
        await wait_for_idle(app, pilot)

        await submit(app, pilot, "/clear ")
        assert any("cleared" in msg for msg in system_messages(app))

        chat = app.query_one("#chat-view")
        user_messages = [w for w in chat.query(MessageWidget) if w.role == "user"]
        assert not user_messages


@pytest.mark.asyncio
async def test_tui_runner_escape_interrupts_active_response(temp_dir):
    config = Config(provider="openai", model="gpt-4o", api_key="test", session_dir=temp_dir)
    provider = CancelAwareProviderFake()
    app = AgentApp(config, provider=provider)

    async with app.run_test() as pilot:
        await pilot.pause()

        await submit(app, pilot, "interrupt me")

        for _ in range(20):
            if app.is_processing and provider.started.is_set():
                break
            await pilot.pause()
        assert app.is_processing

        await pilot.press("escape")
        await pilot.pause()
        await wait_for_idle(app, pilot)

        assert any("interrupted" in msg for msg in system_messages(app))
        assert not app.is_processing


@pytest.mark.asyncio
async def test_tui_runner_skill_command_renders_skill_block_and_args(temp_dir):
    skills_dir = temp_dir / "skills"
    write_skill(skills_dir, "deploy")

    config = Config(
        provider="openai",
        model="gpt-4o",
        api_key="test",
        session_dir=temp_dir,
        skills_dirs=[skills_dir],
    )
    app = AgentApp(config, provider=LLMProviderFake([]))

    async with app.run_test() as pilot:
        await pilot.pause()

        await submit(app, pilot, "$deploy --prod")
        await wait_for_idle(app, pilot)

        chat = app.query_one("#chat-view")
        skill_widgets = list(chat.query(SkillInvocationWidget))
        user_messages = [w for w in chat.query(MessageWidget) if w.role == "user"]

        assert skill_widgets
        assert skill_widgets[0].skill_name == "deploy"
        assert any(w.text_content() == "--prod" for w in user_messages)


@pytest.mark.asyncio
async def test_tui_runner_status_displays_model_thinking_and_session(temp_dir):
    session = Session.new(temp_dir)
    config = Config(
        provider="openai",
        model="gpt-4o",
        api_key="test",
        session_dir=temp_dir,
        thinking_level=ThinkingLevel.LOW,
    )
    app = AgentApp(
        config,
        provider=LLMProviderFake([], name="openai", model="gpt-4o"),
        session=session,
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        left = status_left_text(app)
        assert "gpt-4o" in left
        assert "thinking:low" in left
        assert f"session:{session.metadata.id}" in left


@pytest.mark.asyncio
async def test_tui_runner_context_command_opens_and_closes_modal(temp_dir):
    skills_dir = temp_dir / "skills"
    templates_dir = temp_dir / "templates"
    write_skill(skills_dir, "deploy")
    write_template(templates_dir, "build.md")

    session = Session.new(temp_dir)
    session.append(Message.user("hello"))
    session.append(Message.assistant("world"))

    config = Config(
        provider="openai",
        model="gpt-4o",
        api_key="test",
        session_dir=temp_dir,
        skills_dirs=[skills_dir],
        prompt_template_dirs=[templates_dir],
    )
    app = AgentApp(
        config,
        provider=LLMProviderFake([], name="openai", model="gpt-4o"),
        session=session,
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        await submit(app, pilot, "/context ")
        modal = app.screen
        assert isinstance(modal, ContextModal)

        summary_text = render_text(modal.query_one("#summary-content", Static))
        assert "Provider: openai" in summary_text
        assert "Model: gpt-4o" in summary_text
        assert "MESSAGES" in summary_text
        assert "SKILLS" in summary_text
        assert "TEMPLATES" in summary_text

        messages_text = render_text(modal.query_one("#messages-content", Static))
        assert "USER" in messages_text
        assert "ASSISTANT" in messages_text
        assert "Total:" in messages_text

        system_text = render_text(modal.query_one("#system-content", Static))
        assert "Total:" in system_text or "(no system prompt)" in system_text

        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ContextModal)


@pytest.mark.asyncio
async def test_tui_runner_load_modal_selects_session_via_public_flow(temp_dir):
    target = Session.new(temp_dir)
    target.append(Message.user("target-session"))

    current = Session.new(temp_dir)
    current.append(Message.user("current-session"))

    config = Config(provider="openai", model="gpt-4o", api_key="test", session_dir=temp_dir)
    app = AgentApp(config, provider=LLMProviderFake([]), session=current)

    async with app.run_test() as pilot:
        await pilot.pause()

        await submit(app, pilot, "/load ")
        modal = app.screen
        assert isinstance(modal, SessionLoadModal)

        list_view = modal.query_one("#session-list", ListView)
        selected_item = list_view.children[list_view.index]
        selected_path = getattr(selected_item, "name", None)
        assert selected_path is not None

        list_view.action_select_cursor()
        await pilot.pause()

        assert str(app.agent.session.path) == selected_path
        assert any("loaded session" in msg for msg in system_messages(app))
        assert not isinstance(app.screen, SessionLoadModal)


@pytest.mark.asyncio
async def test_tui_runner_fork_modal_forks_from_selected_message(temp_dir):
    session = Session.new(temp_dir)
    session.append(Message.user("first"))
    session.append(Message.assistant("second"))
    session.append(Message.user("third"))

    parent_session_id = session.metadata.id
    config = Config(provider="openai", model="gpt-4o", api_key="test", session_dir=temp_dir)
    app = AgentApp(config, provider=LLMProviderFake([]), session=session)

    async with app.run_test() as pilot:
        await pilot.pause()

        await submit(app, pilot, "/fork ")
        modal = app.screen
        assert isinstance(modal, SessionForkModal)

        list_view = modal.query_one("#fork-list", ListView)
        while list_view.index > 0:
            list_view.action_cursor_up()
            await pilot.pause()
        selected_item = list_view.children[list_view.index]
        selected_message_id = getattr(selected_item, "name", None)
        assert selected_message_id is not None

        list_view.action_select_cursor()
        await pilot.pause()

        assert app.agent.session.metadata.parent_session_id == parent_session_id
        assert len(app.agent.session.messages) == 1
        assert any(
            f"forked from {parent_session_id} at {selected_message_id}" in msg
            for msg in system_messages(app)
        )
        assert not isinstance(app.screen, SessionForkModal)


@pytest.mark.asyncio
async def test_tui_runner_tree_modal_linear_list_updates_leaf(temp_dir):
    session = Session.new(temp_dir)
    session.append(Message.user("first"))
    session.append(Message.assistant("second"))
    session.append(Message.user("third"))

    config = Config(provider="openai", model="gpt-4o", api_key="test", session_dir=temp_dir)
    app = AgentApp(config, provider=LLMProviderFake([]), session=session)

    async with app.run_test() as pilot:
        await pilot.pause()

        await submit(app, pilot, "/tree ")
        modal = app.screen
        assert isinstance(modal, SessionTreeModal)

        list_view = modal.query_one("#linear-list", ListView)
        while list_view.index > 1:
            list_view.action_cursor_up()
            await pilot.pause()
        selected_item = list_view.children[list_view.index]
        selected_entry_id = getattr(selected_item, "name", None)
        assert selected_entry_id is not None

        list_view.action_select_cursor()
        await pilot.pause()

        assert app.agent.session.leaf_id == selected_entry_id
        assert len(app.agent.session.messages) == 2
        assert any(f"branched to {selected_entry_id}" in msg for msg in system_messages(app))
        assert not isinstance(app.screen, SessionTreeModal)


@pytest.mark.asyncio
async def test_tui_runner_tree_modal_branch_view_updates_leaf(temp_dir):
    session = Session.new(temp_dir)
    first = Message.user("first")
    second = Message.assistant("second")
    third = Message.user("third")
    branch = Message.assistant("branch")
    session.append(first)
    session.append(second)
    session.append(third)
    session.set_leaf(second.id)
    session.append(branch)

    current_leaf = session.leaf_id
    config = Config(provider="openai", model="gpt-4o", api_key="test", session_dir=temp_dir)
    app = AgentApp(config, provider=LLMProviderFake([]), session=session)

    async with app.run_test() as pilot:
        await pilot.pause()

        await submit(app, pilot, "/tree ")
        modal = app.screen
        assert isinstance(modal, SessionTreeModal)

        tree = modal.query_one("#tree-view", Tree)
        tree.action_cursor_parent()
        await pilot.pause()

        cursor_node = tree.cursor_node
        assert cursor_node is not None
        target_entry_id = str(cursor_node.data)
        if target_entry_id == str(current_leaf):
            tree.action_cursor_up()
            await pilot.pause()
            cursor_node = tree.cursor_node
            assert cursor_node is not None
            target_entry_id = str(cursor_node.data)

        assert target_entry_id != str(current_leaf)

        tree.action_select_cursor()
        await pilot.pause()

        assert app.agent.session.leaf_id == target_entry_id
        assert any(f"branched to {target_entry_id}" in msg for msg in system_messages(app))
        assert not isinstance(app.screen, SessionTreeModal)
