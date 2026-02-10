"""Behavior tests for TUI session rehydration."""

from __future__ import annotations

import pytest

from agent.core.config import Config
from agent.core.message import Message, ThinkingContent, ToolCall
from agent.core.session import Session
from agent.tui.app import AgentApp
from agent.tui.chat import MessageWidget, ThinkingWidget, ToolWidget


@pytest.mark.asyncio
async def test_tui_rehydrates_tool_and_thinking_widgets(temp_dir):
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
        model="fake-model",
        api_key="test",
        session_dir=temp_dir,
        context_max_tokens=2048,
        max_output_tokens=2048,
    )

    app = AgentApp(config, session)
    async with app.run_test() as pilot:
        await pilot.pause()

        chat = app.query_one("#chat-view")
        tool_widgets = list(chat.query(ToolWidget))
        thinking_widgets = list(chat.query(ThinkingWidget))
        assistant_widgets = list(chat.query(MessageWidget))

        assert tool_widgets
        assert thinking_widgets

        assert any("Here you go" in w.text_content() for w in assistant_widgets)
