"""Behavior tests for context compaction."""

from __future__ import annotations

import pytest

from agent.core.context import ContextManager
from agent.core.message import Message, ToolCall
from tests.fakes import FakeLLMProvider, make_error_events, make_text_events


@pytest.mark.asyncio
async def test_compaction_uses_structured_summary_and_redacts_prompt_input(temp_dir):
    fake = FakeLLMProvider(
        [
            make_text_events(
                "## Summary\n- ok\n\n## Decisions\n- none\n\n## Files Read\n- a.txt\n\n"
                "## Files Modified\n- b.txt\n\n## Commands Run\n- none\n\n"
                "## Tools Used\n- read\n\n## Open TODOs\n- none\n\n"
                "## Risks/Concerns\n- none\n"
            )
        ]
    )
    manager = ContextManager(fake, max_tokens=4096, keep_recent=2)

    secret = "sk-1234567890secret"
    key_secret = "api_key: abc123"
    messages = [
        Message.system("System rules"),
        Message.user(f"my key is {secret}"),
        Message.user(f"credentials {key_secret}"),
        Message.assistant("Noted"),
        Message.user("next"),
        Message.assistant("ok"),
    ]

    compacted = await manager.compact(messages)

    summary_messages = [
        m for m in compacted if m.role.value == "system" and "Previous conversation" in m.content
    ]
    assert summary_messages
    assert "## Summary" in summary_messages[0].content

    prompt_sent = fake.stream_calls[0]["messages"][0].content
    assert secret not in prompt_sent
    assert key_secret not in prompt_sent
    assert "[REDACTED]" in prompt_sent


@pytest.mark.asyncio
async def test_compaction_fallback_extracts_files_and_redacts(temp_dir):
    fake = FakeLLMProvider([make_error_events("failure")])
    manager = ContextManager(fake, max_tokens=4096, keep_recent=2)

    tool_calls = [ToolCall(id="1", name="read", arguments={"path": "/tmp/sk-abcdef123456.txt"})]
    messages = [
        Message.system("System"),
        Message.user("Check file"),
        Message.assistant("Working", tool_calls=tool_calls),
        Message.user("token: sk-abcdef"),
        Message.user("password: hunter2"),
        Message.assistant("done"),
    ]

    compacted = await manager.compact(messages)

    summary_messages = [
        m for m in compacted if m.role.value == "system" and "Previous conversation" in m.content
    ]
    assert summary_messages
    content = summary_messages[0].content
    assert "## Files Read" in content
    assert "sk-" not in content
    assert "hunter2" not in content
    assert "[REDACTED]" in content
