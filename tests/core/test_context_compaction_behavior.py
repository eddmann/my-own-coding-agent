"""Behavior tests for context compaction."""

from __future__ import annotations

import pytest

from agent.core.context import ContextManager
from agent.core.message import Message, ToolCall
from tests.test_doubles.llm_provider_fake import LLMProviderFake
from tests.test_doubles.llm_stream_builders import make_error_events, make_text_events


class CountingProviderFake(LLMProviderFake):
    """Provider double with deterministic message token counts."""

    def __init__(self, scripts, *, token_count: int):
        super().__init__(scripts)
        self._token_count = token_count

    def count_messages_tokens(self, messages):
        return self._token_count


@pytest.mark.asyncio
async def test_compaction_uses_structured_summary_and_redacts_prompt_input(temp_dir):
    MAX_TOKENS = 4096
    KEEP_RECENT = 2

    fake = LLMProviderFake(
        [
            make_text_events(
                "## Summary\n- ok\n\n## Decisions\n- none\n\n## Files Read\n- a.txt\n\n"
                "## Files Modified\n- b.txt\n\n## Commands Run\n- none\n\n"
                "## Tools Used\n- read\n\n## Open TODOs\n- none\n\n"
                "## Risks/Concerns\n- none\n"
            )
        ]
    )
    manager = ContextManager(
        fake,
        max_tokens=MAX_TOKENS,
        keep_recent=KEEP_RECENT,
        summarization_provider=fake,
    )

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

    compaction = await manager.compact(messages)

    assert "## Summary" in compaction.summary
    assert compaction.first_kept_id == messages[-2].id

    prompt_sent = fake.stream_calls[0]["messages"][0].content
    assert secret not in prompt_sent
    assert key_secret not in prompt_sent
    assert "[REDACTED]" in prompt_sent


@pytest.mark.asyncio
async def test_compaction_fallback_extracts_files_and_redacts(temp_dir):
    MAX_TOKENS = 4096
    KEEP_RECENT = 2

    fake = LLMProviderFake([make_error_events("failure")])
    manager = ContextManager(
        fake,
        max_tokens=MAX_TOKENS,
        keep_recent=KEEP_RECENT,
        summarization_provider=fake,
    )

    tool_calls = [ToolCall(id="1", name="read", arguments={"path": "/tmp/sk-abcdef123456.txt"})]
    messages = [
        Message.system("System"),
        Message.user("Check file"),
        Message.assistant("Working", tool_calls=tool_calls),
        Message.user("token: sk-abcdef"),
        Message.user("password: hunter2"),
        Message.assistant("done"),
    ]

    compaction = await manager.compact(messages)

    content = compaction.summary
    assert "## Files Read" in content
    assert "sk-" not in content
    assert "hunter2" not in content
    assert "[REDACTED]" in content


@pytest.mark.asyncio
async def test_compaction_uses_summarization_provider_and_primary_for_token_counting(temp_dir):
    MAX_TOKENS = 4096
    KEEP_RECENT = 2
    PRIMARY_TOKENS = 111
    SUMMARIZER_TOKENS = 999

    primary = CountingProviderFake(
        [make_text_events("primary should not summarize")],
        token_count=PRIMARY_TOKENS,
    )
    summarizer = CountingProviderFake(
        [
            make_text_events(
                "## Summary\n- from summarizer\n\n## Decisions\n- none\n\n## Files Read\n- none\n\n"
                "## Files Modified\n- none\n\n## Commands Run\n- none\n\n## Tools Used\n- none\n\n"
                "## Open TODOs\n- none\n\n## Risks/Concerns\n- none\n"
            )
        ],
        token_count=SUMMARIZER_TOKENS,
    )
    manager = ContextManager(
        primary,
        max_tokens=MAX_TOKENS,
        keep_recent=KEEP_RECENT,
        summarization_provider=summarizer,
    )

    messages = [
        Message.system("System"),
        Message.user("u1"),
        Message.assistant("a1"),
        Message.user("u2"),
        Message.assistant("a2"),
        Message.user("u3"),
    ]

    compaction = await manager.compact(messages)

    assert "from summarizer" in compaction.summary
    assert manager.current_tokens(messages) == PRIMARY_TOKENS
    assert len(summarizer.stream_calls) == 1
    assert len(primary.stream_calls) == 0
