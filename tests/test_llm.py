"""Tests for the LLM module."""

import json

from agent.core.message import Message, Role, ThinkingContent, ToolCall
from agent.llm import OpenAICompatibleProvider
from agent.llm.openai import OpenAIResponsesProvider


class TestOpenAICompatibleProvider:
    """Tests for OpenAICompatibleProvider."""

    def test_provider_creation(self):
        """Test creating a provider."""
        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com",
            api_key="test-key",
            model="gpt-4o",
        )

        assert provider.base_url == "https://api.openai.com"
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4o"

    def test_count_tokens(self):
        """Test token counting."""
        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com",
            api_key="test",
            model="gpt-4o",
        )

        # Simple text should have some tokens
        count = provider.count_tokens("Hello, world!")
        assert count > 0
        assert count < 10  # Should be relatively small

        # Longer text should have more tokens
        long_count = provider.count_tokens("This is a longer piece of text " * 10)
        assert long_count > count

    def test_count_messages_tokens(self):
        """Test counting tokens in messages."""
        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com",
            api_key="test",
            model="gpt-4o",
        )

        messages = [
            Message.system("You are a helpful assistant."),
            Message.user("Hello!"),
            Message.assistant("Hi there! How can I help you today?"),
        ]

        count = provider.count_messages_tokens(messages)
        assert count > 0


class TestAnthropicProvider:
    """Tests for AnthropicProvider behavior."""

    def test_system_prompt_concatenates_multiple_messages(self):
        """Ensure multiple system messages are preserved."""
        from agent.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test", model="claude-sonnet-4-20250514")
        messages = [
            Message.system("System A"),
            Message.user("Hi"),
            Message.system("System B"),
            Message.assistant("Hello"),
        ]

        system_prompt, api_messages = provider._convert_messages(messages)

        assert "System A" in system_prompt
        assert "System B" in system_prompt
        assert system_prompt.index("System A") < system_prompt.index("System B")
        # Ensure non-system messages still passed through
        assert any(m["role"] == "user" for m in api_messages)

    def test_thinking_signature_is_included_from_metadata(self):
        """Ensure thinking signatures are sourced from provider metadata."""
        from agent.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test", model="claude-sonnet-4-20250514")
        msg = Message(
            role=Role.ASSISTANT,
            content="Hello",
            thinking=ThinkingContent(text="thoughts"),
            provider_metadata={"anthropic": {"thinking_signature": "sig_123"}},
        )

        content = provider._build_assistant_content(msg)

        assert isinstance(content, list)
        thinking_block = next(block for block in content if block["type"] == "thinking")
        assert thinking_block["signature"] == "sig_123"


class TestOpenAIResponsesProvider:
    """Tests for OpenAI Responses provider conversion."""

    def test_replays_reasoning_and_output_item_metadata(self):
        """Ensure provider metadata is converted into responses input items."""
        provider = OpenAIResponsesProvider(
            api_key="test-key",
            model="gpt-5.2-codex",
        )
        reasoning_item = {"type": "reasoning", "summary": [{"text": "r"}]}
        assistant_msg = Message(
            role=Role.ASSISTANT,
            content="Hello",
            provider_metadata={
                "openai_responses": {
                    "output_item_id": "msg_123",
                    "reasoning_item": json.dumps(reasoning_item),
                }
            },
        )
        messages = [
            Message.system("System"),
            Message.user("Hi"),
            assistant_msg,
        ]

        output = provider._convert_responses_messages(messages)

        assistant_items = [item for item in output if item.get("type") in ("reasoning", "message")]
        assert assistant_items[0]["type"] == "reasoning"
        assert assistant_items[1]["type"] == "message"
        assert assistant_items[1]["id"] == "msg_123"

    def test_replays_tool_call_item_id(self):
        """Ensure tool call item ids are preserved for replay."""
        provider = OpenAIResponsesProvider(
            api_key="test-key",
            model="gpt-5.2-codex",
        )
        tool_call = ToolCall(id="call_123|fc_abc", name="read", arguments={"path": "/tmp/x"})
        assistant_msg = Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[tool_call],
            model="gpt-5.2-codex",
        )
        messages = [
            Message.system("System"),
            Message.user("Hi"),
            assistant_msg,
        ]

        output = provider._convert_responses_messages(messages)

        function_calls = [item for item in output if item.get("type") == "function_call"]
        assert function_calls
        assert function_calls[0]["call_id"] == "call_123"
        assert function_calls[0]["id"] == "fc_abc"


class TestToolCall:
    """Tests for ToolCall parsing."""

    def test_from_api_format(self):
        """Test parsing from OpenAI API format."""
        api_data = {
            "id": "call_123",
            "function": {
                "name": "read",
                "arguments": '{"path": "/test/file.txt"}',
            },
        }

        tool_call = ToolCall.from_api(api_data)

        assert tool_call.id == "call_123"
        assert tool_call.name == "read"
        assert tool_call.arguments == {"path": "/test/file.txt"}

    def test_to_api_dict(self):
        """Test converting to OpenAI API format."""
        tool_call = ToolCall(
            id="call_456",
            name="write",
            arguments={"path": "/test/file.txt", "content": "hello"},
        )

        api_dict = tool_call.to_api_dict()

        assert api_dict["id"] == "call_456"
        assert api_dict["type"] == "function"
        assert api_dict["function"]["name"] == "write"
        assert '"path"' in api_dict["function"]["arguments"]


class TestMessage:
    """Tests for Message class."""

    def test_user_message(self):
        """Test creating a user message."""
        msg = Message.user("Hello")

        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.tool_calls is None

    def test_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message.assistant("Hi there")

        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there"

    def test_assistant_message_with_tool_calls(self):
        """Test creating an assistant message with tool calls."""
        tool_calls = [
            ToolCall(id="1", name="read", arguments={"path": "/test"}),
        ]
        msg = Message.assistant("Let me read that file.", tool_calls)

        assert msg.role == Role.ASSISTANT
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "read"

    def test_system_message(self):
        """Test creating a system message."""
        msg = Message.system("You are helpful.")

        assert msg.role == Role.SYSTEM
        assert msg.content == "You are helpful."

    def test_tool_result_message(self):
        """Test creating a tool result message."""
        msg = Message.tool_result("call_123", "File contents here")

        assert msg.role == Role.TOOL
        assert msg.content == "File contents here"
        assert msg.tool_call_id == "call_123"

    def test_to_api_dict(self):
        """Test converting message to API format."""
        msg = Message.user("Hello")
        api_dict = msg.to_api_dict()

        assert api_dict["role"] == "user"
        assert api_dict["content"] == "Hello"

    def test_to_api_dict_with_tool_calls(self):
        """Test converting message with tool calls to API format."""
        tool_calls = [
            ToolCall(id="1", name="read", arguments={"path": "/test"}),
        ]
        msg = Message.assistant("Reading...", tool_calls)
        api_dict = msg.to_api_dict()

        assert "tool_calls" in api_dict
        assert len(api_dict["tool_calls"]) == 1
