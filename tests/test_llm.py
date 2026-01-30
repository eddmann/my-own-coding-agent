"""Tests for the LLM module."""

from agent.core.message import Message, Role, ToolCall
from agent.llm import OpenAICompatibleProvider


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
