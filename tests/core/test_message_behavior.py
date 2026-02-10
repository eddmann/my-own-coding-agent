"""Behavior tests for message and tool call objects."""

from agent.core.message import Message, Role, ToolCall


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
