"""Behavior tests for the example MCP adapter extension."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent.config import Config
from agent.extensions.api import ExtensionUIBindings
from agent.extensions.host import ExtensionHost
from agent.runtime.agent import Agent
from tests.test_doubles.llm_provider_fake import LLMProviderFake
from tests.test_doubles.llm_stream_builders import make_text_events, make_tool_call_events

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
    return REPO_ROOT / "examples" / "extensions" / "mcp-adapter.py"


async def run_command(agent: Agent, prompt: str) -> list[str]:
    outputs: list[str] = []
    async for chunk in agent.run(prompt):
        if chunk.type == "message" and chunk.payload.role.value == "system":
            outputs.append(chunk.payload.content)
    return outputs


def write_fake_mcp_server(path: Path) -> None:
    path.write_text(
        """\
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in {b"\\r\\n", b"\\n"}:
            break
        name, _, value = line.decode("ascii").partition(":")
        headers[name.lower()] = value.strip()

    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    payload = sys.stdin.buffer.read(length)
    return json.loads(payload.decode("utf-8"))


def send_message(payload):
    body = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\\r\\n\\r\\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    if message is None:
        break

    method = message.get("method")
    if method == "initialize":
        send_message(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "protocolVersion": message.get("params", {}).get(
                        "protocolVersion",
                        "2025-11-25",
                    ),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-mcp", "version": "1.0.0"},
                },
            }
        )
    elif method == "tools/list":
        send_message(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo text back to the caller.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "string",
                                        "description": "text to echo",
                                    }
                                },
                                "required": ["text"],
                            },
                        }
                    ]
                },
            }
        )
    elif method == "tools/call":
        arguments = message.get("params", {}).get("arguments", {})
        text = arguments.get("text", "")
        send_message(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"echo: {text}",
                        }
                    ]
                },
            }
        )
    elif "id" in message:
        send_message({"jsonrpc": "2.0", "id": message["id"], "result": {}})
"""
    )


def write_mcp_config(
    temp_dir: Path, server_script: Path, *, direct_tools: bool | list[str] = False
) -> None:
    config_dir = temp_dir / ".agent"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mcpServers": {
            "demo": {
                "command": sys.executable,
                "args": [str(server_script)],
                "directTools": direct_tools,
            }
        }
    }
    (config_dir / "mcp.json").write_text(json.dumps(payload, indent=2) + "\n")


@pytest.mark.asyncio
async def test_mcp_adapter_command_shows_status_and_connects_server(temp_dir):
    server_script = temp_dir / "fake_mcp_server.py"
    write_fake_mcp_server(server_script)
    write_mcp_config(temp_dir, server_script)

    agent, _, host = build_agent(temp_dir, [])
    try:
        await host.load_extensions([extension_path()])

        status_messages = await run_command(agent, "/mcp")
        assert status_messages
        assert "MCP Servers" in status_messages[0]
        assert "- demo [idle] 0 cached tools" in status_messages[0]

        connect_messages = await run_command(agent, "/mcp connect demo")
        assert connect_messages
        assert "Tools for demo" in connect_messages[0]
        assert "demo_echo" in connect_messages[0]

        search_messages = await run_command(agent, "/mcp search echo")
        assert search_messages
        assert "demo_echo" in search_messages[0]
        assert "text to echo" in search_messages[0]
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_mcp_adapter_no_args_uses_presented_browser_view(temp_dir):
    server_script = temp_dir / "fake_mcp_server.py"
    write_fake_mcp_server(server_script)
    write_mcp_config(temp_dir, server_script)

    agent, _, host = build_agent(temp_dir, [])
    presented: list[str] = []
    host.bind_ui(
        ExtensionUIBindings(
            present=lambda view: (
                presented.append(view.render()),
                view.handle_action("command", "status"),
                None,
            )[-1]
        )
    )

    try:
        await host.load_extensions([extension_path()])

        messages = await run_command(agent, "/mcp")
        assert messages
        assert "Closed MCP browser" in messages[0]
        assert presented
        assert "MCP Browser" in presented[0]
        assert "Servers" in presented[0]
        assert "demo [idle]" in presented[0]
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_mcp_adapter_proxy_tool_lazy_connects_and_calls_server(temp_dir):
    server_script = temp_dir / "fake_mcp_server.py"
    write_fake_mcp_server(server_script)
    write_mcp_config(temp_dir, server_script)

    agent, provider, host = build_agent(
        temp_dir,
        [
            make_tool_call_events(
                "call_1",
                "mcp",
                {
                    "tool": "demo_echo",
                    "args": '{"text": "hello"}',
                },
            ),
            make_text_events("done"),
        ],
    )
    try:
        await host.load_extensions([extension_path()])

        async for _ in agent.run("use the mcp echo tool"):
            pass

        tool_messages = [
            message for message in agent.session.messages if message.role.value == "tool"
        ]
        assert tool_messages
        assert "echo: hello" in tool_messages[-1].content

        cache_path = temp_dir / ".agent" / "mcp-cache.json"
        assert cache_path.exists()
        cache_payload = json.loads(cache_path.read_text())
        assert cache_payload["servers"]["demo"][0]["name"] == "echo"

        tool_names = [schema["function"]["name"] for schema in provider.stream_calls[0]["tools"]]
        assert "mcp" in tool_names
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_mcp_adapter_can_promote_direct_tools_when_configured(temp_dir):
    server_script = temp_dir / "fake_mcp_server.py"
    write_fake_mcp_server(server_script)
    write_mcp_config(temp_dir, server_script, direct_tools=True)

    agent, provider, host = build_agent(temp_dir, [make_text_events("done")])
    try:
        await host.load_extensions([extension_path()])

        connect_messages = await run_command(agent, "/mcp connect demo")
        assert connect_messages
        assert "demo_echo" in connect_messages[0]
        assert "demo_echo" in agent.tools

        result = await agent.tools.execute("demo_echo", {"text": "world"})
        assert result.is_error is False
        assert result.content == "echo: world"

        async for _ in agent.run("use the direct mcp tool"):
            pass

        tool_names = [schema["function"]["name"] for schema in provider.stream_calls[0]["tools"]]
        assert "mcp" in tool_names
        assert "demo_echo" in tool_names
    finally:
        await agent.close()
