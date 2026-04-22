"""Minimal stdio MCP echo server for exercising the MCP adapter example."""

from __future__ import annotations

import json
import sys
from typing import Any

PROTOCOL_VERSION = "2025-11-25"


def read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in {b"\r\n", b"\n"}:
            break
        name, _, value = line.decode("ascii", errors="replace").partition(":")
        headers[name.lower()] = value.strip()

    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    payload = sys.stdin.buffer.read(length)
    data = json.loads(payload.decode("utf-8"))
    return data if isinstance(data, dict) else None


def send_message(payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def handle_initialize(message: dict[str, Any]) -> None:
    params = message.get("params", {})
    protocol_version = PROTOCOL_VERSION
    if isinstance(params, dict):
        protocol_version = str(params.get("protocolVersion") or PROTOCOL_VERSION)
    send_message(
        {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "echo-stdio-server", "version": "1.0.0"},
            },
        }
    )


def handle_tools_list(message: dict[str, Any]) -> None:
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
                                    "description": "Text to echo back.",
                                }
                            },
                            "required": ["text"],
                        },
                    }
                ]
            },
        }
    )


def handle_tools_call(message: dict[str, Any]) -> None:
    params = message.get("params", {})
    arguments: dict[str, Any] = {}
    if isinstance(params, dict) and isinstance(params.get("arguments"), dict):
        arguments = params["arguments"]
    text = str(arguments.get("text") or "")
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


def main() -> None:
    while True:
        message = read_message()
        if message is None:
            break

        method = message.get("method")
        if method == "initialize":
            handle_initialize(message)
        elif method == "tools/list":
            handle_tools_list(message)
        elif method == "tools/call":
            handle_tools_call(message)
        elif "id" in message:
            send_message({"jsonrpc": "2.0", "id": message["id"], "result": {}})


if __name__ == "__main__":
    main()
