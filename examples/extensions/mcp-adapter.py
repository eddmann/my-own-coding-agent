from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, create_model

from agent.extensions import ExtensionAPI, PresentedView, ViewControl, WidgetView
from agent.tools.base import BaseTool, ToolError

if TYPE_CHECKING:
    from collections.abc import Callable

PROTOCOL_VERSION = "2025-11-25"
DEFAULT_CONFIG_PATH = Path(".agent/mcp.json")
DEFAULT_CACHE_PATH = Path(".agent/mcp-cache.json")


@dataclass(slots=True)
class McpToolSpec:
    server_name: str
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def prefixed_name(self) -> str:
        return f"{_normalize_name(self.server_name)}_{_normalize_name(self.name)}"


@dataclass(slots=True)
class McpServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    direct_tools: bool | list[str] = False


@dataclass(slots=True)
class McpServerState:
    config: McpServerConfig
    client: McpStdioClient | None = None
    connected: bool = False
    last_error: str | None = None


class McpStatusWidget(WidgetView):
    def __init__(self, manager: McpManager) -> None:
        self._manager = manager

    def render(self) -> str:
        config = self._manager.load_config()
        if not config:
            return "MCP unavailable"

        connected = sum(1 for state in self._manager.server_states.values() if state.connected)
        total = len(config)
        lines = [f"MCP {connected}/{total} connected"]
        for name in sorted(config):
            state = self._manager.server_states.get(name)
            status = "connected" if state and state.connected else "idle"
            cached = len(self._manager.cache.get(name, []))
            lines.append(f"- {name} [{status}] {cached} cached")
        return "\n".join(lines)


class McpBrowserView(PresentedView[None]):
    def __init__(
        self, manager: McpManager, *, sync_direct_tools: Callable[[], int] | None = None
    ) -> None:
        self._manager = manager
        self._sync_direct_tools = sync_direct_tools
        self._selected_server: str | None = None
        self._last_output = manager.status_text()
        self._busy = False
        self._closed = False
        self._task: asyncio.Task[None] | None = None

    def render(self) -> str:
        configs = self._manager.load_config()
        lines = ["MCP Browser", ""]

        if configs:
            lines.append("Servers")
            for name in sorted(configs):
                state = self._manager.server_states.get(name)
                status = "connected" if state and state.connected else "idle"
                cached = len(self._manager.cache.get(name, []))
                marker = "*" if name == self._selected_server else " "
                lines.append(f"{marker} {name} [{status}] {cached} cached")
        else:
            lines.append(f"No MCP servers configured. Create {self._manager.config_path}")

        lines.extend(["", "Output", self._last_output or "(none)"])
        if self._busy:
            lines.extend(["", "Working..."])
        lines.extend(
            [
                "",
                "Commands",
                "  server <name>",
                "  connect <name>",
                "  tools [name]",
                "  search <query>",
                "  describe <tool>",
                "  status",
                "  close",
            ]
        )
        return "\n".join(lines)

    def controls(self) -> list[ViewControl]:
        return [
            ViewControl(
                kind="input",
                name="command",
                label="Command",
                placeholder="Type a browser command and press Enter",
            ),
            ViewControl(kind="button", name="close", label="Close"),
        ]

    def handle_action(self, action: str, value: str | None = None) -> None:
        if action == "close":
            self._closed = True
            return
        if action != "command":
            return
        self._handle_command(value or "")

    def _handle_command(self, data: str) -> None:
        text = data.strip()
        if not text:
            return

        lower = text.lower()
        if lower in {"close", "quit", "exit", "cancel"}:
            self._closed = True
            return

        command, _, remainder = text.partition(" ")
        command = command.lower()
        remainder = remainder.strip()

        if command == "status":
            self._last_output = self._manager.status_text()
            return

        if command == "server":
            if not remainder:
                self._last_output = "Usage: server <name>"
                return
            configs = self._manager.load_config()
            if remainder not in configs:
                self._last_output = f'MCP server "{remainder}" not found'
                return
            self._selected_server = remainder
            self._last_output = self._manager._format_tool_listing(
                remainder,
                self._manager.cache.get(remainder, []),
            )
            return

        if command == "tools":
            server_name = remainder or self._selected_server
            if not server_name:
                self._last_output = "Usage: tools <server>"
                return
            self._last_output = self._manager._format_tool_listing(
                server_name,
                self._manager.cache.get(server_name, []),
            )
            return

        if command == "search":
            if not remainder:
                self._last_output = "Usage: search <query>"
                return
            self._last_output = self._manager.search(remainder, server=self._selected_server)
            return

        if command == "describe":
            if not remainder:
                self._last_output = "Usage: describe <tool>"
                return
            tool = self._manager.resolve_tool(remainder, server=self._selected_server)
            self._last_output = (
                self._manager.describe(tool)
                if tool is not None
                else self._manager._tool_not_found(remainder)
            )
            return

        if command == "connect":
            server_name = remainder or self._selected_server
            if not server_name:
                self._last_output = "Usage: connect <server>"
                return
            self._selected_server = server_name
            self._busy = True
            self._task = asyncio.create_task(self._connect(server_name))
            return

        self._last_output = (
            "Unknown command. Use: server, connect, tools, search, describe, status, close"
        )

    async def _connect(self, server_name: str) -> None:
        try:
            tools = await self._manager.connect(server_name)
            if self._sync_direct_tools is not None:
                self._sync_direct_tools()
            self._last_output = self._manager._format_tool_listing(server_name, tools)
        except Exception as exc:
            self._last_output = str(exc)
        finally:
            self._busy = False

    def is_done(self) -> bool:
        return self._closed

    def result(self) -> None:
        return None


class McpProxyParams(BaseModel):
    tool: str | None = Field(default=None, description="Tool name to call")
    args: str | None = Field(default=None, description="Tool arguments as a JSON object string")
    connect: str | None = Field(default=None, description="Server name to connect and refresh")
    describe: str | None = Field(default=None, description="Tool name to describe")
    search: str | None = Field(default=None, description="Search tools by name or description")
    server: str | None = Field(default=None, description="Filter to a specific server")


class McpProxyTool(BaseTool[McpProxyParams]):
    name = "mcp"
    description = (
        "MCP gateway for configured stdio servers. Search tools, inspect them, connect lazily, "
        "and call them through one proxy tool."
    )
    parameters = McpProxyParams

    async def execute(self, params: McpProxyParams) -> str:
        manager = MANAGERS.current()
        if manager is None:
            raise ToolError("MCP adapter is not active for the current session")
        return await manager.execute_proxy(params)


class DirectMcpTool(BaseTool[BaseModel]):
    def __init__(self, manager: McpManager, spec: McpToolSpec) -> None:
        self._manager = manager
        self._spec = spec
        self.name = spec.prefixed_name
        self.description = (
            spec.description or f"MCP tool {spec.name} from server {spec.server_name}."
        )
        self.parameters = _model_from_schema(self.name, spec.input_schema)

    async def execute(self, params: BaseModel) -> str:
        return await self._manager.call(self._spec, params.model_dump())


class McpStdioClient:
    def __init__(self, config: McpServerConfig, base_cwd: Path) -> None:
        self._config = config
        self._base_cwd = base_cwd
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_id = 1
        self._read_buffer = bytearray()

    async def connect(self) -> None:
        if self._process is not None:
            return

        cwd = self._resolve_cwd()
        env = os.environ.copy()
        env.update(_interpolate_env(self._config.env))
        self._process = await asyncio.create_subprocess_exec(
            self._config.command,
            *self._config.args,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        self._stderr_task = asyncio.create_task(self._stderr_loop())

        await self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "my-own-coding-agent",
                    "version": "0.1.0",
                },
            },
        )
        await self.notify("notifications/initialized", {})

    async def close(self) -> None:
        if self._process is None:
            return

        process = self._process
        self._process = None
        if process.stdin is not None:
            process.stdin.close()

        try:
            await asyncio.wait_for(process.wait(), timeout=1)
        except TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=1)
            except TimeoutError:
                process.kill()
                await process.wait()

        if self._reader_task is not None:
            self._reader_task.cancel()
        if self._stderr_task is not None:
            self._stderr_task.cancel()

    async def list_tools(self) -> list[McpToolSpec]:
        result = await self.request("tools/list", {})
        payload = result.get("tools", [])
        specs: list[McpToolSpec] = []
        if not isinstance(payload, list):
            return specs
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            specs.append(
                McpToolSpec(
                    server_name=self._config.name,
                    name=name,
                    description=str(item.get("description") or "").strip(),
                    input_schema=item.get("inputSchema")
                    if isinstance(item.get("inputSchema"), dict)
                    else {},
                )
            )
        return specs

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self.request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )
        is_error = bool(result.get("isError"))
        text = _stringify_tool_content(result.get("content"))
        if is_error:
            raise ToolError(text or f"MCP tool {name} failed")
        return text or "(empty result)"

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._require_process()
        request_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        await self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        try:
            response = await asyncio.wait_for(future, timeout=10)
        finally:
            self._pending.pop(request_id, None)

        if "error" in response:
            error = response["error"]
            if isinstance(error, dict):
                raise ToolError(str(error.get("message") or "MCP request failed"))
            raise ToolError(f"MCP request failed: {error}")
        result = response.get("result")
        return result if isinstance(result, dict) else {}

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )

    async def _send(self, payload: dict[str, Any]) -> None:
        process = self._require_process()
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        assert process.stdin is not None
        process.stdin.write(header + body)
        await process.stdin.drain()

    async def _reader_loop(self) -> None:
        process = self._require_process()
        assert process.stdout is not None
        try:
            while True:
                message = await self._read_message(process.stdout)
                if message is None:
                    break
                message_id = message.get("id")
                if isinstance(message_id, int):
                    future = self._pending.get(message_id)
                    if future is not None and not future.done():
                        future.set_result(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(exc)

    async def _stderr_loop(self) -> None:
        process = self._require_process()
        assert process.stderr is not None
        try:
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
        except asyncio.CancelledError:
            raise

    async def _read_message(
        self,
        stream: asyncio.StreamReader,
    ) -> dict[str, Any] | None:
        while b"\r\n\r\n" not in self._read_buffer:
            chunk = await stream.read(4096)
            if not chunk:
                return None
            self._read_buffer.extend(chunk)

        header_raw, body = self._read_buffer.split(b"\r\n\r\n", 1)
        self._read_buffer = bytearray(body)
        headers = header_raw.decode("ascii", errors="replace").split("\r\n")
        content_length = 0
        for header in headers:
            name, _, value = header.partition(":")
            if name.lower() == "content-length":
                content_length = int(value.strip())
                break

        while len(self._read_buffer) < content_length:
            chunk = await stream.read(4096)
            if not chunk:
                return None
            self._read_buffer.extend(chunk)

        payload = bytes(self._read_buffer[:content_length])
        del self._read_buffer[:content_length]
        data = json.loads(payload.decode("utf-8"))
        return data if isinstance(data, dict) else None

    def _resolve_cwd(self) -> Path:
        if self._config.cwd:
            raw = Path(self._config.cwd).expanduser()
            return raw if raw.is_absolute() else (self._base_cwd / raw).resolve()
        return self._base_cwd

    def _require_process(self) -> asyncio.subprocess.Process:
        if self._process is None:
            raise ToolError("MCP server is not connected")
        return self._process


class McpManager:
    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd
        self.config_path = cwd / DEFAULT_CONFIG_PATH
        self.cache_path = cwd / DEFAULT_CACHE_PATH
        self.server_states: dict[str, McpServerState] = {}
        self.cache: dict[str, list[McpToolSpec]] = {}
        self._registered_direct_tools: set[str] = set()
        self._load_cache()

    def load_config(self) -> dict[str, McpServerConfig]:
        if not self.config_path.exists():
            return {}
        try:
            payload = json.loads(self.config_path.read_text())
        except Exception:
            return {}

        servers_raw = payload.get("mcpServers")
        if not isinstance(servers_raw, dict):
            return {}

        configs: dict[str, McpServerConfig] = {}
        for server_name, value in servers_raw.items():
            if not isinstance(server_name, str) or not isinstance(value, dict):
                continue
            command = value.get("command")
            if not isinstance(command, str) or not command.strip():
                continue
            args_raw = value.get("args", [])
            env_raw = value.get("env", {})
            configs[server_name] = McpServerConfig(
                name=server_name,
                command=command,
                args=[str(item) for item in args_raw] if isinstance(args_raw, list) else [],
                env={str(k): str(v) for k, v in env_raw.items()}
                if isinstance(env_raw, dict)
                else {},
                cwd=str(value["cwd"]) if isinstance(value.get("cwd"), str) else None,
                direct_tools=_parse_direct_tools_setting(value.get("directTools")),
            )

        for name, config in configs.items():
            self.server_states.setdefault(name, McpServerState(config=config))
            self.server_states[name].config = config
        for name in list(self.server_states):
            if name not in configs:
                self.server_states.pop(name)
                self.cache.pop(name, None)
        return configs

    async def close(self) -> None:
        for state in self.server_states.values():
            if state.client is not None:
                await state.client.close()
                state.client = None
                state.connected = False

    async def execute_proxy(self, params: McpProxyParams) -> str:
        self.load_config()

        if params.connect:
            tools = await self.connect(params.connect)
            return self._format_tool_listing(params.connect, tools)

        if params.search:
            return self.search(params.search, server=params.server)

        if params.describe:
            tool = self.resolve_tool(params.describe, server=params.server)
            if tool is None:
                return self._tool_not_found(params.describe)
            return self.describe(tool)

        if params.tool:
            arguments = _parse_args_json(params.args)
            tool = self.resolve_tool(params.tool, server=params.server)
            if tool is None:
                inferred_server = params.server or self._infer_server_name(params.tool)
                if inferred_server is not None:
                    await self.connect(inferred_server)
                    tool = self.resolve_tool(params.tool, server=inferred_server)
            if tool is None:
                return self._tool_not_found(params.tool)
            return await self.call(tool, arguments)

        if params.server:
            return self._format_tool_listing(params.server, self.cache.get(params.server, []))

        return self.status_text()

    async def connect(self, server_name: str) -> list[McpToolSpec]:
        configs = self.load_config()
        if server_name not in configs:
            raise ToolError(f'MCP server "{server_name}" not found')
        state = self.server_states[server_name]

        if state.client is None:
            state.client = McpStdioClient(state.config, self.cwd)
        try:
            await state.client.connect()
            state.connected = True
            state.last_error = None
            tools = await state.client.list_tools()
            self.cache[server_name] = tools
            self._persist_cache()
            return tools
        except Exception as exc:
            state.connected = False
            state.last_error = str(exc)
            raise ToolError(f'Failed to connect to "{server_name}": {exc}') from exc

    def sync_direct_tools(self, ctx) -> int:
        configs = self.load_config()
        registered = 0
        for server_name, config in configs.items():
            wanted = config.direct_tools
            if not wanted:
                continue

            allowed: set[str] | None = None
            if isinstance(wanted, list):
                allowed = {_normalize_name(name) for name in wanted if str(name).strip()}

            for spec in self.cache.get(server_name, []):
                if allowed is not None and _normalize_name(spec.name) not in allowed:
                    continue
                if spec.prefixed_name in self._registered_direct_tools:
                    continue
                ctx.tools.register(DirectMcpTool(self, spec))
                self._registered_direct_tools.add(spec.prefixed_name)
                registered += 1
        return registered

    async def call(self, tool: McpToolSpec, arguments: dict[str, Any]) -> str:
        await self.connect(tool.server_name)
        state = self.server_states[tool.server_name]
        assert state.client is not None
        try:
            return await state.client.call_tool(tool.name, arguments)
        except Exception as exc:
            state.last_error = str(exc)
            raise

    def search(self, query: str, server: str | None = None) -> str:
        query = query.strip().lower()
        if not query:
            return "Search query cannot be empty"

        results: list[McpToolSpec] = []
        for tool in self._all_cached_tools():
            if server and tool.server_name != server:
                continue
            haystacks = [tool.prefixed_name.lower(), tool.name.lower(), tool.description.lower()]
            if any(query in value for value in haystacks):
                results.append(tool)

        if not results:
            return f'No MCP tools matched "{query}"'

        lines: list[str] = []
        for tool in sorted(results, key=lambda item: item.prefixed_name):
            lines.append(tool.prefixed_name)
            if tool.description:
                lines.append(f"  {tool.description}")
            if tool.input_schema:
                props = tool.input_schema.get("properties")
                if isinstance(props, dict) and props:
                    lines.append("  Parameters:")
                    for name, schema in props.items():
                        if isinstance(schema, dict):
                            schema_type = schema.get("type", "any")
                            desc = str(schema.get("description") or "").strip()
                            suffix = f" - {desc}" if desc else ""
                            lines.append(f"    {name} ({schema_type}){suffix}")
            lines.append("")
        return "\n".join(lines).strip()

    def describe(self, tool: McpToolSpec) -> str:
        lines = [
            tool.prefixed_name,
            f"Server: {tool.server_name}",
        ]
        if tool.description:
            lines.extend(["", tool.description])
        if tool.input_schema:
            props = tool.input_schema.get("properties")
            required = (
                set(tool.input_schema.get("required", []))
                if isinstance(tool.input_schema, dict)
                else set()
            )
            if isinstance(props, dict) and props:
                lines.extend(["", "Parameters:"])
                for name, schema in props.items():
                    if not isinstance(schema, dict):
                        continue
                    schema_type = schema.get("type", "any")
                    desc = str(schema.get("description") or "").strip()
                    required_text = " required" if name in required else ""
                    suffix = f" - {desc}" if desc else ""
                    lines.append(f"- {name} ({schema_type}{required_text}){suffix}")
        return "\n".join(lines)

    def resolve_tool(self, name: str, server: str | None = None) -> McpToolSpec | None:
        normalized = _normalize_name(name)
        matches: list[McpToolSpec] = []
        for tool in self._all_cached_tools():
            if server and tool.server_name != server:
                continue
            if normalized in {_normalize_name(tool.prefixed_name), _normalize_name(tool.name)}:
                matches.append(tool)

        if server and not matches:
            return None
        if len(matches) == 1:
            return matches[0]

        exact_prefixed = [
            tool for tool in matches if _normalize_name(tool.prefixed_name) == normalized
        ]
        if len(exact_prefixed) == 1:
            return exact_prefixed[0]
        return None

    def status_text(self) -> str:
        configs = self.load_config()
        if not configs:
            return f"No MCP servers configured. Create {self.config_path}"

        lines = ["MCP Servers"]
        for name in sorted(configs):
            state = self.server_states.get(name)
            status = "connected" if state and state.connected else "idle"
            cached = len(self.cache.get(name, []))
            lines.append(f"- {name} [{status}] {cached} cached tools")
            if state and state.last_error:
                lines.append(f"  last error: {state.last_error}")
        return "\n".join(lines)

    def _tool_not_found(self, name: str) -> str:
        return f'Tool "{name}" not found. Use mcp({{ search: "..." }}) or /mcp search <query>.'

    def _infer_server_name(self, tool_name: str) -> str | None:
        normalized = _normalize_name(tool_name)
        configs = self.load_config()
        for server_name in configs:
            prefix = _normalize_name(server_name) + "_"
            if normalized.startswith(prefix):
                return server_name
        return None

    def _all_cached_tools(self) -> list[McpToolSpec]:
        tools: list[McpToolSpec] = []
        for server_tools in self.cache.values():
            tools.extend(server_tools)
        return tools

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text())
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        servers = payload.get("servers")
        if not isinstance(servers, dict):
            return
        for server_name, tools_raw in servers.items():
            if not isinstance(server_name, str) or not isinstance(tools_raw, list):
                continue
            parsed: list[McpToolSpec] = []
            for item in tools_raw:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                parsed.append(
                    McpToolSpec(
                        server_name=server_name,
                        name=name,
                        description=str(item.get("description") or "").strip(),
                        input_schema=item.get("input_schema")
                        if isinstance(item.get("input_schema"), dict)
                        else {},
                    )
                )
            self.cache[server_name] = parsed

    def _persist_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "servers": {
                server_name: [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
                    for tool in tools
                ]
                for server_name, tools in self.cache.items()
            }
        }
        self.cache_path.write_text(json.dumps(payload, indent=2) + "\n")

    def _format_tool_listing(self, server_name: str, tools: list[McpToolSpec]) -> str:
        if server_name not in self.server_states:
            return f'MCP server "{server_name}" not found'
        if not tools:
            return f'Server "{server_name}" has no cached tools'
        lines = [f"Tools for {server_name}"]
        for tool in sorted(tools, key=lambda item: item.prefixed_name):
            lines.append(f"- {tool.prefixed_name}")
            if tool.description:
                lines.append(f"  {tool.description}")
        return "\n".join(lines)


class ManagerRegistry:
    def __init__(self) -> None:
        self._managers: dict[Path, McpManager] = {}
        self._current_cwd: Path | None = None

    def activate(self, cwd: Path) -> McpManager:
        resolved = cwd.resolve()
        manager = self._managers.get(resolved)
        if manager is None:
            manager = McpManager(resolved)
            self._managers[resolved] = manager
        self._current_cwd = resolved
        return manager

    def current(self) -> McpManager | None:
        if self._current_cwd is None:
            return None
        return self._managers.get(self._current_cwd)

    async def shutdown(self, cwd: Path) -> None:
        resolved = cwd.resolve()
        manager = self._managers.pop(resolved, None)
        if manager is not None:
            await manager.close()
        if self._current_cwd == resolved:
            self._current_cwd = None


MANAGERS = ManagerRegistry()


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _interpolate_env(raw: dict[str, str]) -> dict[str, str]:
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    def replace(value: str) -> str:
        return pattern.sub(lambda match: os.environ.get(match.group(1), ""), value)

    return {key: replace(value) for key, value in raw.items()}


def _parse_args_json(raw: str | None) -> dict[str, Any]:
    if raw is None or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError(f"Invalid args JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ToolError("Invalid args: expected a JSON object")
    return payload


def _parse_direct_tools_setting(raw: object) -> bool | list[str]:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, list):
        values = [str(item).strip() for item in raw if str(item).strip()]
        return values if values else False
    return False


def _python_type_from_schema(schema: dict[str, Any]) -> object:
    schema_type = schema.get("type")
    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list[Any]
    if schema_type == "object":
        return dict[str, Any]
    return Any


def _model_from_schema(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    properties = schema.get("properties")
    required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
    if not isinstance(properties, dict) or not properties:
        return create_model(f"{_normalize_name(tool_name).title()}Params")

    fields: dict[str, tuple[object, Any]] = {}
    for name, item in properties.items():
        if not isinstance(name, str) or not isinstance(item, dict):
            continue
        annotation = _python_type_from_schema(item)
        description = str(item.get("description") or "").strip() or None
        if name in required:
            default = Field(..., description=description)
        else:
            default = Field(default=None, description=description)
        fields[name] = (annotation, default)

    if not fields:
        return create_model(f"{_normalize_name(tool_name).title()}Params")
    return create_model(f"{_normalize_name(tool_name).title()}Params", **fields)


def _stringify_tool_content(content: object) -> str:
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
                continue
        parts.append(json.dumps(item, sort_keys=True))
    return "\n".join(part for part in parts if part).strip()


async def _mcp_command(args: str, ctx) -> str:
    manager = MANAGERS.activate(ctx.cwd)
    if ctx.ui is not None:
        ctx.ui.set_widget("footer", McpStatusWidget(manager))
    manager.sync_direct_tools(ctx)

    stripped = args.strip()
    if not stripped:
        if ctx.ui is not None:
            await ctx.ui.present(
                McpBrowserView(manager, sync_direct_tools=lambda: manager.sync_direct_tools(ctx))
            )
            return "Closed MCP browser"
        return manager.status_text()

    if stripped == "status":
        return manager.status_text()

    command, _, remainder = stripped.partition(" ")
    remainder = remainder.strip()

    if command == "reconnect" or command == "connect":
        if not remainder:
            return "Usage: /mcp connect <server>"
        tools = await manager.connect(remainder)
        manager.sync_direct_tools(ctx)
        return manager._format_tool_listing(remainder, tools)

    if command == "tools":
        if not remainder:
            return "Usage: /mcp tools <server>"
        return manager._format_tool_listing(remainder, manager.cache.get(remainder, []))

    if command == "search":
        if not remainder:
            return "Usage: /mcp search <query>"
        return manager.search(remainder)

    if command == "describe":
        if not remainder:
            return "Usage: /mcp describe <tool>"
        tool = manager.resolve_tool(remainder)
        if tool is None:
            return manager._tool_not_found(remainder)
        return manager.describe(tool)

    return "Usage: /mcp [status|connect|reconnect|tools|search|describe]"


def _on_context_event(event, ctx) -> None:
    manager = MANAGERS.activate(ctx.cwd)
    manager.sync_direct_tools(ctx)
    if ctx.ui is not None:
        ctx.ui.set_widget("footer", McpStatusWidget(manager))


async def _on_session_end(event, ctx) -> None:
    if ctx.ui is not None:
        ctx.ui.set_widget("footer", None)
    await MANAGERS.shutdown(ctx.cwd)


def setup(api: ExtensionAPI) -> None:
    api.register_tool(McpProxyTool())
    api.register_command("mcp", _mcp_command)
    api.on("session_start", _on_context_event)
    api.on("turn_start", _on_context_event)
    api.on("session_end", _on_session_end)
