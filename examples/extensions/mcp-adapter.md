# MCP Adapter Extension

This example adds MCP support without forcing every external tool into the prompt up front.

It always gives the agent a single `mcp` proxy tool, which keeps the prompt smaller and lets the agent discover and call MCP tools on demand.

It can also optionally promote cached MCP tools into the normal tool list when you enable `directTools` for a server.

## What this version does

- reads MCP server definitions from `.agent/mcp.json`
- connects to stdio MCP servers lazily
- caches tool metadata to `.agent/mcp-cache.json`
- registers one proxy tool named `mcp`
- optionally registers selected cached MCP tools directly into the main tool list
- adds a `/mcp` command for status, connect, tools, search, and describe
- shows a small footer status widget in the TUI
- opens a small presented MCP browser when you run `/mcp` in the TUI

## What this version does not do

It does not yet support:
- HTTP transports
- OAuth or bearer auth flows
- MCP resources, prompts, or UI payloads
- idle timeout and reconnect policies

In practical terms:
- it only knows how to launch local stdio MCP servers from `command` + `args`
- it cannot connect to remote MCP endpoints over HTTP or SSE
- it does not handle login, token exchange, or any other authenticated MCP flow
- it only works with MCP tools; it does not surface MCP resources or prompt catalogs
- it does not try to render MCP-provided UI payloads

## Load it

```bash
uv run agent run -e examples/extensions/mcp-adapter.py
```

## Config

Create `.agent/mcp.json` in the repo root:

```json
{
  "mcpServers": {
    "echo": {
      "command": "uv",
      "args": ["run", "python", "examples/mcp/echo_stdio_server.py"]
    }
  }
}
```

Each server currently supports:
- `command`
- `args`
- `env`
- `cwd`
- `directTools`

Environment values support simple `${VAR}` interpolation.

`directTools` can be:
- `false` or omitted: proxy-only for that server
- `true`: promote every cached tool from that server
- `["toolName"]`: promote only the named tools from that server

Example:

```json
{
  "mcpServers": {
    "echo": {
      "command": "uv",
      "args": ["run", "python", "examples/mcp/echo_stdio_server.py"],
      "directTools": true
    }
  }
}
```

## Included example server

The repo includes a tiny stdio MCP server you can use immediately:

- `examples/mcp/echo_stdio_server.py`

It exposes one tool:

- `echo`

Once configured, you can try:

```text
/mcp
/mcp connect echo
/mcp tools echo
/mcp describe echo_echo
```

Or through the proxy tool:

```text
mcp({ search: "echo" })
mcp({ tool: "echo_echo", args: "{\"text\":\"hello\"}" })
```

If `directTools` is enabled, the same server can also expose a native tool like `echo_echo` directly to the model after it has been connected and cached once.

## How discovery works

The model always sees one MCP-facing tool:

- `mcp`

If `directTools` is enabled and metadata is available in cache, it may also see selected MCP tools as separate first-class tools.

The normal flow is:

1. `mcp({ search: "query" })`
2. `mcp({ describe: "server_tool_name" })`
3. `mcp({ tool: "server_tool_name", args: "{\"key\":\"value\"}" })`

Tool names are shown in a prefixed form like `echo_my_tool` so the server identity stays clear.

Metadata is cached, so `search`, `describe`, and `tools` can still work from cache after a prior connection even if the server is not currently connected.

Direct tool promotion is also cache-backed. On a fresh repo with no cache yet, the server has to connect once before direct MCP tools can be promoted.

## Proxy tool usage

The proxy tool supports these shapes:

```text
mcp({ search: "query" })
mcp({ describe: "server_tool_name" })
mcp({ connect: "server-name" })
mcp({ tool: "server_tool_name", args: "{\"key\":\"value\"}" })
```

`args` must be a JSON object string.

If the tool name includes a server prefix like `echo_my_tool`, the adapter can infer which server to connect to on demand.

## Slash command usage

- `/mcp`
- `/mcp status`
- `/mcp connect <server>`
- `/mcp tools <server>`
- `/mcp search <query>`
- `/mcp describe <tool>`

In the TUI, `/mcp` with no args opens a small browser view where you can type:

- `server <name>`
- `connect <name>`
- `tools [name]`
- `search <query>`
- `describe <tool>`
- `status`
- `close`

In headless mode, `/mcp` with no args falls back to the plain text status output instead of opening the browser view.

## Why it exists

This is an advanced extension example showing how to bridge an external tool ecosystem into the agent without exploding the base tool list.
